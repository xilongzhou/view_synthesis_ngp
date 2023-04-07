from sched import scheduler
import torch
import torch.nn as nn
from torch.nn.functional import interpolate
from torch.optim import lr_scheduler, Adam
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.transforms.functional import gaussian_blur

import numpy as np
import cv2,os,time
from tqdm import tqdm


from models.xfields import XfieldsFlow
from util.args import TrainingArguments


from PIL import Image


from utils import VGGLoss, VGGpreprocess, saveimg

from network import VIINTER, linterp, Unet_Blend, MLP_blend, CondSIREN_meta, MySirenNet
import torch.nn.functional as F

import random

# from torchmeta.utils.gradient_based import gradient_update_parameters
from torchmeta.modules import MetaModule
from collections import OrderedDict

from my_dataloader import DataLoader_helper

from torch.nn.functional import interpolate, grid_sample

# torch.set_num_threads(1)


def gradient_update_parameters(model,
                               loss,
                               params=None,
                               step_size=0.5,
                               first_order=False):
    """Update of the meta-parameters with one step of gradient descent on the
    loss function.
    Parameters
    ----------
    model : `torchmeta.modules.MetaModule` instance
        The model.
    loss : `torch.Tensor` instance
        The value of the inner-loss. This is the result of the training dataset
        through the loss function.
    params : `collections.OrderedDict` instance, optional
        Dictionary containing the meta-parameters of the model. If `None`, then
        the values stored in `model.meta_named_parameters()` are used. This is
        useful for running multiple steps of gradient descent as the inner-loop.
    step_size : int, `torch.Tensor`, or `collections.OrderedDict` instance (default: 0.5)
        The step size in the gradient update. If an `OrderedDict`, then the
        keys must match the keys in `params`.
    first_order : bool (default: `False`)
        If `True`, then the first order approximation of MAML is used.
    Returns
    -------
    updated_params : `collections.OrderedDict` instance
        Dictionary containing the updated meta-parameters of the model, with one
        gradient update wrt. the inner-loss.
    """
    if not isinstance(model, MetaModule):
        raise ValueError('The model must be an instance of `torchmeta.modules.'
                         'MetaModule`, got `{0}`'.format(type(model)))

    if params is None:
        params = OrderedDict(model.meta_named_parameters())

    grads = torch.autograd.grad(loss,
                                params.values(),
                                create_graph=not first_order)

    updated_params = OrderedDict()

    if isinstance(step_size, (dict, OrderedDict)):
        for (name, param), grad in zip(params.items(), grads):
            updated_params[name] = param - step_size[name] * grad
    else:
        for (name, param), grad in zip(params.items(), grads):
            updated_params[name] = param - step_size * grad

    return updated_params


def init(args):

	torch.manual_seed(1234)
	np.random.seed(1234)
	savedir = os.path.join(args.savepath,os.path.split(args.dataset)[1])
	os.makedirs(os.path.join(savedir,"trained model"), exist_ok=True)
	os.makedirs(os.path.join(savedir,"saved training"), exist_ok=True)
	
	return savedir

def rgb_loss(out, images):

	L1 = nn.L1Loss()

	rgb_wrap_loss = L1(images, out)# + (1/2) * L1(rgbout2, gt2) + (1/4) * L1(rgbout4, gt4) + (1/8) * L1(rgbout8, gt8)


	return rgb_wrap_loss, 0, 0, 0


def merge_planes(planes, gtflow):

	# abflow = torch.abs(gtflow[:, :1]).detach()
	# shift_mask1 = (abflow <= 0.0156)
	# shift_mask2 = torch.logical_and(abflow > 0.0156, abflow < 0.0468)
	# shift_mask3 = (abflow >= 0.0468)
	# shifts = [shift_mask1, shift_mask2, shift_mask3]
	
	# planes = [p * s for p,s in zip(planes, shifts)]

	# plane1 = torch.clamp(planes[0], max=0.0156)
	# plane2 = torch.clamp(planes[1], max=0.0468)
	# plane3 = planes[2]#torch.clamp(planes[2], min=0.0468)
	# plane2 = planes[1]
	# planes = [plane1, plane2, plane3]
	# planes = [plane1, plane2]

	# planes = [p * s for p,s in zip(planes, shifts)]
	out = planes[0]

	# print("=======")
	for index in range(1, len(planes)):
		out = torch.maximum(out, planes[index])
		# print(torch.max(planes[index]), torch.min(planes[index]))
	# print("=======")
	# out = sum(planes)
	
	return out


def get_planes(max_disp, num_planes, w_res):

	max_disp = 2*int(max_disp * w_res / 2 + 1)
	
	planes = np.rint(np.linspace(0, max_disp, num_planes)/2)*2

	xp = np.arange(2 * num_planes + 1)

	edges = np.interp(xp[0::2], xp[1::2], planes) / w_res
	edges[-1] = 1.0

	bounds = np.array([edges[:-1], edges[1:]]).T

	return (planes//2).astype(int), bounds, max_disp//2


def save_args(args):
	args = vars(args)
	if not os.path.exists( args["savepath"]):         
		os.makedirs( args["savepath"])
	file_name = os.path.join( args["savepath"], 'opt.txt')
	with open(file_name, 'wt') as opt_file:
		opt_file.write('------------ Options -------------\n')
		for k, v in sorted(args.items()):
			opt_file.write('%s: %s\n' % (str(k), str(v)))
		opt_file.write('-------------- End ----------------\n')
	return 


def meta_learn(args):

	device = "cuda"

	save_imgpath = os.path.join(args.savepath, "imgs")
	save_modelpath = os.path.join(args.savepath, "ckpt")

	if not os.path.exists(save_imgpath):
		os.makedirs(save_imgpath)

	if not os.path.exists(save_modelpath):
		os.makedirs(save_modelpath)

	h_res = args.resolution[0]
	w_res = args.resolution[1]

	# hard-code max_disp
	max_disp = args.max_disp
	planes = np.rint(np.linspace(0, max_disp, args.num_planes)/2)*2
	offsets = (planes//2).astype(int)
	base_shift = max_disp//2
	print(f"offset: {offsets}, base_shift: {base_shift}")

	# ----------------------- define network --------------------------------------
	if args.net=="conv":
		L_tag = torch.tensor([0.03]).cuda().unsqueeze(0).unsqueeze(0).unsqueeze(0).to(torch.float32)
		R_tag = torch.tensor([0.06]).cuda().unsqueeze(0).unsqueeze(0).unsqueeze(0).to(torch.float32)

		flow_model  = XfieldsFlow(h_res, w_res + base_shift * 2, inChannels=1, ngf=args.nfg, outChannels= 3*args.num_planes).cuda()
	elif args.net=="mlp":

		L_tag = torch.tensor([0]).cuda()
		R_tag = torch.tensor([1]).cuda()

		if args.use_mynet:
			net_meta = MySirenNet(2, 256, 3, 5, w0_initial=200., w0=200., final_activation=lambda x: x + .5).cuda()
		else:
			net_meta = CondSIREN_meta(n_emb = 2, norm_p = 1, inter_fn=linterp, D=args.n_layer, z_dim = 128, in_feat=2, out_feat=3*args.num_planes, W=args.n_c, with_res=False, with_norm=args.use_norm).cuda()

		if args.blendnn_type=="unet":
			net_blend = Unet_Blend(3*args.num_planes, 3, 4, (h_res, w_res)).cuda()
		elif args.blendnn_type=="mlp":
			net_blend = MLP_blend(D=args.mlp_d, in_feat=3*args.num_planes, out_feat=3, W=args.mlp_w, with_res=False, with_norm=args.use_norm).cuda()


		coords_h = np.linspace(-1, 1, h_res, endpoint=False)
		coords_w = np.linspace(-1, 1,  w_res + base_shift * 2, endpoint=False)
		# coords_w = np.linspace(-1, 1,  w_res, endpoint=False)
		xy_grid = np.stack(np.meshgrid(coords_w, coords_h), -1)
		xy_grid = torch.FloatTensor(xy_grid).cuda()
		# if not args.rowbatch:
		grid_inp = xy_grid.view(-1, 2).contiguous().unsqueeze(0)

		dx = torch.from_numpy(coords_w).float()
		dy = torch.from_numpy(coords_h).float()
		meshy, meshx = torch.meshgrid((dy, dx))

	print("net_meta: ", net_meta)

	nparams_decoder = sum(p.numel() for p in net_meta.parameters() if p.requires_grad)
	print('Number of learnable parameters (decoder): %d' %(nparams_decoder))

	nparams_nnblend = sum(p.numel() for p in net_blend.parameters() if p.requires_grad)
	print('Number of learnable parameters (blend nn): %d' %(nparams_nnblend))

	if args.net=="mlp":
		print("optimize both flow model and net blend")
		optimizer = torch.optim.Adam(list(net_meta.parameters()) + list(net_blend.parameters()), lr=args.lr)
		# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_iters)
	elif args.net=="conv":
		optimizer = Adam(params=flow_model.parameters(), lr=args.lr)
		scheduler = lr_scheduler.StepLR(optimizer, step_size=int(args.num_iters*0.5), gamma=0.5)


	# ---------------------- Dataloader ----------------------
	Myset = DataLoader_helper(args.datapath, h_res, w_res)
	Mydata = DataLoader(dataset=Myset, batch_size=args.batch_size, shuffle=True, drop_last=True)

	# for debug
	# imgR_path = "./fall/0001R.png"
	# imgL_path = "./fall/0001L.png"
	# imgR_pil = Image.open(imgR_path).convert("RGB")
	# imgL_pil = Image.open(imgL_path).convert("RGB")
	# imgR_pil = imgR_pil.resize((w_res, h_res), Image.LANCZOS )
	# imgL_pil = imgL_pil.resize((w_res, h_res), Image.LANCZOS )
	# imgR = torch.from_numpy(np.array(imgR_pil)).to(device).permute(2,0,1).unsqueeze(0)/255.
	# imgL = torch.from_numpy(np.array(imgL_pil)).to(device).permute(2,0,1).unsqueeze(0)/255.
	# imgtest = imgL

	# for metric
	metric_l1 = nn.L1Loss()
	metric_mse = nn.MSELoss()


	xp = [0, 1]
	fp = [-2, 2]

	iter = 0

	# for metric
	if args.w_vgg>0:
		metric_vgg = VGGLoss()

	while iter < args.num_iters:

		# this is for outerloop
		for i,data in enumerate(Mydata):

			optimizer.zero_grad()
			
			imgL_b, imgR_b, imgtest_b, inter_val_b = data
			outer_loss = torch.tensor(0.).cuda()

			mseloss = torch.tensor(0.).cuda()
			vgloss = torch.tensor(0.).cuda()

			for task_id in range(args.batch_size):

				imgL = imgL_b[task_id:task_id+1].to(device)
				imgR = imgR_b[task_id:task_id+1].to(device)
				imgtest = imgtest_b[task_id:task_id+1].to(device)
				inter = inter_val_b[task_id:task_id+1]

				if args.net=="conv":
					out_L = (1 + flow_model(L_tag))*0.5
					out_R = (1 + flow_model(R_tag))*0.5

					multi_out_L = [out_L[:, 3*i:3*i+3, :, base_shift - offsets[i]:base_shift + w_res - offsets[i]] for i in range(len(offsets))]
					multi_out_R = [out_R[:, 3*i:3*i+3, :, base_shift - offsets[i]:base_shift + w_res - offsets[i]] for i in range(len(offsets))]

					blend_L = None
					blend_R = None
					for l in range(args.num_planes):

						if blend_L is None:
							blend_L = multi_out_L[l]*shift_mask_L[l]
						else:
							blend_L += multi_out_L[l]*shift_mask_L[l]

						if blend_R is None:
							blend_R = multi_out_R[l]*shift_mask_R[l]
						else:
							blend_R += multi_out_R[l]*shift_mask_R[l]

					out_L = blend_L
					out_R = blend_R

					l1loss = metric_l1(out_L, imgL)*args.w_l1 + metric_l1(out_R, imgR)*args.w_l1
					loss = l1loss + vgloss
				
				elif args.net=="mlp":

					if args.debug:
						if random.choice([True, False]):
							use_left = True
							tag = L_tag
							gt_img = imgL
						else:
							use_left = False
							tag = R_tag
							gt_img = imgR

					# for inner loop
					for inner_step in range(args.inner_steps):

						if args.debug:
							out = net_meta(grid_inp, tag)
							out = out.reshape(1, h_res, -1, 3*args.num_planes).permute(0,3,1,2)
							if use_left:
								multi_out = [out[:, 3*i:3*i+3, :, base_shift - offsets[i]:base_shift + w_res - offsets[i]] for i in range(len(offsets))]
							else:
								multi_out = [out[:, 3*i:3*i+3, :, base_shift + offsets[i]:base_shift + w_res + offsets[i]] for i in range(len(offsets))]
							multi_out = torch.cat(multi_out,dim=1)
							blend = net_blend(2*multi_out-1)
							inner_loss = metric_mse(blend, gt_img)

						else:

							# left
							out = net_meta(grid_inp, L_tag) if not args.use_mynet else net_meta(grid_inp, params=OrderedDict(net_meta.meta_named_parameters()))
							out = out.reshape(1, h_res, -1, 3*args.num_planes).permute(0,3,1,2)
							multi_out = [out[:, 3*i:3*i+3, :, base_shift - offsets[i]:base_shift + w_res - offsets[i]] for i in range(len(offsets))]
							multi_out = torch.cat(multi_out,dim=1)
							blend_l = net_blend(2*multi_out-1)
							inner_loss_l = metric_mse(blend_l, imgL)

							# right
							out = net_meta(grid_inp, R_tag) if not args.use_mynet else net_meta(grid_inp, params=OrderedDict(net_meta.meta_named_parameters()))
							out = out.reshape(1, h_res, -1, 3*args.num_planes).permute(0,3,1,2)
							multi_out = [out[:, 3*i:3*i+3, :, base_shift + offsets[i]:base_shift + w_res + offsets[i]] for i in range(len(offsets))]
							multi_out = torch.cat(multi_out,dim=1)
							blend_r = net_blend(2*multi_out-1)
							inner_loss_r = metric_mse(blend_r, imgR)

							inner_loss = (inner_loss_l + inner_loss_r) * 0.5

						params_meta = gradient_update_parameters(net_meta, inner_loss, step_size=args.inner_lr, first_order=args.first_order)

					# for debug
					if args.in_out_same:

						if args.debug:

							out = net_meta(grid_inp, tag, params=params_meta)
							out = out.reshape(1, h_res, -1, 3*args.num_planes).permute(0,3,1,2)
							if use_left:
								multi_out = [out[:, 3*i:3*i+3, :, base_shift - offsets[i]:base_shift + w_res - offsets[i]] for i in range(len(offsets))]
							else:
								multi_out = [out[:, 3*i:3*i+3, :, base_shift + offsets[i]:base_shift + w_res + offsets[i]] for i in range(len(offsets))]
							multi_out = torch.cat(multi_out,dim=1)
							blend_te = net_blend(2*multi_out-1)
							outer_loss = metric_mse(blend_te, gt_img)

						else:
							# left
							out = net_meta(grid_inp, L_tag, params=params_meta) if not args.use_mynet else net_meta(grid_inp, params=params_meta)
							out = out.reshape(1, h_res, -1, 3*args.num_planes).permute(0,3,1,2)
							multi_out = [out[:, 3*i:3*i+3, :, base_shift - offsets[i]:base_shift + w_res - offsets[i]] for i in range(len(offsets))]
							multi_out = torch.cat(multi_out,dim=1)
							blend_l_out = net_blend(2*multi_out-1)

							# right
							out = net_meta(grid_inp, R_tag, params=params_meta) if not args.use_mynet else net_meta(grid_inp, params=params_meta)
							out = out.reshape(1, h_res, -1, 3*args.num_planes).permute(0,3,1,2)
							multi_out = [out[:, 3*i:3*i+3, :, base_shift + offsets[i]:base_shift + w_res + offsets[i]] for i in range(len(offsets))]
							multi_out = torch.cat(multi_out,dim=1)
							blend_r_out = net_blend(2*multi_out-1)

							outer_loss = (metric_mse(blend_l_out, imgL) + metric_mse(blend_r_out, imgR)) * 0.5

					else:
						# compute outer loss
						z0 = net_meta.ret_z(torch.LongTensor([0.]).cuda(), params=params_meta).squeeze()
						z1 = net_meta.ret_z(torch.LongTensor([1.]).cuda(), params=params_meta).squeeze()
						zi = linterp(inter.to(device), z0, z1).unsqueeze(0)
						out = net_meta.forward_with_z(grid_inp, zi, params=params_meta)
						out = out.reshape(1, h_res, -1, 3*args.num_planes).permute(0,3,1,2)
						if max_disp==0:
							multi_out = [out[:, 3*i:3*i+3, :, base_shift - offsets[i]:base_shift + w_res - offsets[i]] for i in range(len(offsets))]
							multi_out = torch.cat(multi_out,dim=1)
						else:
							off_scl = np.interp(inter, xp, fp)
							multi_out_list=[]
							for i in range(args.num_planes):
								meshxw = meshx + (base_shift * 2 + off_scl * offsets[i]) / (w_res + base_shift * 2)
								grid = torch.stack((meshxw, meshy), 2)[None].to(device).to(torch.float32)
								multi_out = grid_sample(out[:, 3*i:3*i+3], grid, mode='bilinear', align_corners=True)[:, :, :, :-base_shift*2]
								multi_out_list.append(multi_out)
							multi_out = torch.cat(multi_out_list,dim=1)
						blend_test = net_blend(2*multi_out-1)
						mseloss += metric_mse(blend_test, imgtest)

						if args.w_vgg>0:
							blend_test_vg = VGGpreprocess(blend_test)
							imgtest_vg = VGGpreprocess(imgtest)
							vgloss += metric_vgg(blend_test_vg, imgtest_vg) * args.w_vgg

			mseloss.div_(args.batch_size)
			vgloss.div_(args.batch_size)
			outer_loss = mseloss + vgloss
			outer_loss.backward()
			optimizer.step()
			# scheduler.step()

			if (iter % args.progress_iter) ==0:
				print(f"iterations: {iter}, outer_loss: {outer_loss}, mseloss: {mseloss}, vgloss: {vgloss} ")

				save_dict = {'state_dict':net_meta.state_dict(), \
							  'nn_blend':net_blend.state_dict(), \
							  'nfg':args.nfg, \
							  'n_layer':args.n_layer, \
							  'use_norm':args.use_norm, \
							  'n_c':args.n_c, \
							  'mlp_d':args.mlp_d, \
							  'mlp_w':args.mlp_w, \
							  'num_freqs_pe':args.num_freqs_pe, \
							  'inner_lr':args.inner_lr, \
							  'offsets':offsets, \
							  'base_shift':base_shift, \
							  'num_planes':args.num_planes, \
							  'resolution':args.resolution, \
							  'max_disp':max_disp }
				torch.save(save_dict, "%s/model.ckpt"%(save_modelpath))
				# save_progress() if not args.debug else save_progress_debug(model_out, gt_img, iter)

				if args.net=="conv":
					save_img(out_L, f"out_L_{iter}")
					save_img(out_R, f"out_R_{iter}")
					save_img(imgR[0], f"gt_R")
					save_img(imgL[0], f"gt_L")
					[save_img((shift_mask_L[i].expand(-1, 3, h_res, w_res))[0], "mask_L{}".format(i)) for i in range(len(offsets))]
					[save_img((shift_mask_R[i].expand(-1, 3, h_res, w_res))[0], "mask_R{}".format(i)) for i in range(len(offsets))]
				elif args.net =="mlp":

					if args.debug:
						saveimg(blend, f"{save_imgpath}/{iter}_out.png")
						saveimg(blend_te, f"{save_imgpath}/{iter}_out_te.png")
						saveimg(gt_img, f"{save_imgpath}/{iter}_gt.png")
					else:
						# save train image
						saveimg(blend_l, f"{save_imgpath}/{iter}_outl.png")
						saveimg(blend_r, f"{save_imgpath}/{iter}_outr.png")
						saveimg(imgL, f"{save_imgpath}/{iter}_gtl.png")
						saveimg(imgR, f"{save_imgpath}/{iter}_gtr.png")


						# save test image
						if not args.in_out_same:
							saveimg(blend_test, f"{save_imgpath}/{iter}_outtest.png")
							saveimg(imgtest[0], f"{save_imgpath}/{iter}_gttest.png")
                 

			iter +=1


def regular_train(args):
	device = "cuda"

	save_imgpath = os.path.join(args.savepath, "imgs")
	save_modelpath = os.path.join(args.savepath, "ckpt")

	if not os.path.exists(save_imgpath):
		os.makedirs(save_imgpath)

	if not os.path.exists(save_modelpath):
		os.makedirs(save_modelpath)

	h_res = args.resolution[0]
	w_res = args.resolution[1]

	# hard-code max_disp
	max_disp = args.max_disp
	planes = np.rint(np.linspace(0, max_disp, args.num_planes)/2)*2
	offsets = (planes//2).astype(int)
	base_shift = max_disp//2
	print(f"offset: {offsets}, base_shift: {base_shift}")

	# ----------------------- define network --------------------------------------
	if args.net=="conv":
		L_tag = torch.tensor([0.03]).cuda().unsqueeze(0).unsqueeze(0).unsqueeze(0).to(torch.float32)
		R_tag = torch.tensor([0.06]).cuda().unsqueeze(0).unsqueeze(0).unsqueeze(0).to(torch.float32)

		flow_model  = XfieldsFlow(h_res, w_res + base_shift * 2, inChannels=1, ngf=args.nfg, outChannels= 3*args.num_planes).cuda()
	elif args.net=="mlp":

		L_tag = torch.tensor([0]).cuda()
		R_tag = torch.tensor([1]).cuda()

		if args.use_mynet:
			net_meta = MySirenNet(2, 256, 3, 5, w0_initial=200., w0=200., final_activation=lambda x: x + .5).cuda()
		else:
			net_meta = CondSIREN_meta(n_emb = 2, norm_p = 1, inter_fn=linterp, D=args.n_layer, z_dim = 128, in_feat=2, out_feat=3*args.num_planes, W=args.n_c, with_res=False, with_norm=args.use_norm).cuda()

		if args.blendnn_type=="unet":
			net_blend = Unet_Blend(3*args.num_planes, 3, 4, (h_res, w_res)).cuda()
		elif args.blendnn_type=="mlp":
			net_blend = MLP_blend(D=args.mlp_d, in_feat=3*args.num_planes, out_feat=3, W=args.mlp_w, with_res=False, with_norm=args.use_norm).cuda()


		coords_h = np.linspace(-1, 1, h_res, endpoint=False)
		coords_w = np.linspace(-1, 1,  w_res + base_shift * 2, endpoint=False)
		# coords_w = np.linspace(-1, 1,  w_res, endpoint=False)
		xy_grid = np.stack(np.meshgrid(coords_w, coords_h), -1)
		xy_grid = torch.FloatTensor(xy_grid).cuda()
		# if not args.rowbatch:
		grid_inp = xy_grid.view(-1, 2).contiguous().unsqueeze(0)

		dx = torch.from_numpy(coords_w).float()
		dy = torch.from_numpy(coords_h).float()
		meshy, meshx = torch.meshgrid((dy, dx))

	print("net_meta: ", net_meta)

	nparams_decoder = sum(p.numel() for p in net_meta.parameters() if p.requires_grad)
	print('Number of learnable parameters (decoder): %d' %(nparams_decoder))

	nparams_nnblend = sum(p.numel() for p in net_blend.parameters() if p.requires_grad)
	print('Number of learnable parameters (blend nn): %d' %(nparams_nnblend))

	if args.net=="mlp":
		print("optimize both flow model and net blend")
		optimizer = torch.optim.Adam(list(net_meta.parameters()) + list(net_blend.parameters()), lr=args.lr)
		scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_iters, eta_min=args.lr*0.1)
	elif args.net=="conv":
		optimizer = Adam(params=flow_model.parameters(), lr=args.lr)
		scheduler = lr_scheduler.StepLR(optimizer, step_size=int(args.num_iters*0.5), gamma=0.5)


	# ---------------------- Dataloader ----------------------
	Myset = DataLoader_helper(args.datapath, h_res, w_res)
	Mydata = DataLoader(dataset=Myset, batch_size=1, shuffle=True, drop_last=True)

	# for metric
	metric_l1 = nn.L1Loss()
	metric_mse = nn.MSELoss()

	# for metric
	if args.w_vgg>0:
		metric_vgg = VGGLoss()



	mseloss = 0
	vgloss = 0

	xp = [0, 1]
	fp = [-2, 2]

	iter = 0

	while iter < args.num_iters:

		# this is for outerloop
		for i,data in enumerate(Mydata):

			optimizer.zero_grad()
			
			imgL_b, imgR_b, imgtest_b, inter_val_b, _ = data

			imgtest = imgtest_b.to(device)
			inter = inter_val_b


			# compute outer loss
			z0 = net_meta.ret_z(torch.LongTensor([0.]).cuda()).squeeze()
			z1 = net_meta.ret_z(torch.LongTensor([1.]).cuda()).squeeze()
			zi = linterp(inter.to(device), z0, z1).unsqueeze(0)
			out = net_meta.forward_with_z(grid_inp, zi)
			out = out.reshape(1, h_res, -1, 3*args.num_planes).permute(0,3,1,2)
			if max_disp==0:
				multi_out = [out[:, 3*i:3*i+3, :, base_shift - offsets[i]:base_shift + w_res - offsets[i]] for i in range(len(offsets))]
				multi_out = torch.cat(multi_out,dim=1)
			else:
				off_scl = np.interp(inter, xp, fp)
				multi_out_list=[]
				for i in range(args.num_planes):
					meshxw = meshx + (base_shift * 2 + off_scl * offsets[i]) / (w_res + base_shift * 2)
					grid = torch.stack((meshxw, meshy), 2)[None].to(device).to(torch.float32)
					multi_out = grid_sample(out[:, 3*i:3*i+3], grid, mode='bilinear', align_corners=True)[:, :, :, :-base_shift*2]
					multi_out_list.append(multi_out)
				multi_out = torch.cat(multi_out_list,dim=1)
			blend_test = net_blend(2*multi_out-1)
			mseloss = metric_mse(blend_test, imgtest)

			if args.w_vgg>0:
				blend_test_vg = VGGpreprocess(blend_test)
				imgtest_vg = VGGpreprocess(imgtest)
				vgloss = metric_vgg(blend_test_vg, imgtest_vg) * args.w_vgg

			outer_loss = mseloss + vgloss

			outer_loss.backward()
			optimizer.step()
			scheduler.step()

			if (iter % args.progress_iter) ==0:
				print(f"iterations: {iter}, outer_loss: {outer_loss}, mseloss: {mseloss}, vgloss: {vgloss} ")

				save_dict = {'state_dict':net_meta.state_dict(), \
							  'nn_blend':net_blend.state_dict(), \
							  'nfg':args.nfg, \
							  'n_layer':args.n_layer, \
							  'use_norm':args.use_norm, \
							  'n_c':args.n_c, \
							  'mlp_d':args.mlp_d, \
							  'mlp_w':args.mlp_w, \
							  'num_freqs_pe':args.num_freqs_pe, \
							  'inner_lr':args.inner_lr, \
							  'offsets':offsets, \
							  'base_shift':base_shift, \
							  'num_planes':args.num_planes, \
							  'resolution':args.resolution, \
							  'max_disp':max_disp }
				torch.save(save_dict, "%s/model.ckpt"%(save_modelpath))
				# save_progress() if not args.debug else save_progress_debug(model_out, gt_img, iter)

				saveimg(blend_test, f"{save_imgpath}/{iter}_outtest.png")
				saveimg(imgtest[0], f"{save_imgpath}/{iter}_gttest.png")
                 
			iter +=1



	
if __name__=='__main__':
	
	args = TrainingArguments().parser.parse_args()
	print(args)

	save_args(args)

	if not os.path.exists(args.savepath):
		os.makedirs(args.savepath)

	if args.meta_learn:
		meta_learn(args)
	else:
		regular_train(args)