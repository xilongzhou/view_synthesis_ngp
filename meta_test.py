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

from network import VIINTER, linterp, Unet_Blend, MLP_blend, CondSIREN_meta
import torch.nn.functional as F

import random

# from torchmeta.utils.gradient_based import gradient_update_parameters
from torchmeta.modules import MetaModule
from collections import OrderedDict

from meta_dataloader import DataLoader_helper, DataLoader_helper_test

from torch.nn.functional import interpolate, grid_sample

import argparse

import imageio


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


## assign the grad of source to target
def write_params(model, param_dict):

	for name, param in model.named_parameters():
		param.data.copy_(param_dict[name])


def meta_test(args):


	device = "cuda"

	load_path = os.path.join(args.load_path, "ckpt")
	save_path = os.path.join(args.load_path, f"{args.save_name}")
	if not os.path.exists(save_path):
		os.makedirs(save_path)

	saved_dict = torch.load("%s/model.ckpt"%(load_path))
	offsets = saved_dict['offsets']
	base_shift = saved_dict['base_shift']
	num_planes = saved_dict['num_planes']

	if "nfg" in saved_dict:
		args.nfg = saved_dict['nfg']

	if "max_disp" in saved_dict:
		args.max_disp = saved_dict['max_disp']

	if "n_layer" in saved_dict:
		args.n_layer = saved_dict['n_layer']

	if "mlp_d" in saved_dict:
		args.mlp_d = saved_dict['mlp_d']

	if "mlp_w" in saved_dict:
		args.mlp_w = saved_dict['mlp_w']

	if "resolution" in saved_dict:
		args.resolution = saved_dict['resolution']

	if "num_freqs_pe" in saved_dict:
		args.num_freqs_pe = saved_dict['num_freqs_pe']

	if "inner_lr" in saved_dict:
		args.inner_lr = saved_dict['inner_lr']

	if "n_c" in saved_dict:
		args.n_c = saved_dict['n_c']

	if "use_norm" in saved_dict:
		args.use_norm = saved_dict['use_norm']

	h_res = args.resolution[0]
	w_res = args.resolution[1]


	# ----------------------- define network --------------------------------------
	if args.net=="conv":
		L_tag = torch.tensor([0.03]).cuda().unsqueeze(0).unsqueeze(0).unsqueeze(0).to(torch.float32)
		R_tag = torch.tensor([0.06]).cuda().unsqueeze(0).unsqueeze(0).unsqueeze(0).to(torch.float32)

		flow_model  = XfieldsFlow(h_res, w_res + base_shift * 2, inChannels=1, ngf=args.nfg, outChannels= 3* num_planes).cuda()
	elif args.net=="mlp":

		L_tag = torch.tensor([0]).cuda()
		R_tag = torch.tensor([1]).cuda()

		net_meta = CondSIREN_meta(n_emb = 2, norm_p = 1, inter_fn=linterp, D=args.n_layer, z_dim = 128, in_feat=2, out_feat=3*num_planes, W=args.n_c, with_res=False, with_norm=args.use_norm).cuda()

		net_blend = MLP_blend(D=args.mlp_d, in_feat=3*num_planes, out_feat=3, W=args.mlp_w, with_res=False, with_norm=args.use_norm).cuda()

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

	# ---------------------- Dataloader ----------------------
	Myset = DataLoader_helper_test(args.test_dataset, h_res, w_res)
	Mydata = DataLoader(dataset=Myset, batch_size=1, shuffle=False, drop_last=True)

	metric_l1 = nn.L1Loss()
	metric_mse = nn.MSELoss()

	# for metric
	if args.w_vgg>0:
		metric_vgg = VGGLoss()

	xp = [0, 1]
	fp = [-2, 2]

	inter_list = [j for j in np.linspace(0,1,num=20)]
	print("inter_list: ", inter_list)


	# set up optimizer
	if args.no_blend_optim:
		optimizer = torch.optim.Adam(list(net_meta.parameters()), lr=args.lr)
	else:
		optimizer = torch.optim.Adam(list(net_meta.parameters()) + list(net_blend.parameters()), lr=args.lr)



	# this is for outerloop
	for i,data in enumerate(Mydata):
		
		imgL, imgR, name = data

		name = name[0]
		video_out = imageio.get_writer(os.path.join(save_path, f"{name}.mp4"), mode='I', fps=12, codec='libx264')

		imgL = imgL.to(device)
		imgR = imgR.to(device)

		# load network
		net_meta.load_state_dict(saved_dict['state_dict'])
		net_blend.load_state_dict(saved_dict['nn_blend'])

		print(f"for the scene {name}.......finish loading the model.......")

		net_meta.eval()
		net_blend.eval()

		if args.net=="conv":
			out_L = (1 + flow_model(L_tag))*0.5
			out_R = (1 + flow_model(R_tag))*0.5

			multi_out_L = [out_L[:, 3*i:3*i+3, :, base_shift - offsets[i]:base_shift + w_res - offsets[i]] for i in range(len(offsets))]
			multi_out_R = [out_R[:, 3*i:3*i+3, :, base_shift - offsets[i]:base_shift + w_res - offsets[i]] for i in range(len(offsets))]

			blend_L = None
			blend_R = None
			for l in range(num_planes):

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

			# for one step of inner loop
			for inner_step in range(1):

				# out = net_meta(grid_inp, tag)
				# out = out.reshape(1, h_res, -1, 3*args.num_planes).permute(0,3,1,2)
				# if use_left:
				# 	multi_out = [out[:, 3*i:3*i+3, :, base_shift - offsets[i]:base_shift + w_res - offsets[i]] for i in range(len(offsets))]
				# else:
				# 	multi_out = [out[:, 3*i:3*i+3, :, base_shift + offsets[i]:base_shift + w_res + offsets[i]] for i in range(len(offsets))]
				# multi_out = torch.cat(multi_out,dim=1)
				# blend = net_blend(2*multi_out-1)

				# left
				out = net_meta(grid_inp, L_tag)
				out = out.reshape(1, h_res, -1, 3*num_planes).permute(0,3,1,2)
				multi_out = [out[:, 3*i:3*i+3, :, base_shift - offsets[i]:base_shift + w_res - offsets[i]] for i in range(len(offsets))]
				multi_out = torch.cat(multi_out,dim=1)
				blend_l = net_blend(2*multi_out-1)
				inner_loss_l = metric_mse(blend_l, imgL)

				# right
				out = net_meta(grid_inp, R_tag)
				out = out.reshape(1, h_res, -1, 3*num_planes).permute(0,3,1,2)
				multi_out = [out[:, 3*i:3*i+3, :, base_shift + offsets[i]:base_shift + w_res + offsets[i]] for i in range(len(offsets))]
				multi_out = torch.cat(multi_out,dim=1)
				blend_r = net_blend(2*multi_out-1)
				inner_loss_r = metric_mse(blend_r, imgR)

				inner_loss = (inner_loss_l + inner_loss_r) * 0.5

				params_meta = gradient_update_parameters(net_meta, inner_loss, step_size=args.inner_lr, first_order=True)

				# params_meta2 = params_meta

				write_params(net_meta, params_meta)

				# for k in params_meta:
				# 	print(torch.sum(params_meta[k] - params_meta2[k]))

				# for name, param in net_meta.named_parameters():
				# 	if name not in params_meta:
				# 		print(name)
				# 	param.data.copy_(params_meta[name])

			# net_meta.eval()
			# net_blend.eval()

			# for more steps of optimization
			if args.total_step > 0:
				for i in range(args.total_step):

					# left
					out = net_meta(grid_inp, L_tag)
					out = out.reshape(1, h_res, -1, 3*num_planes).permute(0,3,1,2)
					multi_out = [out[:, 3*i:3*i+3, :, base_shift - offsets[i]:base_shift + w_res - offsets[i]] for i in range(len(offsets))]
					multi_out = torch.cat(multi_out,dim=1)
					blend_l = net_blend(2*multi_out-1)
					inner_loss_l = metric_mse(blend_l, imgL)

					# right
					out = net_meta(grid_inp, R_tag)
					out = out.reshape(1, h_res, -1, 3*num_planes).permute(0,3,1,2)
					multi_out = [out[:, 3*i:3*i+3, :, base_shift + offsets[i]:base_shift + w_res + offsets[i]] for i in range(len(offsets))]
					multi_out = torch.cat(multi_out,dim=1)
					blend_r = net_blend(2*multi_out-1)
					inner_loss_r = metric_mse(blend_r, imgR)
					inner_loss_r = 0

					loss = (inner_loss_l + inner_loss_r) * 0.5

					if args.w_vgg>0:
						blend_l_vg = VGGpreprocess(blend_l)
						imgL_vg = VGGpreprocess(imgL)

						blend_r_vg = VGGpreprocess(blend_r)
						imgR_vg = VGGpreprocess(imgR)

						loss += (metric_vgg(blend_l_vg, imgL_vg) + metric_vgg(blend_r_vg, imgR_vg)* 0.) * args.w_vgg * 0.5

					optimizer.zero_grad()
					loss.backward()
					optimizer.step()

					if i%10 == 0:
						print(f"{i} step, {loss} loss")

			saveimg(blend_l, f"{save_path}/{name}_blend_l.png")
			saveimg(blend_r, f"{save_path}/{name}_blend_r.png")


			# compute outer loss
			for inter in inter_list:

				z0 = net_meta.ret_z(torch.LongTensor([0.]).to(device)).squeeze()
				z1 = net_meta.ret_z(torch.LongTensor([1.]).to(device)).squeeze()
				zi = linterp(inter, z0, z1).unsqueeze(0).to(device)
				out = net_meta.forward_with_z(grid_inp, zi)
				out = out.reshape(1, h_res, -1, 3*num_planes).permute(0,3,1,2)
				if args.max_disp==0:
					multi_out = [out[:, 3*i:3*i+3, :, base_shift - offsets[i]:base_shift + w_res - offsets[i]] for i in range(len(offsets))]
					multi_out = torch.cat(multi_out,dim=1)
				else:
					off_scl = np.interp(inter, xp, fp)
					multi_out_list=[]
					for i in range(num_planes):
						meshxw = meshx + (base_shift * 2 + off_scl * offsets[i]) / (w_res + base_shift * 2)
						grid = torch.stack((meshxw, meshy), 2)[None].to(device).to(torch.float32)
						multi_out = grid_sample(out[:, 3*i:3*i+3], grid, mode='bilinear', align_corners=True)[:, :, :, :-base_shift*2]
						multi_out_list.append(multi_out)
					multi_out = torch.cat(multi_out_list,dim=1)

				blend_test = net_blend(2*multi_out-1)

				saveimg(blend_test, f"{save_path}/{name}_{inter:0.2f}.png")
				blend_test = (blend_test.permute(0, 2, 3, 1) * 255).clamp(0, 255).to(torch.uint8).squeeze(0)
				blend_test = blend_test.detach().cpu().numpy()
				video_out.append_data(blend_test)

		video_out.close()


	
if __name__=='__main__':
	

	parser = argparse.ArgumentParser()
	parser.add_argument('--test_dataset',  type=str, help='path to test dataset',default = 'dataset/time/v')
	parser.add_argument('--load_path',  type=str, help='path to dataset',default = 'dataset/time/v')
	parser.add_argument('--save_name',  type=str, help='saved name of folder', default = 'test')
	parser.add_argument('--net',  type=str, help='type of network', default = 'mlp')
	parser.add_argument('--nfg',  type=int, help='# of channel',default = 8)
	parser.add_argument('--num_freqs_pe',type=int,help='#frequencies for positional encoding',    default = 10)	
	parser.add_argument('--n_layer', help='layer number of meta MLP',type=int, default = 8)
	parser.add_argument('--mlp_d', help='nnblend MLP layer number',type=int, default = 2)
	parser.add_argument('--mlp_w', help='channel of nnblend MLP',type=int, default = 128)
	parser.add_argument('--lr', help='learning rate of post optimization',type=float, default = 1e-5)
	parser.add_argument('--inner_lr', help='inner learning rate',type=float, default = 0.001)
	parser.add_argument('--w_vgg', help='weight of vgg',type=float, default = 0.01)
	parser.add_argument('--resolution', help='resolution [h,w]',nargs='+', type=int, default = [270, 480])
	parser.add_argument('--use_norm', help='use my network for debugging', action='store_true')
	parser.add_argument('--n_c', help='number of channel in netMet',type=int, default = 256)
	parser.add_argument('--max_disp', help='maximum disparity',type=int, default = 30)
	parser.add_argument('--total_step', help='total step of optimization',type=int, default = 0)
	parser.add_argument('--no_blend_optim', help='do not optimize blend network', action='store_true')

	args = parser.parse_args()

	print(args)

	meta_test(args)
