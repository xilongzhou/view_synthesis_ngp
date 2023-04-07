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

from xfields_blending import Blending_train_stereo_video_only as Blend
from dataloaderSVPE import Video
from models.xfields import XfieldsFlow
from util.args import TrainingArguments


from PIL import Image

from utils import VGGLoss, VGGpreprocess

from network import VIINTER, linterp, Unet_Blend, MLP_blend
import torch.nn.functional as F

import random

torch.set_num_threads(1)

def init(args):

	torch.manual_seed(1234)
	np.random.seed(1234)
	savedir = os.path.join(args.savepath,os.path.split(args.dataset)[1])
	os.makedirs(os.path.join(savedir,"trained model"), exist_ok=True)
	os.makedirs(os.path.join(savedir,"saved training"), exist_ok=True)
	
	return savedir


def ms_loss(out, images, masks, flows, gtflows, coeff, mask_coeff, shift_mask, planes, delta, rgb_layer=None):

	L1 = nn.L1Loss()
	masks = masks * mask_coeff
	outM, gtM = out*masks, images*masks
	# warp_loss = torch.Tensor([0]).cuda()
	warp_loss  = L1(outM, gtM)
	outM2 = interpolate(outM, mode='bilinear', scale_factor=1/2, antialias=True)
	gtM2  = interpolate(gtM,  mode='bilinear', scale_factor=1/2, antialias=True)
	warp_loss += (1/2) * L1(outM2, gtM2)
	outM4 = interpolate(outM, mode='bilinear', scale_factor=1/4, antialias=True)
	gtM4  = interpolate(gtM,  mode='bilinear', scale_factor=1/4, antialias=True)
	warp_loss += (1/4) * L1(outM4, gtM4)
	outM8 = interpolate(outM, mode='bilinear', scale_factor=1/8, antialias=True)
	gtM8  = interpolate(gtM,  mode='bilinear', scale_factor=1/8, antialias=True)
	warp_loss += (1/8) * L1(outM8, gtM8)
	# flow_loss = (coeff)*L1(flows*(1-masks), gtflows*(1-masks))
	# flow_loss = (coeff)*L1(shift_mask * flows, shift_mask * gtflows)
	flow_loss = (coeff)*L1(flows, gtflows)
	
	aux_flow_loss = sum([L1(msk*delta*pln, msk*gtflows[:, :1]) for pln, msk in zip(planes, shift_mask)])


	if rgb_layer is not None:
		rgb_wrap = None
		for tmp_rgb, tmp_mask in zip(rgb_layer, shift_mask):
			# print("tmp_rgb: ", tmp_rgb.shape)
			# print("tmp_mask: ", tmp_mask.shape)
			if rgb_wrap is None:
				rgb_wrap = tmp_mask*tmp_rgb
			else:
				rgb_wrap += tmp_mask*tmp_rgb

		rgbout2 = interpolate(rgb_wrap, mode='bilinear', scale_factor=1/2, antialias=True)
		gt2 = interpolate(images, mode='bilinear', scale_factor=1/2, antialias=True)
		rgbout4 = interpolate(rgb_wrap, mode='bilinear', scale_factor=1/4, antialias=True)
		gt4 = interpolate(images, mode='bilinear', scale_factor=1/4, antialias=True)
		rgbout8 = interpolate(rgb_wrap, mode='bilinear', scale_factor=1/8, antialias=True)
		gt8 = interpolate(images, mode='bilinear', scale_factor=1/8, antialias=True)

		rgb_wrap_loss = L1(images, rgb_wrap)# + (1/2) * L1(rgbout2, gt2) + (1/4) * L1(rgbout4, gt4) + (1/8) * L1(rgbout8, gt8)

		warp_loss *= 0
		aux_flow_loss *= 0
		flow_loss *= 0

		return warp_loss+ flow_loss + aux_flow_loss + rgb_wrap_loss, warp_loss, flow_loss, aux_flow_loss

	return warp_loss + flow_loss + aux_flow_loss, warp_loss, flow_loss, aux_flow_loss


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


def run_training(args):
	def save_flow(flows_out, txt, max_mot, shift_mask=None):
		disp = torch.clamp(-flows_out,0, max_mot.item()).permute(1, 2, 0).detach().cpu().numpy()[:, :, 0]

		disp_vis = (disp - min_motion.detach().cpu().numpy()) / (max_mot.detach().cpu().numpy() - min_motion.detach().cpu().numpy()) * 255.0
		disp_vis = disp_vis.astype("uint8")
		disp_vis = cv2.applyColorMap(disp_vis, cv2.COLORMAP_INFERNO)
		
		if shift_mask is not None:
			disp_vis[~shift_mask] = [255, 255, 255]
		cv2.imwrite("%s/saved training/flow%s.png"%(savedir, txt),np.uint8(disp_vis))
		return

	def save_img(out, txt):
		# out_img = np.minimum(np.maximum(out.permute(1, 2, 0).detach().cpu().numpy(),0.0),1.0)
		# cv2.imwrite("%s/saved training/recons%s.png"%(savedir, txt),np.uint8(out_img*255))

		if out.dim()==4:
			out = out[0]

		out_img = (out.permute(1, 2, 0) * 255).clamp(0, 255).to(torch.uint8).detach().cpu().numpy()

		if out_img.shape[-1]==1:
			out_img = np.repeat(out_img,3, axis=-1)

		# print(out_img.shape)
		Image.fromarray(out_img, 'RGB').save("%s/saved training/%s.png"%(savedir, txt))

		return

	# def save_init():
	#     out = Blending_train_stereo(input_coords, images, gtflows, h_res, w_res)
	#     [save_flow((gtflows*masks)[idx], "GT{}".format(idx)) for idx in range(2)]
	#     [save_img(out[idx], "GT{}".format(idx)) for idx in range(2)]
	#     [save_img((torch.flip(images, [0])*masks)[idx], "GTM{}".format(idx)) for idx in range(2)]
	#     return

	def save_progress_disp():
		save_img(out_L[0], "train")
		save_img(imgL[0], "trainGT")
		save_img(mask[0], "trainmask")
		save_img(shift_mask_L[0][0], "trainmask_s")
		save_flow(flow_L[0], "train", max_mot=max_motion)
		save_flow(flowgt_L[0], "trainGT", max_mot=max_motion)
		with torch.no_grad():
			model_out = flow_model(L_tag)

			multi_disp_L = [model_out[:, i:i+1, :, base_shift - offsets[i]:base_shift + w_res - offsets[i]] for i in range(len(offsets))]

			jacobianX = merge_planes(multi_disp_L, flowgt_L)
			
			jacobianY = torch.zeros_like(jacobianX).cuda()
			jacobian  = torch.cat((jacobianX, jacobianY), dim=1)
			flowgt_L = torch.abs(flowgt_L)
			shift_maskV = [torch.logical_and(flowgt_L >= bounds[i][0], flowgt_L <= bounds[i][1]).detach() for i in range(len(offsets))]
			outV, flowV, _ = Blend(indicesL, imgL, jacobian)

		save_img(outV[0], "")
		save_img(imgL[0], "GT")
		# save_img(maskV[0], "mask")
		# save_img(shift_maskV[0][0], "mask_s")
		save_flow(flowV[0], "", max_mot=max_motion_v)
		save_flow(flowgt_L[0], "GT", max_mot=max_motion_v)
		[save_flow((shift_maskV[i]*flowgt_L)[0], "GTSM0{}".format(i), shift_mask=shift_maskV[i][0, 0].detach().cpu().numpy(), max_mot=max_motion_v) for i in range(len(offsets))]
		[save_flow((shift_maskV[i]*flowgt_L)[0], "GTSMN0{}".format(i), max_mot=max_motion_v) for i in range(len(offsets))]
		[save_flow((-multi_disp_L[i])[0], "SM0{}".format(i), max_mot=max_motion_v) for i in range(len(offsets))]
		[save_img((shift_maskV[i].expand(-1, 3, h_res, w_res))[0], "mask_s0{}".format(i)) for i in range(len(offsets))]
		return

	def save_progress_debug(rgb_warp, rgb_gt, iter):
		save_img(rgb_warp[0], f"train_{iter}")
		save_img(rgb_gt[0], f"trainGT")



	savedir = init(args)

	# sample = load_imgs(args, args.dataset)
	# images, input_coords, val_coords, gtflows, masks = [x.cuda() for x in sample[:-2]]
	# h_res, w_res = sample[-2:]

	video_pairs = Video(args, args.dataset)
	dataloader = DataLoader(video_pairs, batch_size=1, shuffle=True)
	print(video_pairs)
	
	imagesV, input_coordsV, gtflowV, maskV, indicesV, gtflowV1, max_motion = video_pairs.__getitem__(0)
	# _, _, h_res, w_res = imagesV.shape
	h_res = 270
	w_res = 480
	print(torch.max(gtflowV), torch.min(gtflowV), torch.max(gtflowV1), torch.min(gtflowV1))
	# exit()
	# save_init()
	max_motion_v = torch.min(torch.max(torch.abs(gtflowV)), torch.max(torch.abs(gtflowV1)))
	min_motion = torch.min(torch.min(torch.abs(gtflowV)), torch.min(torch.abs(gtflowV1)))
	print(max_motion, min_motion)
	# exit()
	
	offsets, bounds, base_shift = get_planes(max_motion, args.num_planes, w_res)

	print(offsets * 2, bounds, bounds * w_res, base_shift, max_motion * w_res)
	print([[i, i+1, base_shift + offsets[i], base_shift + w_res + offsets[i]] for i in range(len(offsets))])
	print([[bounds[i][0], bounds[i][1]] for i in range(len(offsets))])
	# exit()
	# hist, bin_edges = torch.histogram(gtflowV[:, :1].cpu(), bins=10)
	# print(hist, bin_edges)

	# import matplotlib.pyplot as plt
	# plt.hist(bin_edges[:-1], bin_edges, weights=hist)
	# plt.savefig('hist.png')
	# exit()
	# bounds = np.array([[0, 15], [12, 32], [28, w_res]]) / w_res
	# shift_mask.append(torch.logical_and(abs_gt >= 0.000, abs_gt <= 0.023).detach())
	# shift_mask.append(torch.logical_and(abs_gt >= 0.015, abs_gt <= 0.053).detach())
	# shift_mask.append(torch.logical_and(abs_gt >= 0.047, abs_gt <= 1.000).detach())

	## out rgb + disp
	if args.out_rgbdisp:
		flow_model  = XfieldsFlow(h_res, w_res + base_shift * 2, inChannels=1, ngf=args.nfg, outChannels=args.num_planes*4).cuda()

	## out rgb
	if args.out_rgb:
		# use different network
		if args.net=="conv":
			L_tag = torch.tensor([0.03]).cuda().unsqueeze(0).unsqueeze(0).unsqueeze(0).to(torch.float32)
			R_tag = torch.tensor([0.06]).cuda().unsqueeze(0).unsqueeze(0).unsqueeze(0).to(torch.float32)

			flow_model  = XfieldsFlow(h_res, w_res + base_shift * 2, inChannels=1, ngf=args.nfg, outChannels= 3*args.num_planes).cuda()
		elif args.net=="mlp":

			L_tag = torch.tensor([0]).cuda()
			R_tag = torch.tensor([1]).cuda()

			flow_model = VIINTER(n_emb = 2, norm_p = 1, inter_fn=linterp, D=8, z_dim = 128, in_feat=2, out_feat=3*args.num_planes, W=512, with_res=False, with_norm=True).cuda()
			if args.nn_blend:
				if args.blendnn_type=="unet":
					net_blend = Unet_Blend(3*args.num_planes, 3, 4, (h_res, w_res)).cuda()
				elif args.blendnn_type=="mlp":
					net_blend = MLP_blend(D=args.mlp_d, in_feat=3*args.num_planes, out_feat=3, W=args.mlp_w, with_res=False, with_norm=True).cuda()


			coords_h = np.linspace(-1, 1, h_res, endpoint=False)
			# if args.nn_blend:
			#     coords_w = np.linspace(-1, 1,  w_res, endpoint=False)
			# else:
			coords_w = np.linspace(-1, 1,  w_res + base_shift * 2, endpoint=False)
			xy_grid = np.stack(np.meshgrid(coords_w, coords_h), -1)
			xy_grid = torch.FloatTensor(xy_grid).cuda()
			# if not args.rowbatch:
			grid_inp = xy_grid.view(-1, 2).contiguous().unsqueeze(0)
			print(h_res, w_res)
			print("grid_inp: ", grid_inp.shape)

	## out disp
	if args.out_disp:
		flow_model  = XfieldsFlow(h_res, w_res + base_shift * 2, inChannels=1, ngf=args.nfg, outChannels= args.num_planes).cuda()

	nparams_decoder = sum(p.numel() for p in flow_model.parameters() if p.requires_grad)
	print('Number of learnable parameters (decoder): %d' %(nparams_decoder))
	if args.nn_blend:
		nparams_nnblend = sum(p.numel() for p in net_blend.parameters() if p.requires_grad)
		print('Number of learnable parameters (blend nn): %d' %(nparams_nnblend))

	if args.net=="mlp":

		if args.nn_blend:
			print("optimize both flow model and net blend")

			optimizer = torch.optim.Adam(list(flow_model.parameters()) + list(net_blend.parameters()), lr=1e-5)
			scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_iters, eta_min=1e-6)
		else:
			optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, flow_model.parameters()), lr=1e-5)
			scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_iters, eta_min=1e-6)			


	elif args.net=="conv":
		optimizer = Adam(params=flow_model.parameters(), lr=args.lr)
		# scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[args.num_iters//3, (2*args.num_iters)//3], gamma=0.4)
		scheduler = lr_scheduler.StepLR(optimizer, step_size=int(args.num_iters*0.5), gamma=0.5)

	st = time.time()
	loss_t, loss_w_t, loss_f_t, loss_af_t = 0, 0, 0, 0
	coeff, iter = 20, 0

	# load image and mask
	imgR_path = "./fall/0001R.png"
	imgL_path = "./fall/0001L.png"
	imgR_pil = Image.open(imgR_path).convert("RGB")
	imgL_pil = Image.open(imgL_path).convert("RGB")
	imgR_pil = imgR_pil.resize((480, 270), Image.LANCZOS )
	imgL_pil = imgL_pil.resize((480, 270), Image.LANCZOS )

	w, h = imgR_pil.size

	imgR = torch.from_numpy(np.array(imgR_pil)).cuda().permute(2,0,1).unsqueeze(0)/255.
	imgL = torch.from_numpy(np.array(imgL_pil)).cuda().permute(2,0,1).unsqueeze(0)/255.

	# load flow
	if not args.nn_blend:
		flowgt_L = torch.from_numpy(np.load("./fall/disp_0_1.npy")).cuda().permute(2, 0, 1)[None]
		flowgt_R = torch.from_numpy(np.load("./fall/disp_1_0.npy")).cuda().permute(2, 0, 1)[None]
		flowgt_L = F.interpolate(flowgt_L, size=(270,480), mode='bilinear', align_corners=True)   
		flowgt_R = F.interpolate(flowgt_R, size=(270,480), mode='bilinear', align_corners=True)   
		print("flowgt_L: ", flowgt_L.shape)
		print("flowgt_R: ", flowgt_R.shape)

		save_flow(flowgt_L[0], "dispL_gt", max_mot=max_motion)
		save_flow(flowgt_R[0], "dispR_gt", max_mot=max_motion)


		absgt_L = torch.abs(flowgt_L[:,:1,:,:])
		shift_mask_L = [torch.logical_and(absgt_L >= bounds[i][0], absgt_L <= bounds[i][1]).detach() for i in range(len(offsets))]

		absgt_R = torch.abs(flowgt_R[:,:1,:,:])
		shift_mask_R = [torch.logical_and(absgt_R >= bounds[i][0], absgt_R <= bounds[i][1]).detach() for i in range(len(offsets))]


	# self.max_motion = torch.max(torch.max(torch.abs(flow_L)), self.max_motion)
	# self.max_motion = torch.max(torch.max(torch.abs(flow_L)), self.max_motion)
	# self.flows[key] = flow6

	# indices_L = torch.tensor([[[0]],[[0]]]).cuda()

	indices = video_pairs.return_indices()
	indicesL = torch.index_select(indices, 0, torch.tensor([0,1], device="cuda"))
	indicesR = torch.index_select(indices, 0, torch.tensor([1,0], device="cuda"))
	print(indicesL)
	print(indicesR)

	mask = video_pairs.return_mask()

	# for metric
	if args.w_vgg>0:
		metric_vgg = VGGLoss()
		imgL_vg = VGGpreprocess(imgL)
		imgR_vg = VGGpreprocess(imgR)

	metric_l1 = nn.L1Loss()
	metric_mse = nn.MSELoss()

	l1loss = 0
	vgloss = 0

	for _ in range(args.num_iters):

		optimizer.zero_grad()
		
		# output rgb only
		if args.out_rgb:

			if args.net=="conv":
				out_L = (1 + flow_model(L_tag))*0.5
				out_R = (1 + flow_model(R_tag))*0.5

				multi_out_L = [out_L[:, 3*i:3*i+3, :, base_shift - offsets[i]:base_shift + w_res - offsets[i]] for i in range(len(offsets))]
				multi_out_R = [out_R[:, 3*i:3*i+3, :, base_shift - offsets[i]:base_shift + w_res - offsets[i]] for i in range(len(offsets))]
				# print("multi_out_L: ",multi_out_L[0].shape)
				# print("multi_out_L: ",multi_out_L[1].shape)
				# print("out_L: ",out_L[0:1,3*l:3*l+3,...].shape)

				blend_L = None
				blend_R = None
				for l in range(args.num_planes):
					# print("shift mask: ",shift_mask_L[l].shape)
					# print("out_L: ",out_L[0:1,3*l:3*l+3,...].shape)

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
				
				if args.w_vgg>0:
					outL_vg = VGGpreprocess(out_L) 
					outR_vg = VGGpreprocess(out_R) 
	 
					vgloss = (metric_vgg(outL_vg, imgL_vg) + metric_vgg(outR_vg, imgR_vg))*args.w_vgg

				loss = l1loss + vgloss

			elif args.net=="mlp":
				# sind = num_pixels
				if random.choice([True, False]):
					use_left = True
					tag = L_tag
					gt_img = imgL
					if not args.nn_blend:
						shift_mask = shift_mask_L

				else:
					use_left = False
					tag = R_tag
					gt_img = imgR
					if not args.nn_blend:
						shift_mask = shift_mask_R

				# print(tag)

				out = flow_model(grid_inp, tag)
				out = out.reshape(1, h_res, -1, 3*args.num_planes).permute(0,3,1,2)

				if use_left:
					# print("l")
					multi_out = [out[:, 3*i:3*i+3, :, base_shift - offsets[i]:base_shift + w_res - offsets[i]] for i in range(len(offsets))]
				else:
					# print("r")
					multi_out = [out[:, 3*i:3*i+3, :, base_shift + offsets[i]:base_shift + w_res + offsets[i]] for i in range(len(offsets))]
				# print("multi_out_L: ",multi_out_L[0].shape)
				# print("out_L: ",out_L[0:1,3*l:3*l+3,...].shape)

				if args.nn_blend:
					multi_out = torch.cat(multi_out,dim=1)
					blend = net_blend(2*multi_out-1)
				else:
					blend = None
					for l in range(args.num_planes):
						if blend is None:
							blend = multi_out[l]*shift_mask[l]
						else:
							blend += multi_out[l]*shift_mask[l]

				l1loss = metric_mse(blend, gt_img)*args.w_l1 

				loss = l1loss 

		loss.backward()
		optimizer.step()
		scheduler.step()

		# loss_t += loss.item()
		# loss_w_t += warp_loss.item()
		# loss_f_t += flow_loss.item()
		# loss_af_t += aux_flow_loss.item()

		if (iter % args.progress_iter) ==0:
			print(f"iterations: {iter}, loss: {loss}, l1loss: {l1loss}, vgloss: {vgloss} ")
			# print('\r Iteration %6d Cumulative -> Loss = %3.3f WLoss = %3.3f FLoss = %3.3f AFLoss = %3.3f coeff = %3.3f '%(iter+1, loss, warp_loss, flow_loss, aux_flow_loss, coeff),end=" " )

		# if ((iter % args.progress_iter) == (args.progress_iter - 1)):
			# print(" elapsed time %3.1f m  Averaged -> Loss = %3.5f WLoss = %3.5f FLoss = %3.5f AFLoss = %3.5f"%((time.time()-st)/60, loss_t/args.progress_iter, loss_w_t/args.progress_iter, loss_f_t/args.progress_iter, loss_af_t/args.progress_iter))
			# save_dict = {'state_dict':flow_model.state_dict(), 'offsets':offsets, 'base_shift':base_shift, 'num_planes':args.num_planes}
			if not args.nn_blend:
				save_dict = {'state_dict':flow_model.state_dict(), 'offsets':offsets, 'base_shift':base_shift, 'num_planes':args.num_planes}
			else:
				save_dict = {'state_dict':flow_model.state_dict(), 'nn_blend':net_blend.state_dict(), 'offsets':offsets, 'base_shift':base_shift, 'num_planes':args.num_planes}
			torch.save(save_dict, "%s/trained model/model.ckpt"%(savedir))
			# save_progress() if not args.debug else save_progress_debug(model_out, gt_img, iter)

			if args.out_rgb:
				if args.net=="conv":
					save_img(out_L, f"out_L_{iter}")
					save_img(out_R, f"out_R_{iter}")
					save_img(imgR[0], f"gt_R")
					save_img(imgL[0], f"gt_L")
					[save_img((shift_mask_L[i].expand(-1, 3, h_res, w_res))[0], "mask_L{}".format(i)) for i in range(len(offsets))]
					[save_img((shift_mask_R[i].expand(-1, 3, h_res, w_res))[0], "mask_R{}".format(i)) for i in range(len(offsets))]
				elif args.net =="mlp":
					save_img(blend, f"out_{iter}")
					save_img(imgR[0], f"gt_R")
					save_img(imgL[0], f"gt_L")
					if not args.nn_blend:
						[save_img((shift_mask_L[i].expand(-1, 3, h_res, w_res))[0], "mask_L{}".format(i)) for i in range(len(offsets))]
						[save_img((shift_mask_R[i].expand(-1, 3, h_res, w_res))[0], "mask_R{}".format(i)) for i in range(len(offsets))]                    

		iter += 1





if __name__=='__main__':
	
	args = TrainingArguments().parser.parse_args()
	print(args)

	run_training(args)