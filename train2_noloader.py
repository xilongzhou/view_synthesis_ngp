from sched import scheduler
import torch
import torch.nn as nn
from torch.nn.functional import interpolate
from torch.optim import lr_scheduler, Adam
from torch.utils.data import DataLoader
import torchvision.transforms as T

from os.path import join as join

import numpy as np
import cv2,os,time
from tqdm import tqdm


from PIL import Image


from utils import VGGLoss, VGGpreprocess, saveimg

from network import VIINTER, linterp, Unet_Blend, MLP_blend, CondSIREN
import torch.nn.functional as F

import random

from collections import OrderedDict


from torch.nn.functional import interpolate, grid_sample

import argparse
import commentjson as json

torch.set_num_threads(20)


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

	L_tag = torch.tensor([0]).cuda()
	R_tag = torch.tensor([1]).cuda()

	## ------------------------ no use dataloader -------------------

	all_ids = [0,1,2,3,4,5,6,7,8,9,10,11]

	R_id = 11
	L_id = 0
	dist = 1/(R_id - L_id)
	inter_list = [np.float32(i*dist) for i in range(R_id - L_id+1)]


	########### load image 
	imgL_name = f"rgba_00000.png"
	imgR_name = f"rgba_00011.png"
	imgL_pil = Image.open(join(args.datapath, imgL_name)).convert("RGB")
	imgR_pil = Image.open(join(args.datapath,imgR_name)).convert("RGB")

	# all_testimgs = []
	# for rand_id in all_ids:
	# 	imgtest_name = f"rgba_{rand_id:05d}.png"
	# 	imgtest_pil = Image.open(join(args.datapath, imgtest_name)).convert("RGB")
	# 	imgtest = torch.from_numpy(np.array(imgtest_pil)).permute(2,0,1)/255.
	# 	all_testimgs.append(imgtest)
	imgL = torch.from_numpy(np.array(imgL_pil)).permute(2,0,1)/255.
	imgR = torch.from_numpy(np.array(imgR_pil)).permute(2,0,1)/255.


	########### load mask 
	all_maskL = []
	all_maskR = []
	all_rgbL = []
	all_rgbR = []
	for i in range(args.num_planes):

		if args.warp_data:
			maskL_name = f"warpmask{i}_00000.png"
			maskR_name = f"warpmask{i}_00011.png"

			rgbL_name = f"warprgb{i}_00000.png"
			rgbR_name = f"warprgb{i}_00011.png"

			# left
			maskL = np.array(Image.open(join(args.datapath,maskL_name))).astype(np.float32)
			rgbL = np.array(Image.open(join(args.datapath,rgbL_name))).astype(np.float32)
			maskL = torch.from_numpy(maskL).permute(2,0,1)[0,:,:]/255.
			rgbL = torch.from_numpy(rgbL).permute(2,0,1)/255.

			# right
			maskR = np.array(Image.open(join(args.datapath,maskR_name))).astype(np.float32)
			rgbR = np.array(Image.open(join(args.datapath,rgbR_name))).astype(np.float32)
			maskR = torch.from_numpy(maskR).permute(2,0,1)[0,:,:]/255.
			rgbR = torch.from_numpy(rgbR).permute(2,0,1)/255.


			all_rgbL.append(rgbL)		
			all_rgbR.append(rgbR)		
			all_maskL.append(maskL)		
			all_maskR.append(maskR)		

		else:
			maskL_name = f"mask{i}_00000.png"
			maskR_name = f"mask{i}_00011.png"

			# left
			maskL = np.array(Image.open(join(args.datapath,maskL_name))).astype(np.float32)
			maskL = torch.from_numpy(maskL)#/255.

			# right
			maskR = np.array(Image.open(join(args.datapath,maskR_name))).astype(np.float32)
			maskR = torch.from_numpy(maskR)#/255.	
			all_maskL.append(maskL)		
			all_maskR.append(maskR)		

	all_maskL = torch.stack(all_maskL)
	all_maskR = torch.stack(all_maskR)

	if args.warp_data:
		all_rgbL = torch.stack(all_rgbL).to(device)
		all_rgbR = torch.stack(all_rgbR).to(device)

	# print("all_rgbL: ", all_rgbL.shape)
	# print("all_maskL: ", all_maskL.shape)


	########### load disparity
	disp = np.load(os.path.join(args.datapath,"disp_0_11.npy"))


	# ------------------------ start processing -----------------------

	imgL = imgL.to(device)
	imgR = imgR.to(device)

	all_maskL = all_maskL.to(device)
	all_maskR = all_maskR.to(device)
	disp = torch.from_numpy(disp).to(device)

	for i in range(args.num_planes):
		mask_imgL = imgL*all_maskL[i] if not args.warp_data else all_rgbL[i]
		saveimg(mask_imgL, f"{save_imgpath}/{i}_outL_gt.png")


		mask_imgR = imgR*all_maskR[i] if not args.warp_data else all_rgbR[i]
		saveimg(mask_imgR, f"{save_imgpath}/{i}_outR_gt.png")

		if args.add_mask:
			saveimg(all_maskL[i].unsqueeze(0).repeat(3,1,1), f"{save_imgpath}/{i}_maskL_gt.png")
			saveimg(all_maskR[i].unsqueeze(0).repeat(3,1,1), f"{save_imgpath}/{i}_maskR_gt.png")


	# compute offsets, baseline (scene-dependent)
	disp = torch.abs(disp[:,:,0])
	max_disp = torch.max(disp)
	min_disp = torch.min(disp)
	planes = torch.round(torch.linspace(min_disp, max_disp, args.num_planes+1)/2)*2
	base_shift = int(max_disp//2)
	offsets = [ int((planes[i]/2+planes[i+1]/2)//2) for i in range(args.num_planes)]

	# print(f"offsets: {offsets}, baseshift: {base_shift}")

	coords_h = np.linspace(-1, 1, h_res, endpoint=False)
	coords_w = np.linspace(-1, 1,  w_res + base_shift * 2, endpoint=False)
	# coords_w = np.linspace(-1, 1,  w_res, endpoint=False)
	xy_grid = np.stack(np.meshgrid(coords_w, coords_h), -1)
	xy_grid = torch.FloatTensor(xy_grid).cuda()
	grid_inp = xy_grid.view(-1, 2).contiguous().unsqueeze(0)

	# ----------------------- define network --------------------------------------
	if args.use_viinter:
		net_scene = VIINTER(n_emb = 2, norm_p = 1, inter_fn=linterp, D=args.n_layer, z_dim = 128, in_feat=2, out_feat=3*args.num_planes if not args.add_mask else 4*args.num_planes, W=args.n_c, with_res=False, with_norm=True).cuda()
	elif args.use_tcnn:
		with open(args.config) as config_file:
			config = json.load(config_file)
		net_scene = tiny_cuda(n_emb = 2, config=config ,norm_p = 1, out_feat=3*args.num_planes, debug=args.debug).cuda()
	else:
		net_scene = CondSIREN(n_emb = 2, norm_p = 1, inter_fn=linterp, D=args.n_layer, z_dim = 128, in_feat=2, out_feat=3*args.num_planes if not args.add_mask else 4*args.num_planes, W=args.n_c, with_res=False, with_norm=args.use_norm, use_sig=args.use_sigmoid).cuda()
	
	if args.use_viinter:
		optimizer = torch.optim.Adam(net_scene.parameters(), lr=1e-5)
		scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_iters, eta_min=1e-6)
	else:
		optimizer = torch.optim.Adam(net_scene.parameters(), lr=args.lr)

	print("net_scene: ", net_scene)

	# for metric
	metric_l1 = nn.L1Loss()
	metric_mse = nn.MSELoss()

	# for metric
	if args.w_vgg>0:
		metric_vgg = VGGLoss()

	mseloss = 0
	vgloss = 0
	blend_loss = 0
	inter_loss = 0

	xp = [0, 1]
	fp = [-2, 2]

	iter = 0
	shuffle = True

	while iter < args.num_iters:

		rand_id = random.choice(all_ids) 
		inter = inter_list[rand_id]

		mseloss = torch.tensor(0.).cuda()
		vgloss = torch.tensor(0.).cuda()
		scene_loss = torch.tensor(0.).cuda()

		# ----------- shuffle ------------------------
		flag = random.choice([True, False])
		# flag = True

		if flag:

			out_L = net_scene(grid_inp, L_tag)
			out_L = out_L.reshape(1, h_res, -1, 3*args.num_planes if not args.add_mask else 4*args.num_planes).permute(0,3,1,2)
		
			for i in range(args.num_planes):

				if args.add_mask:
					if args.warp_data:
						mask_imgL = all_rgbL[i]
					else:
						mask_imgL = imgL*all_maskL[i]
					out_L_shift = out_L[:, 4*i:4*i+3, :, base_shift - offsets[i]:base_shift + w_res - offsets[i]]

					if args.out_black:
						mseloss = metric_mse(mask_imgL, out_L_shift)
						scene_loss = scene_loss + mseloss

						if args.w_vgg>0:
							mask_imgL_vg = VGGpreprocess(mask_imgL.unsqueeze(0))
							out_L_shift_vg = VGGpreprocess(out_L_shift)
							vgloss = metric_vgg(mask_imgL_vg, out_L_shift_vg) * args.w_vgg
							scene_loss = scene_loss + vgloss
					else:
						scene_loss = scene_loss + metric_mse(mask_imgL, out_L_shift*all_maskL[i])


					mask_L_shift = out_L[:, 4*i+3:4*i+4, :, base_shift - offsets[i]:base_shift + w_res - offsets[i]]
					scene_loss = scene_loss + metric_mse(mask_L_shift, all_maskL[i])
				else:
					mask_imgL = imgL*all_maskL[i]
					out_L_shift = out_L[:, 3*i:3*i+3, :, base_shift - offsets[i]:base_shift + w_res - offsets[i]]
					scene_loss = scene_loss + metric_mse(mask_imgL, out_L_shift*all_maskL[i])

		else:
			out_R = net_scene(grid_inp, R_tag)
			out_R = out_R.reshape(1, h_res, -1, 3*args.num_planes if not args.add_mask else 4*args.num_planes).permute(0,3,1,2)

			for i in range(args.num_planes):

				if args.add_mask:
					if args.warp_data:
						mask_imgR = all_rgbR[i]
					else:
						mask_imgR = imgR*all_maskR[i]

					out_R_shift = out_R[:, 4*i:4*i+3, :, base_shift + offsets[i]:base_shift + w_res + offsets[i]]

					if args.out_black:
						mseloss = metric_mse(mask_imgR, out_R_shift)
						scene_loss = scene_loss + mseloss

						if args.w_vgg>0:
							mask_imgR_vg = VGGpreprocess(mask_imgR.unsqueeze(0))
							out_R_shift_vg = VGGpreprocess(out_R_shift)
							vgloss = metric_vgg(mask_imgR_vg, out_R_shift_vg) * args.w_vgg
							scene_loss = scene_loss + vgloss
					else:
						scene_loss = scene_loss + metric_mse(mask_imgR, out_R_shift*all_maskR[i])

					mask_R_shift = out_R[:, 4*i+3:4*i+4, :, base_shift + offsets[i]:base_shift + w_res + offsets[i]]
					scene_loss = scene_loss + metric_mse(mask_R_shift, all_maskR[i])
				else:
					mask_imgR = imgR*all_maskR[i]
					out_R_shift = out_R[:, 3*i:3*i+3, :, base_shift + offsets[i]:base_shift + w_res + offsets[i]]
					scene_loss = scene_loss + metric_mse(mask_imgR, out_R_shift*all_maskR[i])

			# ----------- end of shuffle ------------------------
			
		optimizer.zero_grad()
		scene_loss.backward()
		optimizer.step()
		if args.use_viinter:
			scheduler.step()


		if (iter % args.progress_iter) ==0:

			print(f"iterations: {iter}, interval: {inter}, blend_loss: {blend_loss}, mseloss: {mseloss}, vgloss: {vgloss}, scene_loss: {scene_loss}, inter_loss: {inter_loss}")
			print(f"offset: {offsets}, base_shift: {base_shift}, planes: {planes},")

			save_dict = { 
						'mlp':net_scene.state_dict(), \
						'nfg':args.nfg, \
						'n_layer':args.n_layer, \
						'num_planes':args.num_planes, \
						}
			torch.save(save_dict, "%s/model.ckpt"%(save_modelpath))
			# save_progress() if not args.debug else save_progress_debug(model_out, gt_img, iter)

			with torch.no_grad():

				z0 = net_scene.ret_z(torch.LongTensor([0.]).cuda()).squeeze()
				z1 = net_scene.ret_z(torch.LongTensor([1.]).cuda()).squeeze()

				zi = linterp(inter, z0, z1).unsqueeze(0)
				out = net_scene.forward_with_z(grid_inp, zi)
				out = out.reshape(1, h_res, -1, 3*args.num_planes if not args.add_mask else 4*args.num_planes).permute(0,3,1,2)


			# ----------- shuffle ------------------------
			if flag:
				# visualize layer
				for i in range(args.num_planes):
					# mask_imgL = imgL*all_maskL[i]
					# out_L_shift = out_L[:, 3*i:3*i+3, :, base_shift - offsets[i]:base_shift + w_res - offsets[i]]
					# saveimg(mask_imgL, f"{save_imgpath}/{i}_maskL_{iter}.png")
					# saveimg(out_L_shift, f"{save_imgpath}/{i}_outL_shift_{iter}.png")
					if args.add_mask:

						saveimg(out_L[:, 4*i:4*i+3], f"{save_imgpath}/{i}_outL_{iter}.png")
						saveimg(out[:, 4*i:4*i+3], f"{save_imgpath}/{i}_out_{iter}.png")

						saveimg(out_L[:, 4*i+3:4*i+4].repeat(1,3,1,1), f"{save_imgpath}/{i}_maskL_{iter}.png")
						saveimg(out[:, 4*i+3:4*i+4].repeat(1,3,1,1), f"{save_imgpath}/{i}_mask_{iter}.png")

					else:

						saveimg(out_L[:, 3*i:3*i+3], f"{save_imgpath}/{i}_outL_{iter}.png")
						saveimg(out[:, 3*i:3*i+3], f"{save_imgpath}/{i}_out_{iter}.png")
			else:
				# visualize layer
				for i in range(args.num_planes):

					# mask_imgR = imgR*all_maskR[i]
					# out_R_shift = out_R[:, 3*i:3*i+3, :, base_shift + offsets[i]:base_shift + w_res + offsets[i]]
					# saveimg(mask_imgR, f"{save_imgpath}/{i}_maskR_{iter}.png")
					# saveimg(out_R_shift, f"{save_imgpath}/{i}_outR_shift_{iter}.png")

					if args.add_mask:

						saveimg(out_R[:, 4*i:4*i+3], f"{save_imgpath}/{i}_outR_{iter}.png")
						saveimg(out[:, 4*i:4*i+3], f"{save_imgpath}/{i}_out_{iter}.png")

						saveimg(out_R[:, 4*i+3:4*i+4].repeat(1,3,1,1), f"{save_imgpath}/{i}_maskR_{iter}.png")
						saveimg(out[:, 4*i+3:4*i+4].repeat(1,3,1,1), f"{save_imgpath}/{i}_mask_{iter}.png")

					else:
						saveimg(out_R[:, 3*i:3*i+3], f"{save_imgpath}/{i}_outR_{iter}.png")
						saveimg(out[:, 3*i:3*i+3], f"{save_imgpath}/{i}_out_{iter}.png")



		iter +=1



def separate_train(args):

	layer_n = args.separate_layer

	device = "cuda"

	save_imgpath = os.path.join(args.savepath, f"imgs{layer_n}")
	save_modelpath = os.path.join(args.savepath, f"ckpt{layer_n}")

	if not os.path.exists(save_imgpath):
		os.makedirs(save_imgpath)

	if not os.path.exists(save_modelpath):
		os.makedirs(save_modelpath)

	h_res = args.resolution[0]
	w_res = args.resolution[1]

	L_tag = torch.tensor([0]).cuda()
	R_tag = torch.tensor([1]).cuda()

	## ------------------------ no use dataloader -------------------

	########### load image 
	imgL_name = f"rgba_00000.png"
	imgR_name = f"rgba_00011.png"
	imgL_pil = Image.open(join(args.datapath, imgL_name)).convert("RGB")
	imgR_pil = Image.open(join(args.datapath,imgR_name)).convert("RGB")

	imgL = torch.from_numpy(np.array(imgL_pil)).permute(2,0,1)/255.
	imgR = torch.from_numpy(np.array(imgR_pil)).permute(2,0,1)/255.

	imgL = imgL.unsqueeze(0).to(device)
	imgR = imgR.unsqueeze(0).to(device)

	########### load mask 

	maskL_name = f"warpmask{layer_n}_00000.png"
	maskR_name = f"warpmask{layer_n}_00011.png"

	rgbL_name = f"warprgb{layer_n}_00000.png"
	rgbR_name = f"warprgb{layer_n}_00011.png"

	# left
	maskL = np.array(Image.open(join(args.datapath,maskL_name))).astype(np.float32)
	rgbL = np.array(Image.open(join(args.datapath,rgbL_name))).astype(np.float32)
	maskL = torch.from_numpy(maskL).permute(2,0,1)[0,:,:]/255.
	rgbL = torch.from_numpy(rgbL).permute(2,0,1)/255.

	# right
	maskR = np.array(Image.open(join(args.datapath,maskR_name))).astype(np.float32)
	rgbR = np.array(Image.open(join(args.datapath,rgbR_name))).astype(np.float32)
	maskR = torch.from_numpy(maskR).permute(2,0,1)[0,:,:]/255.
	rgbR = torch.from_numpy(rgbR).permute(2,0,1)/255.

	rgbR = rgbR.to(device)
	maskR = maskR.to(device)	
	rgbL = rgbL.to(device)
	maskL = maskL.to(device)

	########### load disparity
	disp = np.load(os.path.join(args.datapath,"disp_0_11.npy"))


	# ------------------------ start processing -----------------------

	# all_maskL = all_maskL.to(device)
	# all_maskR = all_maskR.to(device)
	disp = torch.from_numpy(disp).to(device)

	mask_imgL = rgbL
	saveimg(mask_imgL, f"{save_imgpath}/{layer_n}_outL_gt.png")

	mask_imgR = rgbR
	saveimg(mask_imgR, f"{save_imgpath}/{layer_n}_outR_gt.png")

	saveimg(maskL.unsqueeze(0).repeat(3,1,1), f"{save_imgpath}/{layer_n}_maskL_gt.png")
	saveimg(maskR.unsqueeze(0).repeat(3,1,1), f"{save_imgpath}/{layer_n}_maskR_gt.png")


	# compute offsets, baseline (scene-dependent)
	epsilon=1e-2
	disp = torch.abs(disp[:,:,0])
	max_disp = torch.max(disp)
	min_disp = torch.min(disp)
	planes = torch.round(torch.linspace(min_disp-epsilon, max_disp+epsilon, args.num_planes+1)/2)*2
	base_shift = int(max_disp//2)
	offsets = [ int((planes[i]/2+planes[i+1]/2)//2) for i in range(args.num_planes)]

	# print(f"offsets: {offsets}, baseshift: {base_shift}")

	coords_h = np.linspace(-1, 1, h_res, endpoint=False)
	coords_w = np.linspace(-1, 1,  w_res + base_shift * 2, endpoint=False)
	# coords_w = np.linspace(-1, 1,  w_res, endpoint=False)
	xy_grid = np.stack(np.meshgrid(coords_w, coords_h), -1)
	xy_grid = torch.FloatTensor(xy_grid).cuda()
	grid_inp = xy_grid.view(-1, 2).contiguous().unsqueeze(0)

	# ----------------------- define network --------------------------------------
	if args.use_viinter:
		net_scene = VIINTER(n_emb = 2, norm_p = 1, inter_fn=linterp, D=args.n_layer, z_dim = 128, in_feat=2, out_feat= 4, W=args.n_c, with_res=False, with_norm=True).cuda()
	else:
		net_scene = CondSIREN(n_emb = 2, norm_p = 1, inter_fn=linterp, D=args.n_layer, z_dim = 128, in_feat=2, out_feat=4, W=args.n_c, with_res=False, with_norm=args.use_norm, use_sig=args.use_sigmoid).cuda()
	
	if args.use_viinter:
		optimizer = torch.optim.Adam(net_scene.parameters(), lr=1e-5)
	else:
		optimizer = torch.optim.Adam(net_scene.parameters(), lr=args.lr)

	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = int(args.num_iters*0.2), gamma = 0.5)

	print("net_scene: ", net_scene)

	# for metric
	metric_l1 = nn.L1Loss()
	metric_mse = nn.MSELoss()

	# for metric
	if args.w_vgg>0:
		metric_vgg = VGGLoss()

	mseloss = 0
	vgloss = 0
	blend_loss = 0
	inter_loss = 0

	xp = [0, 1]
	fp = [-2, 2]

	iter = 0
	shuffle = True

	while iter < args.num_iters+1:

		mseloss = torch.tensor(0.).cuda()
		vgloss = torch.tensor(0.).cuda()
		scene_loss = torch.tensor(0.).cuda()

		# ----------- shuffle ------------------------
		flag = random.choice([True, False])
		# flag = True

		if flag:

			out_L = net_scene(grid_inp, L_tag)
			out_L = out_L.reshape(1, h_res, -1, 4).permute(0,3,1,2)

			## rgb
			out_L_shift = out_L[:, 0:3, :, base_shift - offsets[layer_n]:base_shift + w_res - offsets[layer_n]]

			mask_imgL = rgbL

			if args.out_black:
				mseloss = metric_mse(mask_imgL, out_L_shift)
				scene_loss = scene_loss + mseloss

				if args.w_vgg>0:
					if args.recon_vg:
						recon_out_vg = imgL*(1-maskL) + out_L_shift*(maskL)
						recon_gt_vg = imgL*(1-maskL) + mask_imgL*(maskL)

						gt_vg = VGGpreprocess(recon_gt_vg)
						out_vg = VGGpreprocess(recon_out_vg)
						vgloss = metric_vgg(gt_vg, out_vg) * args.w_vgg
					else:
						mask_imgL_vg = VGGpreprocess(mask_imgL.unsqueeze(0))
						out_L_shift_vg = VGGpreprocess(out_L_shift)
						vgloss = metric_vgg(mask_imgL_vg, out_L_shift_vg) * args.w_vgg

					scene_loss = scene_loss + vgloss
			else:
				scene_loss = scene_loss + metric_mse(mask_imgL, out_L_shift*maskL)

			## mask
			mask_L_shift = out_L[:, 3:4, :, base_shift - offsets[layer_n]:base_shift + w_res - offsets[layer_n]]
			scene_loss = scene_loss + metric_mse(mask_L_shift, maskL)


		else:
			out_R = net_scene(grid_inp, R_tag)
			out_R = out_R.reshape(1, h_res, -1, 4).permute(0,3,1,2)

			## rgb
			out_R_shift = out_R[:, 0:3, :, base_shift + offsets[layer_n]:base_shift + w_res + offsets[layer_n]]

			mask_imgR = rgbR

			if args.out_black:
				mseloss = metric_mse(mask_imgR, out_R_shift)
				scene_loss = scene_loss + mseloss

				if args.w_vgg>0:
					if args.recon_vg:
						recon_out_vg = imgR*(1-maskR) + out_R_shift*maskR
						recon_gt_vg = imgR*(1-maskR) + mask_imgR*maskR
						gt_vg = VGGpreprocess(recon_gt_vg)
						out_vg = VGGpreprocess(recon_out_vg)
						vgloss = metric_vgg(gt_vg, out_vg) * args.w_vgg
					else:
						mask_imgR_vg = VGGpreprocess(mask_imgR.unsqueeze(0))
						out_R_shift_vg = VGGpreprocess(out_R_shift)
						vgloss = metric_vgg(mask_imgR_vg, out_R_shift_vg) * args.w_vgg

					scene_loss = scene_loss + vgloss
			else:
				scene_loss = scene_loss + metric_mse(mask_imgR, out_R_shift*maskR)

			## mask
			mask_R_shift = out_R[:, 3:4, :, base_shift + offsets[layer_n]:base_shift + w_res + offsets[layer_n]]
			scene_loss = scene_loss + metric_mse(mask_R_shift, maskR)


			# ----------- end of shuffle ------------------------
			
		optimizer.zero_grad()
		scene_loss.backward()
		optimizer.step()
		scheduler.step()


		if (iter % args.progress_iter) ==0:

			lr = optimizer.param_groups[0]['lr']
			print(f"iterations: {iter}, lr: {lr}, mseloss: {mseloss}, vgloss: {vgloss}, scene_loss: {scene_loss}, inter_loss: {inter_loss}")
			print(f"offset: {offsets}, base_shift: {base_shift}, planes: {planes},")

			save_dict = { 
						'mlp':net_scene.state_dict(), \
						'num_planes':args.num_planes, \
						}
			torch.save(save_dict, "%s/model.ckpt"%(save_modelpath))
			# save_progress() if not args.debug else save_progress_debug(model_out, gt_img, iter)

			with torch.no_grad():

				z0 = net_scene.ret_z(torch.LongTensor([0.]).cuda()).squeeze()
				z1 = net_scene.ret_z(torch.LongTensor([1.]).cuda()).squeeze()

				zi = linterp(0.5, z0, z1).unsqueeze(0)
				out = net_scene.forward_with_z(grid_inp, zi)
				out = out.reshape(1, h_res, -1, 4).permute(0,3,1,2)


			# ----------- shuffle ------------------------
			if flag:

				saveimg(out_L_shift[:, 0:3], f"{save_imgpath}/{layer_n}_outL_{iter}.png")
				saveimg(out[:, 0:3], f"{save_imgpath}/{layer_n}_out_{iter}.png")

				saveimg(mask_L_shift[:, 0:1].repeat(1,3,1,1), f"{save_imgpath}/{layer_n}_maskL_{iter}.png")
				saveimg(out[:, 3:4].repeat(1,3,1,1), f"{save_imgpath}/{layer_n}_mask_{iter}.png")

			else:

				saveimg(out_R_shift[:, 0:3], f"{save_imgpath}/{layer_n}_outR_{iter}.png")
				saveimg(out[:, 0:3], f"{save_imgpath}/{layer_n}_out_{iter}.png")

				saveimg(mask_R_shift[:, 0:1].repeat(1,3,1,1), f"{save_imgpath}/{layer_n}_maskR_{iter}.png")
				saveimg(out[:, 3:4].repeat(1,3,1,1), f"{save_imgpath}/{layer_n}_mask_{iter}.png")

			if args.recon_vg:
				saveimg(recon_out_vg, f"{save_imgpath}/{layer_n}_outvg_{iter}.png")
				saveimg(recon_gt_vg, f"{save_imgpath}/{layer_n}_gtvg_{iter}.png")

		iter +=1




def hyper_train(args):

	from hypernetwork import CondSIREN_meta, HyperNetwork

	device = "cuda"

	save_imgpath = os.path.join(args.savepath, "imgs")
	save_modelpath = os.path.join(args.savepath, "ckpt")

	if not os.path.exists(save_imgpath):
		os.makedirs(save_imgpath)

	if not os.path.exists(save_modelpath):
		os.makedirs(save_modelpath)

	h_res = args.resolution[0]
	w_res = args.resolution[1]

	scalar = 0.1

	L_tag = torch.tensor([0.]).cuda()*scalar
	R_tag = torch.tensor([1.]).cuda()*scalar

	## ------------------------ no use dataloader -------------------

	all_ids = [0,1,2,3,4,5,6,7,8,9,10,11]

	R_id = 11
	L_id = 0
	dist = 1/(R_id - L_id)
	inter_list = [np.float32(i*dist) for i in range(R_id - L_id+1)]


	########### load image 
	imgL_name = f"rgba_00000.png"
	imgR_name = f"rgba_00011.png"
	imgL_pil = Image.open(join(args.datapath, imgL_name)).convert("RGB")
	imgR_pil = Image.open(join(args.datapath,imgR_name)).convert("RGB")

	# all_testimgs = []
	# for rand_id in all_ids:
	# 	imgtest_name = f"rgba_{rand_id:05d}.png"
	# 	imgtest_pil = Image.open(join(args.datapath, imgtest_name)).convert("RGB")
	# 	imgtest = torch.from_numpy(np.array(imgtest_pil)).permute(2,0,1)/255.
	# 	all_testimgs.append(imgtest)
	imgL = torch.from_numpy(np.array(imgL_pil)).permute(2,0,1)/255.
	imgR = torch.from_numpy(np.array(imgR_pil)).permute(2,0,1)/255.


	########### load mask 
	all_maskL = []
	all_maskR = []
	for i in range(args.num_planes):
		# left
		maskL_name = f"mask{i}_00000.png"
		maskL = np.array(Image.open(join(args.datapath,maskL_name))).astype(np.float32)
		maskL = torch.from_numpy(maskL)#/255.
		# right
		maskR_name = f"mask{i}_00011.png"
		maskR = np.array(Image.open(join(args.datapath,maskR_name))).astype(np.float32)
		maskR = torch.from_numpy(maskR)#/255.	
		all_maskL.append(maskL)		
		all_maskR.append(maskR)		
	all_maskL = torch.stack(all_maskL)
	all_maskR = torch.stack(all_maskR)

	########### load disparity
	disp = np.load(os.path.join(args.datapath,"disp_0_11.npy"))


	# ------------------------ start processing -----------------------

	imgL = imgL.to(device)
	imgR = imgR.to(device)

	all_maskL = all_maskL.to(device)
	all_maskR = all_maskR.to(device)
	disp = torch.from_numpy(disp).to(device)

	for i in range(args.num_planes):
		mask_imgL = imgL*all_maskL[i]
		saveimg(mask_imgL, f"{save_imgpath}/{i}_outL_gt.png")

		mask_imgR = imgR*all_maskR[i]
		saveimg(mask_imgR, f"{save_imgpath}/{i}_outR_gt.png")

		if args.add_mask:
			saveimg(all_maskL[i].unsqueeze(0).repeat(3,1,1), f"{save_imgpath}/{i}_maskL_gt.png")
			saveimg(all_maskR[i].unsqueeze(0).repeat(3,1,1), f"{save_imgpath}/{i}_maskR_gt.png")


	# compute offsets, baseline (scene-dependent)
	disp = torch.abs(disp[:,:,0])
	max_disp = torch.max(disp)
	min_disp = torch.min(disp)
	planes = torch.round(torch.linspace(min_disp, max_disp, args.num_planes+1)/2)*2
	base_shift = int(max_disp//2)
	offsets = [ int((planes[i]/2+planes[i+1]/2)//2) for i in range(args.num_planes)]

	# print(f"offsets: {offsets}, baseshift: {base_shift}")

	coords_h = np.linspace(-1, 1, h_res, endpoint=False)
	coords_w = np.linspace(-1, 1,  w_res + base_shift * 2, endpoint=False)
	# coords_w = np.linspace(-1, 1,  w_res, endpoint=False)
	xy_grid = np.stack(np.meshgrid(coords_w, coords_h), -1)
	xy_grid = torch.FloatTensor(xy_grid).cuda()
	grid_inp = xy_grid.view(-1, 2).contiguous().unsqueeze(0)

	# ----------------------- define network --------------------------------------

	if args.debug:
		from metasiren import MetaSirenGrid
		import warnings
		warnings.filterwarnings("ignore", category=UserWarning) 

		net_scene = MetaSirenGrid(args, w_res=w_res + base_shift * 2, h_res=h_res, hidden_features=128, hidden_layers=5, out_features=3*args.num_planes if not args.add_mask else 4*args.num_planes, outermost_linear=True, first_omega_0=10, hidden_omega_0=10).to(device).train()
	else:
		if args.mask_only:
			net_scene = CondSIREN_meta(D=args.n_layer, in_feat=2, out_feat=args.num_planes , W=args.n_c, with_res=False).cuda()
		else:
			net_scene = CondSIREN_meta(D=args.n_layer, in_feat=2, out_feat=3*args.num_planes if not args.add_mask else 4*args.num_planes, W=args.n_c, with_res=False).cuda()
	

	hypernet = HyperNetwork(hyper_in_features=1, hyper_hidden_layers=3, hyper_hidden_features=128, hypo_module=net_scene).cuda()


	optimizer = torch.optim.Adam(hypernet.parameters(), lr=1e-6)

	# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_iters, eta_min=args.lr*0.1)


	print("net_scene: ", net_scene)

	# for metric
	metric_l1 = nn.L1Loss()
	metric_mse = nn.MSELoss()

	# for metric
	if args.w_vgg>0:
		metric_vgg = VGGLoss()

	mseloss = 0
	vgloss = 0
	blend_loss = 0
	inter_loss = 0

	xp = [0, 1]
	fp = [-2, 2]

	iter = 0
	shuffle = True

	while iter < args.num_iters:

		rand_id = random.choice(all_ids) 
		inter = inter_list[rand_id]

		mseloss = torch.tensor(0.).cuda()
		vgloss = torch.tensor(0.).cuda()
		scene_loss = torch.tensor(0.).cuda()

		# ----------- shuffle ------------------------
		flag = random.choice([True, False])
		# flag = True

		if flag:

			params_model = hypernet(L_tag)
			if not args.debug:
				out_L = net_scene(grid_inp, params=params_model)
			else:
				out_L = net_scene(params=params_model) 
			if args.mask_only:
				out_L = out_L.reshape(1, h_res, -1, args.num_planes).permute(0,3,1,2)
			else:
				out_L = out_L.reshape(1, h_res, -1, 3*args.num_planes if not args.add_mask else 4*args.num_planes).permute(0,3,1,2)
		

			for i in range(args.num_planes):
				if args.mask_only:
					mask_L_shift = out_L[:, i:i+1, :, base_shift - offsets[i]:base_shift + w_res - offsets[i]]
					scene_loss = scene_loss + metric_mse(mask_L_shift, all_maskL[i])
				elif args.add_mask:
					mask_imgL = imgL*all_maskL[i]
					out_L_shift = out_L[:, 4*i:4*i+3, :, base_shift - offsets[i]:base_shift + w_res - offsets[i]]
					# print("out_L_shift: ", out_L_shift.shape, mask_imgL.shape)

					scene_loss = scene_loss + metric_mse(mask_imgL, out_L_shift*all_maskL[i])
					mask_L_shift = out_L[:, 4*i+3:4*i+4, :, base_shift - offsets[i]:base_shift + w_res - offsets[i]]
					scene_loss = scene_loss + metric_mse(mask_L_shift, all_maskL[i])
				else:
					mask_imgL = imgL*all_maskL[i]
					out_L_shift = out_L[:, 3*i:3*i+3, :, base_shift - offsets[i]:base_shift + w_res - offsets[i]]
					scene_loss = scene_loss + metric_mse(mask_imgL, out_L_shift*all_maskL[i])
		else:

			params_model = hypernet(R_tag)
			if not args.debug:
				out_R = net_scene(grid_inp, params=params_model)
			else:
				out_R = net_scene(params=params_model) 
			if args.mask_only:
				out_R = out_R.reshape(1, h_res, -1, args.num_planes).permute(0,3,1,2)
			else:
				out_R = out_R.reshape(1, h_res, -1, 3*args.num_planes if not args.add_mask else 4*args.num_planes).permute(0,3,1,2)

			for i in range(args.num_planes):

				if args.mask_only:
					mask_R_shift = out_R[:, i:i+1, :, base_shift - offsets[i]:base_shift + w_res - offsets[i]]
					scene_loss = scene_loss + metric_mse(mask_R_shift, all_maskR[i])
				elif args.add_mask:
					mask_imgR = imgR*all_maskR[i]
					out_R_shift = out_R[:, 4*i:4*i+3, :, base_shift + offsets[i]:base_shift + w_res + offsets[i]]
					scene_loss = scene_loss + metric_mse(mask_imgR, out_R_shift*all_maskR[i])
					mask_R_shift = out_R[:, 4*i+3:4*i+4, :, base_shift + offsets[i]:base_shift + w_res + offsets[i]]
					scene_loss = scene_loss + metric_mse(mask_R_shift, all_maskR[i])
				else:
					mask_imgR = imgR*all_maskR[i]
					out_R_shift = out_R[:, 3*i:3*i+3, :, base_shift + offsets[i]:base_shift + w_res + offsets[i]]
					scene_loss = scene_loss + metric_mse(mask_imgR, out_R_shift*all_maskR[i])

			# ----------- end of shuffle ------------------------
			
		optimizer.zero_grad()
		scene_loss.backward()
		optimizer.step()
		# scheduler.step()


		if (iter % 50000) ==0:

			print(f"iterations: {iter}, interval: {inter}, blend_loss: {blend_loss}, mseloss: {mseloss}, vgloss: {vgloss}, scene_loss: {scene_loss}, inter_loss: {inter_loss}")
			print(f"offset: {offsets}, base_shift: {base_shift}, planes: {planes},")

			save_dict = { 
						'hypernet':hypernet.state_dict(), \
						}
			torch.save(save_dict, "%s/model.ckpt"%(save_modelpath))
			# save_progress() if not args.debug else save_progress_debug(model_out, gt_img, iter)

			with torch.no_grad():

				inter_tag = torch.tensor([inter]).cuda()
				params_model = hypernet(inter_tag*scalar)
				if not args.debug:
					out = net_scene(grid_inp, params=params_model)
				else:
					out = net_scene(params=params_model) 
				if args.mask_only:
					out = out.reshape(1, h_res, -1, args.num_planes).permute(0,3,1,2)
				else:
					out = out.reshape(1, h_res, -1, 3*args.num_planes if not args.add_mask else 4*args.num_planes).permute(0,3,1,2)

			# ----------- shuffle ------------------------
			if flag:
				# visualize layer
				for i in range(args.num_planes):
					# mask_imgL = imgL*all_maskL[i]
					# out_L_shift = out_L[:, 3*i:3*i+3, :, base_shift - offsets[i]:base_shift + w_res - offsets[i]]
					# saveimg(mask_imgL, f"{save_imgpath}/{i}_maskL_{iter}.png")
					# saveimg(out_L_shift, f"{save_imgpath}/{i}_outL_shift_{iter}.png")
					if args.mask_only:
					
						saveimg(out_L[:, i:i+1].repeat(1,3,1,1), f"{save_imgpath}/{i}_maskL_{iter}.png")
						saveimg(out[:, i:i+1].repeat(1,3,1,1), f"{save_imgpath}/{i}_mask_{iter}.png")

					elif args.add_mask:

						saveimg(out_L[:, 4*i:4*i+3], f"{save_imgpath}/{i}_outL_{iter}.png")
						saveimg(out[:, 4*i:4*i+3], f"{save_imgpath}/{i}_out_{iter}.png")

						saveimg(out_L[:, 4*i+3:4*i+4].repeat(1,3,1,1), f"{save_imgpath}/{i}_maskL_{iter}.png")
						saveimg(out[:, 4*i+3:4*i+4].repeat(1,3,1,1), f"{save_imgpath}/{i}_mask_{iter}.png")

					else:

						saveimg(out_L[:, 3*i:3*i+3], f"{save_imgpath}/{i}_outL_{iter}.png")
						saveimg(out[:, 3*i:3*i+3], f"{save_imgpath}/{i}_out_{iter}.png")
			else:
				# visualize layer
				for i in range(args.num_planes):

					# mask_imgR = imgR*all_maskR[i]
					# out_R_shift = out_R[:, 3*i:3*i+3, :, base_shift + offsets[i]:base_shift + w_res + offsets[i]]
					# saveimg(mask_imgR, f"{save_imgpath}/{i}_maskR_{iter}.png")
					# saveimg(out_R_shift, f"{save_imgpath}/{i}_outR_shift_{iter}.png")
					if args.mask_only:
					
						saveimg(out_R[:, i:i+1].repeat(1,3,1,1), f"{save_imgpath}/{i}_maskR_{iter}.png")
						saveimg(out[:, i:i+1].repeat(1,3,1,1), f"{save_imgpath}/{i}_mask_{iter}.png")
					elif args.add_mask:

						saveimg(out_R[:, 4*i:4*i+3], f"{save_imgpath}/{i}_outR_{iter}.png")
						saveimg(out[:, 4*i:4*i+3], f"{save_imgpath}/{i}_out_{iter}.png")

						saveimg(out_R[:, 4*i+3:4*i+4].repeat(1,3,1,1), f"{save_imgpath}/{i}_maskR_{iter}.png")
						saveimg(out[:, 4*i+3:4*i+4].repeat(1,3,1,1), f"{save_imgpath}/{i}_mask_{iter}.png")

					else:
						saveimg(out_R[:, 3*i:3*i+3], f"{save_imgpath}/{i}_outR_{iter}.png")
						saveimg(out[:, 3*i:3*i+3], f"{save_imgpath}/{i}_out_{iter}.png")



		iter +=1


	
if __name__=='__main__':
	

	parser = argparse.ArgumentParser()

	parser.add_argument('--nfg',type=int,help='capacity multiplier',default = 8)
	parser.add_argument('--num_planes',type=int,help='number of planes',default = 6)
	parser.add_argument('--num_freqs_pe', type=int,help='#frequencies for positional encoding',default = 5)
	parser.add_argument('--num_iters',type=int,help='num of iterations',default = 250000)
	parser.add_argument('--progress_iter',type=int,help='update frequency',default = 50000)
	parser.add_argument('--lr',type=float,help='learning rate',default = 1e-4)
	parser.add_argument('--savepath',type=str,help='saving path',default = 'resultsTest1')
	parser.add_argument('--w_vgg',help='weight of loss',type=float,default = 0.0)
	parser.add_argument('--w_l1',help='weight of l1 loss',type=float,default = 1.0)
	parser.add_argument('--w_multi',help='weight of multi constraints loss',type=float,default = 0.5)
	parser.add_argument('--n_layer',help='layer number of meta MLP',type=int,default = 5)
	parser.add_argument('--max_disp',help='max_disp for shifring',type=int,default = 10)
	parser.add_argument('--resolution',help='resolution [h,w]',nargs='+',type=int,default = [360, 640])
	parser.add_argument('--use_norm',help='use my network for debugging',action='store_true')
	parser.add_argument('--add_mask',help='output mask',action='store_true')
	parser.add_argument('--use_sigmoid',help='add sigmoid to the end of CondSiren',action='store_true')
	parser.add_argument('--no_lr_cons',help='use my network for debugging',action='store_true')
	parser.add_argument('--load_one',help='load one scene only',action='store_true')
	parser.add_argument('--no_constraints',help='not add constraints',action='store_true')
	parser.add_argument('--no_inter_cons',help='no using outer loss, inner only',action='store_true')
	parser.add_argument('--no_multi_cons',help='no using outer loss, inner only',action='store_true')
	parser.add_argument('--reg_train',help='regular training, for debugging',action='store_true')
	parser.add_argument('--use_viinter',help='use viinter network or not',action='store_true')
	parser.add_argument('--use_tcnn',help='use tcnn network or not',action='store_true')
	parser.add_argument('--debug',help='use tcnn network or not',action='store_true')
	parser.add_argument('--warp_data',help='use warpped dataset',action='store_true')
	parser.add_argument('--out_black',help='output contains black part',action='store_true')
	parser.add_argument('--hypernet',help='use hyper network or not',action='store_true')
	parser.add_argument('--recon_vg',help='recon_vg',action='store_true')
	parser.add_argument('--separate_layer',help='seperate layers',type=int,default=-1)
	parser.add_argument('--mask_only',help='output maskonly',action='store_true')
	parser.add_argument('--n_c',help='number of channel in netMet',type=int,default = 256)
	parser.add_argument('--batch_size',help='batch size ',type=int,default = 1)
	parser.add_argument('--datapath',help='the path of training dataset',type=str,default="../../Dataset/SynData_s1_all")
	parser.add_argument('--config',help='the config path',type=str,default="./config/config.json")
	args = parser.parse_args()



	save_args(args)

	if not os.path.exists(args.savepath):
		os.makedirs(args.savepath)


	if args.hypernet:
		hyper_train(args)
	else:
		if args.separate_layer >= 0:
			separate_train(args)
		else:
			regular_train(args)