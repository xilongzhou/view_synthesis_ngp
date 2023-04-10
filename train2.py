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

from network import VIINTER, linterp, Unet_Blend, MLP_blend, CondSIREN, CondSIREN_meta, tiny_cuda
import torch.nn.functional as F

import random

# from torchmeta.utils.gradient_based import gradient_update_parameters
from collections import OrderedDict

from my_dataloader import DataLoader_helper, DataLoader_helper2

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

	# filter args
	if args.no_blend:
		args.no_inter_cons = True


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
	# max_disp = 0#args.max_disp
	# planes = np.rint(np.linspace(0, max_disp, args.num_planes)/2)*2
	# offsets = (planes//2).astype(int)
	# base_shift = max_disp//2
	# print(f"offset: {offsets}, base_shift: {base_shift}")

	L_tag = torch.tensor([0]).cuda()
	R_tag = torch.tensor([1]).cuda()

	# # ---------------------- Dataloader ----------------------
	# Myset = DataLoader_helper(args.datapath, h_res, w_res)
	Myset = DataLoader_helper2(args.datapath, h_res, w_res, one_scene=args.load_one)
	Mydata = DataLoader(dataset=Myset, batch_size=args.batch_size, shuffle=True, drop_last=True)

	total_data = Myset.get_total_num()

	print(f"length of data is {total_data}")


	# ----------------------- define network --------------------------------------


	# common net
	if not args.no_blend:
		net_blend = MLP_blend(D=args.mlp_d, in_feat=3*args.num_planes, out_feat=3, W=args.mlp_w, with_res=False, with_norm=args.use_norm).cuda()


	if args.use_viinter:
		net_scene = VIINTER(n_emb = 2, norm_p = 1, inter_fn=linterp, D=args.n_layer, z_dim = 128, in_feat=2, out_feat=3*args.num_planes, W=args.n_c, with_res=False, with_norm=True).cuda()
	elif args.use_tcnn:

		with open(args.config) as config_file:
			config = json.load(config_file)

		net_scene = tiny_cuda(n_emb = 2, config=config ,norm_p = 1, out_feat=3*args.num_planes, debug=args.debug).cuda()
	else:
		net_scene = CondSIREN(n_emb = 2, norm_p = 1, inter_fn=linterp, D=args.n_layer, z_dim = 128, in_feat=2, out_feat=3*args.num_planes, W=args.n_c, with_res=False, with_norm=args.use_norm, use_sig=args.use_sigmoid).cuda()
	
	if args.use_tcnn:
		optimizer = torch.optim.Adam(net_scene.parameters(), lr=0.01)
	else:
		optimizer = torch.optim.Adam(net_scene.parameters(), lr=args.lr)


	# if not args.no_lr_cons and not args.no_inter_cons:
	# 	print("optimize blend and mlp seperately")
	# 	optimizer_blend = torch.optim.Adam(net_blend.parameters(), lr=args.lr)
	# 	optimizer = torch.optim.Adam(net_scene.parameters(), lr=args.lr)
	# else:
	# 	print("optimize blend and mlp together")
	# 	if args.no_blend:
	# 		optimizer = torch.optim.Adam(net_scene.parameters(), lr=args.lr)
	# 	else:
	# 		optimizer = torch.optim.Adam(list(net_scene.parameters()) + list(net_blend.parameters()), lr=args.lr)

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

		# this is for outerloop
		for i,data in enumerate(Mydata):
			imgL_b, imgR_b, imgtest_b, inter_val_b, index_b, all_maskL_b, all_maskR_b, disp_b = data
			# imgL_b, imgR_b, imgtest_b, inter_val_b, index_b = data

		# if True:
		# 	inter = random.randint(0,11)
		# 	# inter = random.choice([0,11]) 
		# 	imgtest = imgs_dict[inter]
		# 	inter = torch.from_numpy(np.array(inter))

			mseloss = torch.tensor(0.).cuda()
			vgloss = torch.tensor(0.).cuda()
			scene_loss = torch.tensor(0.).cuda()

			for task_id in range(args.batch_size):

				# --------------- fetch --------------------
				imgtest = imgtest_b[task_id:task_id+1].to(device)
				inter = inter_val_b[task_id:task_id+1]
				imgL = imgL_b[task_id:task_id+1].to(device)
				imgR = imgR_b[task_id:task_id+1].to(device)
				index = index_b[task_id]
				all_maskL = all_maskL_b[task_id].to(device)
				all_maskR = all_maskR_b[task_id].to(device)
				disp = disp_b[task_id].to(device)

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
				# if not args.rowbatch:
				grid_inp = xy_grid.view(-1, 2).contiguous().unsqueeze(0)

				dx = torch.from_numpy(coords_w).float()
				dy = torch.from_numpy(coords_h).float()
				meshy, meshx = torch.meshgrid((dy, dx))

				# for debugging
				half_dx =  0.5 / (w_res + base_shift * 2)
				half_dy =  0.5 / h_res
				xs = torch.linspace(half_dx, 1-half_dx, (w_res + base_shift * 2), device=device)
				ys = torch.linspace(half_dy, 1-half_dy, h_res, device=device)
				xv, yv = torch.meshgrid([xs, ys])
				xy = torch.stack((yv.flatten(), xv.flatten())).t()

				# print("grid_inp: ", grid_inp[0:10])
				# print("xy: ", xy[0:10])

				# --------------- end fetch ------------------

				# index optimzer and network
				# if not args.reg_train:
				# 	net_scene = nets_scene[index]
				# 	optimizer = optimizers[index]

				if not args.no_lr_cons:

					if shuffle:
						# ----------- shuffle ------------------------
						# flag = random.choice([True, False])
						flag = True

						if flag:

							if not args.no_multi_cons:
								out_L = net_scene(grid_inp if not args.debug else xy, L_tag)
								out_L = out_L.reshape(1, h_res, -1, 3*args.num_planes).permute(0,3,1,2)
							
								for i in range(args.num_planes):
									mask_imgL = imgL*all_maskL[i]
									out_L_shift = out_L[:, 3*i:3*i+3, :, base_shift - offsets[i]:base_shift + w_res - offsets[i]]
									scene_loss = scene_loss + metric_mse(mask_imgL, out_L_shift*all_maskL[i])

							else:

								# left
								out_L = net_scene(grid_inp, L_tag)
								out_L = out_L.reshape(1, h_res, -1, 3*args.num_planes).permute(0,3,1,2)
								multi_out = [out_L[:, 3*i:3*i+3, :, base_shift - offsets[i]:base_shift + w_res - offsets[i]] for i in range(len(offsets))]
								if args.no_blend:
									blend_l = None
									for l in range(args.num_planes):
										if blend_l is None:
											blend_l = multi_out[l]*all_maskL[l]
										else:
											blend_l += multi_out[l]*all_maskL[l]
								else:
									multi_out = torch.cat(multi_out,dim=1)
									blend_l = net_blend(2*multi_out-1)
								
								loss_l = metric_mse(blend_l, imgL)
								if args.w_vgg>0:
									blend_l_vg = VGGpreprocess(blend_l)
									imgL_vg = VGGpreprocess(imgL)
									loss_l = loss_l + metric_vgg(blend_l_vg, imgL_vg) * args.w_vgg
							
								scene_loss = loss_l

						else:
							if not args.no_multi_cons:
								out_R = net_scene(grid_inp if not args.debug else xy, R_tag)
								out_R = out_R.reshape(1, h_res, -1, 3*args.num_planes).permute(0,3,1,2)

								for i in range(args.num_planes):
									mask_imgR = imgR*all_maskR[i]
									out_R_shift = out_R[:, 3*i:3*i+3, :, base_shift + offsets[i]:base_shift + w_res + offsets[i]]
									scene_loss = scene_loss + metric_mse(mask_imgR, out_R_shift*all_maskR[i])
							else:

								# right
								out_R = net_scene(grid_inp, R_tag)
								out_R = out_R.reshape(1, h_res, -1, 3*args.num_planes).permute(0,3,1,2)
								multi_out = [out_R[:, 3*i:3*i+3, :, base_shift + offsets[i]:base_shift + w_res + offsets[i]] for i in range(len(offsets))]
								if args.no_blend:
									blend_r = None
									for l in range(args.num_planes):
										if blend_r is None:
											blend_r = multi_out[l]*all_maskR[l]
										else:
											blend_r += multi_out[l]*all_maskR[l]
								else:
									multi_out = torch.cat(multi_out,dim=1)
									blend_r = net_blend(2*multi_out-1)

								loss_r = metric_mse(blend_r, imgR)
								if args.w_vgg>0:
									blend_r_vg = VGGpreprocess(blend_r)
									imgR_vg = VGGpreprocess(imgR)
									loss_r = loss_r + metric_vgg(blend_r_vg, imgR_vg) * args.w_vgg

								scene_loss = loss_r
					# ----------- end of shuffle ------------------------
					
					else:

						# left
						out_L = net_scene(grid_inp, L_tag)
						out_L = out_L.reshape(1, h_res, -1, 3*args.num_planes).permute(0,3,1,2)
						multi_out = [out_L[:, 3*i:3*i+3, :, base_shift - offsets[i]:base_shift + w_res - offsets[i]] for i in range(len(offsets))]
						if args.no_blend:
							blend_l = None
							for l in range(args.num_planes):
								if blend_l is None:
									blend_l = multi_out[l]*all_maskL[l]
								else:
									blend_l += multi_out[l]*all_maskL[l]
						else:
							multi_out = torch.cat(multi_out,dim=1)
							blend_l = net_blend(2*multi_out-1)
						loss_l = metric_mse(blend_l, imgL)
						if args.w_vgg>0:
							blend_l_vg = VGGpreprocess(blend_l)
							imgL_vg = VGGpreprocess(imgL)
							loss_l = loss_l + metric_vgg(blend_l_vg, imgL_vg) * args.w_vgg

						# right
						out_R = net_scene(grid_inp, R_tag)
						out_R = out_R.reshape(1, h_res, -1, 3*args.num_planes).permute(0,3,1,2)
						multi_out = [out_R[:, 3*i:3*i+3, :, base_shift + offsets[i]:base_shift + w_res + offsets[i]] for i in range(len(offsets))]
						if args.no_blend:
							blend_r = None
							for l in range(args.num_planes):
								if blend_r is None:
									blend_r = multi_out[l]*all_maskR[l]
								else:
									blend_r += multi_out[l]*all_maskR[l]
						else:
							multi_out = torch.cat(multi_out,dim=1)
							blend_r = net_blend(2*multi_out-1)
						loss_r = metric_mse(blend_r, imgR)
						if args.w_vgg>0:
							blend_r_vg = VGGpreprocess(blend_r)
							imgR_vg = VGGpreprocess(imgR)
							loss_r = loss_r + metric_vgg(blend_r_vg, imgR_vg) * args.w_vgg
						scene_loss = (loss_l + loss_r) * 0.5

					# # add constraints
					# inter_loss = 0
					# if not args.no_multi_cons:
					# 	for i in range(len(offsets)):

					# 		mask_imgL = imgL*all_maskL[i]
					# 		out_L_shift = out_L[:, 3*i:3*i+3, :, base_shift - offsets[i]:base_shift + w_res - offsets[i]]

					# 		mask_imgR = imgR*all_maskR[i]
					# 		out_R_shift = out_R[:, 3*i:3*i+3, :, base_shift + offsets[i]:base_shift + w_res + offsets[i]]

					# 		# print("mask_imgR: ", mask_imgR)
					# 		# print("out_R: ", out_R)

					# 		inter_loss += (metric_mse(mask_imgL, out_L_shift*all_maskL[i]) + metric_mse(mask_imgR, out_R_shift*all_maskR[i]))*args.w_multi


					# 	inter_loss = inter_loss/len(offsets)
					
					# scene_loss = scene_loss + inter_loss

					optimizer.zero_grad()
					scene_loss.backward()
					optimizer.step()
					# scheduler.step()

				# compute outer loss
				if not args.no_inter_cons:

					# # ------------ debug ---------------------
					# # left
					# out_L = net_scene(grid_inp, L_tag)
					# out_L = out_L.reshape(1, h_res, -1, 3*args.num_planes).permute(0,3,1,2)
					# multi_out = [out_L[:, 3*i:3*i+3, :, base_shift - offsets[i]:base_shift + w_res - offsets[i]] for i in range(len(offsets))]
					# multi_out = torch.cat(multi_out,dim=1)
					# blend_l = net_blend(2*multi_out-1)
					# loss_l = metric_mse(blend_l, imgL)
					# if args.w_vgg>0:
					# 	blend_l_vg = VGGpreprocess(blend_l)
					# 	imgL_vg = VGGpreprocess(imgL)
					# 	loss_l = loss_l + metric_vgg(blend_l_vg, imgL_vg) * args.w_vgg

					# # right
					# out_R = net_scene(grid_inp, R_tag)
					# out_R = out_R.reshape(1, h_res, -1, 3*args.num_planes).permute(0,3,1,2)
					# multi_out = [out_R[:, 3*i:3*i+3, :, base_shift + offsets[i]:base_shift + w_res + offsets[i]] for i in range(len(offsets))]
					# multi_out = torch.cat(multi_out,dim=1)
					# blend_r = net_blend(2*multi_out-1)
					# loss_r = metric_mse(blend_r, imgR)
					# if args.w_vgg>0:
					# 	blend_r_vg = VGGpreprocess(blend_r)
					# 	imgR_vg = VGGpreprocess(imgR)
					# 	loss_r = loss_r + metric_vgg(blend_r_vg, imgR_vg) * args.w_vgg
					
					# mseloss = loss_r
					# vgloss = loss_l
					# ------------ debug ---------------------

					z0 = net_scene.ret_z(torch.LongTensor([0.]).cuda()).squeeze()
					z1 = net_scene.ret_z(torch.LongTensor([1.]).cuda()).squeeze()
					zi = linterp(inter.to(device), z0, z1).unsqueeze(0)
					out = net_scene.forward_with_z(grid_inp, zi)
					out = out.reshape(1, h_res, -1, 3*args.num_planes).permute(0,3,1,2)
					off_scl = np.interp(inter, xp, fp)
					multi_out_list=[]
					for i in range(args.num_planes):
						meshxw = meshx + (base_shift * 2 + off_scl * offsets[i]) / (w_res + base_shift * 2)
						grid = torch.stack((meshxw, meshy), 2)[None].to(device).to(torch.float32)
						multi_out = grid_sample(out[:, 3*i:3*i+3], grid, mode='bilinear', align_corners=True)[:, :, :, :-base_shift*2]
						multi_out_list.append(multi_out)

						if args.no_blend:
							blend_test = None
							for l in range(args.num_planes):
								if blend_test is None:
									blend_test = multi_out[l]*all_maskR[l]
								else:
									blend_test += multi_out[l]*all_maskR[l]
						else:
							multi_out = torch.cat(multi_out_list,dim=1)
							blend_test = net_blend(2*multi_out-1) # multi out [0,1]-->[-1,1]

					mseloss += metric_mse(blend_test, imgtest)
					if args.w_vgg>0:
						blend_test_vg = VGGpreprocess(blend_test)
						imgtest_vg = VGGpreprocess(imgtest)
						vgloss += metric_vgg(blend_test_vg, imgtest_vg) * args.w_vgg

			if not args.no_inter_cons:
				mseloss.div_(args.batch_size)
				vgloss.div_(args.batch_size)
				blend_loss = mseloss + vgloss

				if not args.no_lr_cons:
					optimizer_blend.zero_grad()
					blend_loss.backward()
					optimizer_blend.step()
				else:
					optimizer.zero_grad()
					blend_loss.backward()
					optimizer.step()
					# scheduler.step()

			# save to new 
			# if not args.reg_train:
			# 	nets_scene[index] = net_scene
			# 	optimizers[index] = optimizer 

			if (iter % args.progress_iter) ==0:

				print(f"iterations: {iter}, interval: {inter}, blend_loss: {blend_loss}, mseloss: {mseloss}, vgloss: {vgloss}, scene_loss: {scene_loss}, inter_loss: {inter_loss}")
				print(f"offset: {offsets}, base_shift: {base_shift}, planes: {planes},")

				save_dict = { 
							'mlp':net_scene.state_dict(), \
							'nn_blend':net_blend.state_dict() if not args.no_blend else None, \
							'nfg':args.nfg, \
							'n_layer':args.n_layer, \
							'use_norm':args.use_norm, \
							'n_c':args.n_c, \
							'mlp_d':args.mlp_d, \
							'mlp_w':args.mlp_w, \
							'num_freqs_pe':args.num_freqs_pe, \
							'num_planes':args.num_planes, \
							'resolution':args.resolution, \
							}
				torch.save(save_dict, "%s/model.ckpt"%(save_modelpath))
				# save_progress() if not args.debug else save_progress_debug(model_out, gt_img, iter)

				if not args.debug:
					with torch.no_grad():

						z0 = net_scene.ret_z(torch.LongTensor([0.]).cuda()).squeeze()
						z1 = net_scene.ret_z(torch.LongTensor([1.]).cuda()).squeeze()

						zi = linterp(inter.to(device), z0, z1).unsqueeze(0)
						out = net_scene.forward_with_z(grid_inp, zi)
						out = out.reshape(1, h_res, -1, 3*args.num_planes).permute(0,3,1,2)

				if not args.no_lr_cons:

					# ----------- shuffle ------------------------
					if flag:
						# visualize layer
						for i in range(args.num_planes):
							# mask_imgL = imgL*all_maskL[i]
							# out_L_shift = out_L[:, 3*i:3*i+3, :, base_shift - offsets[i]:base_shift + w_res - offsets[i]]
							# saveimg(mask_imgL, f"{save_imgpath}/{i}_maskL_{iter}.png")
							# saveimg(out_L_shift, f"{save_imgpath}/{i}_outL_shift_{iter}.png")

							saveimg(out_L[:, 3*i:3*i+3], f"{save_imgpath}/{i}_outL_{iter}.png")
							if not args.debug:
								saveimg(out[:, 3*i:3*i+3], f"{save_imgpath}/{i}_out_{iter}.png")
					else:
						# visualize layer
						for i in range(args.num_planes):

							# mask_imgR = imgR*all_maskR[i]
							# out_R_shift = out_R[:, 3*i:3*i+3, :, base_shift + offsets[i]:base_shift + w_res + offsets[i]]
							# saveimg(mask_imgR, f"{save_imgpath}/{i}_maskR_{iter}.png")
							# saveimg(out_R_shift, f"{save_imgpath}/{i}_outR_shift_{iter}.png")
							saveimg(out_R[:, 3*i:3*i+3], f"{save_imgpath}/{i}_outR_{iter}.png")

							if not args.debug:
								saveimg(out[:, 3*i:3*i+3], f"{save_imgpath}/{i}_out_{iter}.png")



				if not args.no_inter_cons:
					saveimg(blend_test, f"{save_imgpath}/{iter}_blendtest.png")
					saveimg(imgtest, f"{save_imgpath}/{iter}_gttest.png")

			iter +=1



	
if __name__=='__main__':
	

	parser = argparse.ArgumentParser()

	parser.add_argument('--nfg',type=int,help='capacity multiplier',default = 8)
	parser.add_argument('--num_planes',type=int,help='number of planes',default = 6)
	parser.add_argument('--num_freqs_pe', type=int,help='#frequencies for positional encoding',default = 5)
	parser.add_argument('--num_iters',type=int,help='num of iterations',default = 500000)
	parser.add_argument('--progress_iter',type=int,help='update frequency',default = 10000)
	parser.add_argument('--lr',type=float,help='learning rate',default = 0.0001)
	parser.add_argument('--savepath',type=str,help='saving path',default = 'resultsTest1')
	parser.add_argument('--w_vgg',help='weight of loss',type=float,default = 0.0)
	parser.add_argument('--w_l1',help='weight of l1 loss',type=float,default = 1.0)
	parser.add_argument('--w_multi',help='weight of multi constraints loss',type=float,default = 0.5)
	parser.add_argument('--mlp_d',help='nnblend MLP layer number',type=int,default = 2)
	parser.add_argument('--mlp_w',help='channel of nnblend MLP',type=int,default = 16)
	parser.add_argument('--n_layer',help='layer number of meta MLP',type=int,default = 5)
	parser.add_argument('--max_disp',help='max_disp for shifring',type=int,default = 10)
	parser.add_argument('--resolution',help='resolution [h,w]',nargs='+',type=int,default = [270, 480])
	parser.add_argument('--use_norm',help='use my network for debugging',action='store_true')
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
	parser.add_argument('--no_blend',help='no blend network',action='store_true')
	parser.add_argument('--n_c',help='number of channel in netMet',type=int,default = 256)
	parser.add_argument('--batch_size',help='batch size ',type=int,default = 2)
	parser.add_argument('--datapath',help='the path of training dataset',type=str,default="../../Dataset/SynData_s1_all")
	parser.add_argument('--config',help='the config path',type=str,default="./config/config.json")
	args = parser.parse_args()



	save_args(args)

	if not os.path.exists(args.savepath):
		os.makedirs(args.savepath)

	regular_train(args)