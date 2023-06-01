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

import imageio

from models.xfields import XfieldsFlow
from util.args import TrainingArguments


from PIL import Image


from utils import VGGLoss, VGGpreprocess, saveimg

from network import VIINTER, linterp, Unet_Blend, MLP_blend, CondSIREN
import torch.nn.functional as F

import random

# from torchmeta.utils.gradient_based import gradient_update_parameters
from collections import OrderedDict

from my_dataloader import DataLoader_helper, DataLoader_helper2, DataLoader_helper_test

from torch.nn.functional import interpolate, grid_sample

import argparse

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

	idx = args.savepath.split("/")[-1]
	print("idx: ", idx)

	# save_imgpath = os.path.join(args.savepath, "imgs")
	save_testpath = os.path.join(args.savepath, "test")
	save_modelpath = os.path.join(args.savepath, "ckpt")

	# if not os.path.exists(save_imgpath):
	# 	os.makedirs(save_imgpath)

	if not os.path.exists(save_testpath):
		os.makedirs(save_testpath)

	print("save_testpath: ", save_testpath)

	# if not os.path.exists(save_modelpath):
	# 	os.makedirs(save_modelpath)

	h_res = args.resolution[0]
	w_res = args.resolution[1]
	L_tag = torch.tensor([0]).cuda()
	R_tag = torch.tensor([1]).cuda()

	# # ---------------------- Dataloader ----------------------
	# Myset = DataLoader_helper(args.datapath, h_res, w_res)
	Myset = DataLoader_helper_test(os.path.join(args.datapath, idx), h_res, w_res)
	Mydata = DataLoader(dataset=Myset, batch_size=args.batch_size, shuffle=True, drop_last=True)


	# ----------------------- define network --------------------------------------

	if args.use_viinter:
		net_scene = VIINTER(n_emb = 2, norm_p = 1, inter_fn=linterp, D=args.n_layer, z_dim = 128, in_feat=2, out_feat=3*args.num_planes, W=args.n_c, with_res=False, with_norm=True).cuda()
	else:
		net_scene = CondSIREN(n_emb = 2, norm_p = 1, inter_fn=linterp, D=args.n_layer, z_dim = 128, in_feat=2, out_feat=3*args.num_planes if not args.add_mask else 4*args.num_planes, W=args.n_c, with_res=False, with_norm=args.use_norm, use_sig=args.use_sigmoid).cuda()
	

	# load network
	saved_dict = torch.load("%s/model.ckpt"%(save_modelpath), map_location='cuda:0')
	net_scene.load_state_dict(saved_dict['mlp'])

	print("finish loading net_scene: ", net_scene)

	xp = [0, 1]
	fp = [-2, 2]

	dist = 1/20
	inter_list = [np.float32(i*dist) for i in range(21)]

	if args.add_mask:
		video_dict = {}
		for l in range(args.num_planes):
			print("................", l)
			video_dict[f"layer_{l}"] = imageio.get_writer(os.path.join(save_testpath, f"layer_{l}.mp4"), mode='I', fps=6, codec='libx264')
			video_dict[f"mask_{l}"] = imageio.get_writer(os.path.join(save_testpath, f"mask_{l}.mp4"), mode='I', fps=6, codec='libx264')

			video_dict[f"layer_shift_{l}"] = imageio.get_writer(os.path.join(save_testpath, f"layer_shift_{l}.mp4"), mode='I', fps=6, codec='libx264')
			video_dict[f"mask_shift_{l}"] = imageio.get_writer(os.path.join(save_testpath, f"mask_shift_{l}.mp4"), mode='I', fps=6, codec='libx264')

	else:
		video_dict = {}
		for l in range(args.num_planes):
			print("................", l)
			video_dict[f"layer_{l}"] = imageio.get_writer(os.path.join(save_testpath, f"layer_{l}.mp4"), mode='I', fps=6, codec='libx264')
			# video_dict[f"mask_{l}"] = imageio.get_writer(os.path.join(save_testpath, f"mask{l}.mp4"), mode='I', fps=6, codec='libx264')

	# this is for outerloop
	for i,data in enumerate(Mydata):
		disp_b = data

		# --------------- fetch --------------------
		disp = disp_b[0].to(device)

		# compute offsets, baseline (scene-dependent)
		disp = torch.abs(disp[:,:,0])
		max_disp = torch.max(disp)
		min_disp = torch.min(disp)
		planes = torch.round(torch.linspace(min_disp, max_disp, args.num_planes+1)/2)*2
		base_shift = int(max_disp//2)
		offsets = [ int((planes[i]/2+planes[i+1]/2)//2) for i in range(args.num_planes)]
		print("max_disp: ", max_disp, "min_disp: ", min_disp)

		print(f"baseshift: {base_shift}")

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


		# --------------- end fetch ------------------


		for inter in inter_list:
			print("inter: ", inter)
			
			# inter = torch.from_numpy(inter)
			z0 = net_scene.ret_z(torch.LongTensor([0.]).cuda()).squeeze()
			z1 = net_scene.ret_z(torch.LongTensor([1.]).cuda()).squeeze()

			zi = linterp(inter, z0, z1).unsqueeze(0)
			out = net_scene.forward_with_z(grid_inp, zi)
			out = out.reshape(1, h_res, -1, 3*args.num_planes if not args.add_mask else 4*args.num_planes).permute(0,3,1,2)


			off_scl = np.interp(inter, xp, fp)

			# visualize layer
			if args.add_mask:
				for i in range(args.num_planes):

					tt = (out[:, 4*i:4*i+3].permute(0, 2, 3, 1) * 255).clamp(0, 255).to(torch.uint8).squeeze(0)
					tt = tt.detach().cpu().numpy()
					video_dict[f"layer_{i}"].append_data(tt)

					tt = (out[:, 4*i+3:4*i+4].repeat(1,3,1,1).permute(0, 2, 3, 1) * 255).clamp(0, 255).to(torch.uint8).squeeze(0)
					tt = tt.detach().cpu().numpy()
					video_dict[f"mask_{i}"].append_data(tt)

					# shift version
					meshxw = meshx + (base_shift * 2 + off_scl * offsets[i]) / (w_res + base_shift * 2)
					grid = torch.stack((meshxw, meshy), 2)[None].to(device).to(torch.float32)
					out_shift = grid_sample(out, grid, mode='bilinear', align_corners=True)[:, :, :, :-base_shift*2]

					tt_shift = (out_shift[:, 4*i:4*i+3].permute(0, 2, 3, 1) * 255).clamp(0, 255).to(torch.uint8).squeeze(0)
					tt_shift = tt_shift.detach().cpu().numpy()
					video_dict[f"layer_shift_{i}"].append_data(tt_shift)

					tt_shift = (out_shift[:, 4*i+3:4*i+4].repeat(1,3,1,1).permute(0, 2, 3, 1) * 255).clamp(0, 255).to(torch.uint8).squeeze(0)
					tt_shift = tt_shift.detach().cpu().numpy()
					video_dict[f"mask_shift_{i}"].append_data(tt_shift)



			else:
				for i in range(args.num_planes):

					tt = (out[:, 3*i:3*i+3].permute(0, 2, 3, 1) * 255).clamp(0, 255).to(torch.uint8).squeeze(0)
					tt = tt.detach().cpu().numpy()
					video_dict[f"layer_{i}"].append_data(tt)


				# for i in range(args.num_planes):
				# 	saveimg(out[:, 3*i:3*i+3], f"{save_testpath}/{inter}_out_{i}.png")

		break


def hyper_test(args):

	from hypernetwork import CondSIREN_meta, HyperNetwork

	device = "cuda"

	idx = args.savepath.split("/")[-1]
	print("idx: ", idx)

	# save_imgpath = os.path.join(args.savepath, "imgs")
	save_testpath = os.path.join(args.savepath, "test")
	save_modelpath = os.path.join(args.savepath, "ckpt")

	# if not os.path.exists(save_imgpath):
	# 	os.makedirs(save_imgpath)

	if not os.path.exists(save_testpath):
		os.makedirs(save_testpath)

	print("save_testpath: ", save_testpath)

	# if not os.path.exists(save_modelpath):
	# 	os.makedirs(save_modelpath)

	h_res = args.resolution[0]
	w_res = args.resolution[1]
	L_tag = torch.tensor([0]).cuda()
	R_tag = torch.tensor([1]).cuda()

	# # ---------------------- Dataloader ----------------------
	# Myset = DataLoader_helper(args.datapath, h_res, w_res)
	Myset = DataLoader_helper_test(os.path.join(args.datapath, idx), h_res, w_res)
	Mydata = DataLoader(dataset=Myset, batch_size=args.batch_size, shuffle=True, drop_last=True)

	scalar = 0.1
	# ----------------------- define network --------------------------------------
	if args.debug:
		from metasiren import MetaSirenGrid
		import warnings
		warnings.filterwarnings("ignore", category=UserWarning) 

	else:
		if args.mask_only:
			net_scene = CondSIREN_meta(D=args.n_layer, in_feat=2, out_feat=args.num_planes , W=args.n_c, with_res=False).cuda()
		else:
			net_scene = CondSIREN_meta(D=args.n_layer, in_feat=2, out_feat=3*args.num_planes if not args.add_mask else 4*args.num_planes, W=args.n_c, with_res=False).cuda()
		hypernet = HyperNetwork(hyper_in_features=1, hyper_hidden_layers=3, hyper_hidden_features=128, hypo_module=net_scene).cuda()

		# load network
		saved_dict = torch.load("%s/model.ckpt"%(save_modelpath), map_location='cuda:0')
		hypernet.load_state_dict(saved_dict['hypernet'])

	# print("finish loading net_scene: ", net_scene)

	xp = [0, 1]
	fp = [-2, 2]

	dist = 1/11
	inter_list = [np.float32(i*dist) for i in range(12)]

	if args.mask_only:
		video_dict = {}
		for l in range(args.num_planes):
			print("................", l)
			video_dict[f"mask_{l}"] = imageio.get_writer(os.path.join(save_testpath, f"mask{l}.mp4"), mode='I', fps=6, codec='libx264')		

	elif args.add_mask:
		video_dict = {}
		for l in range(args.num_planes):
			print("................", l)
			video_dict[f"layer_{l}"] = imageio.get_writer(os.path.join(save_testpath, f"layer{l}.mp4"), mode='I', fps=6, codec='libx264')
			video_dict[f"mask_{l}"] = imageio.get_writer(os.path.join(save_testpath, f"mask{l}.mp4"), mode='I', fps=6, codec='libx264')

	else:
		video_dict = {}
		for l in range(args.num_planes):
			print("................", l)
			video_dict[f"layer_{l}"] = imageio.get_writer(os.path.join(save_testpath, f"layer{l}.mp4"), mode='I', fps=6, codec='libx264')
			# video_dict[f"mask_{l}"] = imageio.get_writer(os.path.join(save_testpath, f"mask{l}.mp4"), mode='I', fps=6, codec='libx264')

	# this is for outerloop
	for i,data in enumerate(Mydata):
		disp_b = data

		# --------------- fetch --------------------
		disp = disp_b[0].to(device)

		# compute offsets, baseline (scene-dependent)
		disp = torch.abs(disp[:,:,0])
		max_disp = torch.max(disp)
		min_disp = torch.min(disp)
		planes = torch.round(torch.linspace(min_disp, max_disp, args.num_planes+1)/2)*2
		base_shift = int(max_disp//2)
		offsets = [ int((planes[i]/2+planes[i+1]/2)//2) for i in range(args.num_planes)]
		print("max_disp: ", max_disp, "min_disp: ", min_disp)

		print(f"baseshift: {base_shift}")

		if args.debug:
			net_scene = MetaSirenGrid(args, w_res=w_res + base_shift * 2, h_res=h_res, hidden_features=128, hidden_layers=5, out_features=3*args.num_planes if not args.add_mask else 4*args.num_planes, outermost_linear=True, first_omega_0=10, hidden_omega_0=10).to(device).train()
			hypernet = HyperNetwork(hyper_in_features=1, hyper_hidden_layers=3, hyper_hidden_features=128, hypo_module=net_scene).cuda()
			# load network
			saved_dict = torch.load("%s/model.ckpt"%(save_modelpath), map_location='cuda:0')
			hypernet.load_state_dict(saved_dict['hypernet'])

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


		# --------------- end fetch ------------------

		for inter in inter_list:
			print("inter: ", inter)
			
			inter_tag = torch.tensor([inter]).cuda()*scalar
			
			params_model = hypernet(inter_tag)
			if not args.debug:
				out = net_scene(grid_inp, params=params_model)
			else:
				out = net_scene(params=params_model) 

			if args.mask_only:
				out = out.reshape(1, h_res, -1, args.num_planes).permute(0,3,1,2)
			else:
				out = out.reshape(1, h_res, -1, 3*args.num_planes if not args.add_mask else 4*args.num_planes).permute(0,3,1,2)

			# visualize layer
			if args.mask_only:
				for i in range(args.num_planes):
					tt = (out[:, i:i+1].repeat(1,3,1,1).permute(0, 2, 3, 1) * 255).clamp(0, 255).to(torch.uint8).squeeze(0)
					tt = tt.detach().cpu().numpy()
					video_dict[f"mask_{i}"].append_data(tt)


			elif args.add_mask:
				for i in range(args.num_planes):

					tt = (out[:, 4*i:4*i+3].permute(0, 2, 3, 1) * 255).clamp(0, 255).to(torch.uint8).squeeze(0)
					tt = tt.detach().cpu().numpy()
					video_dict[f"layer_{i}"].append_data(tt)


					tt = (out[:, 4*i+3:4*i+4].repeat(1,3,1,1).permute(0, 2, 3, 1) * 255).clamp(0, 255).to(torch.uint8).squeeze(0)
					tt = tt.detach().cpu().numpy()
					video_dict[f"mask_{i}"].append_data(tt)

			else:
				for i in range(args.num_planes):

					tt = (out[:, 3*i:3*i+3].permute(0, 2, 3, 1) * 255).clamp(0, 255).to(torch.uint8).squeeze(0)
					tt = tt.detach().cpu().numpy()
					video_dict[f"layer_{i}"].append_data(tt)


				# for i in range(args.num_planes):
				# 	saveimg(out[:, 3*i:3*i+3], f"{save_testpath}/{inter}_out_{i}.png")

		break


def separate_train(args):

	layer_n = args.separate_layer

	device = "cuda"

	idx = args.savepath.split("/")[-1]
	print("idx: ", idx)

	# save_imgpath = os.path.join(args.savepath, "imgs")
	save_testpath = os.path.join(args.savepath, f"test")
	save_modelpath = os.path.join(args.savepath, f"ckpt{layer_n}")

	# if not os.path.exists(save_imgpath):
	# 	os.makedirs(save_imgpath)

	if not os.path.exists(save_testpath):
		os.makedirs(save_testpath)

	print("save_testpath: ", save_testpath)

	# if not os.path.exists(save_modelpath):
	# 	os.makedirs(save_modelpath)

	h_res = args.resolution[0]
	w_res = args.resolution[1]
	L_tag = torch.tensor([0]).cuda()
	R_tag = torch.tensor([1]).cuda()

	# # ---------------------- Dataloader ----------------------
	# Myset = DataLoader_helper(args.datapath, h_res, w_res)
	Myset = DataLoader_helper_test(os.path.join(args.datapath, idx), h_res, w_res)
	Mydata = DataLoader(dataset=Myset, batch_size=args.batch_size, shuffle=True, drop_last=True)


	# ----------------------- define network --------------------------------------


	if args.use_viinter:
		net_scene = VIINTER(n_emb = 2, norm_p = 1, inter_fn=linterp, D=args.n_layer, z_dim = 128, in_feat=2, out_feat=3, W=args.n_c, with_res=False, with_norm=True).cuda()
	else:
		net_scene = CondSIREN(n_emb = 2, norm_p = 1, inter_fn=linterp, D=args.n_layer, z_dim = 128, in_feat=2, out_feat=4, W=args.n_c, with_res=False, with_norm=args.use_norm, use_sig=args.use_sigmoid).cuda()
	

	# load network
	saved_dict = torch.load("%s/model.ckpt"%(save_modelpath), map_location='cuda:0')
	net_scene.load_state_dict(saved_dict['mlp'])

	print("finish loading net_scene: ", net_scene)

	xp = [0, 1]
	fp = [-2, 2]

	dist = 1/20
	inter_list = [np.float32(i*dist) for i in range(21)]

	if args.add_mask:
		video_dict = {}
		video_dict[f"layer"] = imageio.get_writer(os.path.join(save_testpath, f"layer_{layer_n}.mp4"), mode='I', fps=6, codec='libx264')
		video_dict[f"mask"] = imageio.get_writer(os.path.join(save_testpath, f"mask_{layer_n}.mp4"), mode='I', fps=6, codec='libx264')

		video_dict[f"layer_shift"] = imageio.get_writer(os.path.join(save_testpath, f"layer_shift_{layer_n}.mp4"), mode='I', fps=6, codec='libx264')
		video_dict[f"mask_shift"] = imageio.get_writer(os.path.join(save_testpath, f"mask_shift_{layer_n}.mp4"), mode='I', fps=6, codec='libx264')

	# this is for outerloop
	for i,data in enumerate(Mydata):
		disp_b = data

		# --------------- fetch --------------------
		disp = disp_b[0].to(device)

		# compute offsets, baseline (scene-dependent)
		disp = torch.abs(disp[:,:,0])
		max_disp = torch.max(disp)
		min_disp = torch.min(disp)
		planes = torch.round(torch.linspace(min_disp, max_disp, args.num_planes+1)/2)*2
		base_shift = int(max_disp//2)
		offsets = [ int((planes[i]/2+planes[i+1]/2)//2) for i in range(args.num_planes)]
		print("max_disp: ", max_disp, "min_disp: ", min_disp)

		print(f"baseshift: {base_shift}")

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


		# --------------- end fetch ------------------


		for inter in inter_list:
			print("inter: ", inter)
			
			# inter = torch.from_numpy(inter)
			z0 = net_scene.ret_z(torch.LongTensor([0.]).cuda()).squeeze()
			z1 = net_scene.ret_z(torch.LongTensor([1.]).cuda()).squeeze()

			zi = linterp(inter, z0, z1).unsqueeze(0)
			out = net_scene.forward_with_z(grid_inp, zi)
			out = out.reshape(1, h_res, -1, 4).permute(0,3,1,2)


			off_scl = np.interp(inter, xp, fp)


			tt = (out[:, 0:3].permute(0, 2, 3, 1) * 255).clamp(0, 255).to(torch.uint8).squeeze(0)
			tt = tt.detach().cpu().numpy()
			video_dict["layer"].append_data(tt)

			tt = (out[:, 3:4].repeat(1,3,1,1).permute(0, 2, 3, 1) * 255).clamp(0, 255).to(torch.uint8).squeeze(0)
			tt = tt.detach().cpu().numpy()
			video_dict["mask"].append_data(tt)

			# shift version
			meshxw = meshx + (base_shift * 2 + off_scl * offsets[layer_n]) / (w_res + base_shift * 2)
			grid = torch.stack((meshxw, meshy), 2)[None].to(device).to(torch.float32)
			out_shift = grid_sample(out, grid, mode='bilinear', align_corners=True)[:, :, :, :-base_shift*2]

			tt_shift = (out_shift[:, :3].permute(0, 2, 3, 1) * 255).clamp(0, 255).to(torch.uint8).squeeze(0)
			tt_shift = tt_shift.detach().cpu().numpy()
			video_dict["layer_shift"].append_data(tt_shift)

			tt_shift = (out_shift[:, 3:4].repeat(1,3,1,1).permute(0, 2, 3, 1) * 255).clamp(0, 255).to(torch.uint8).squeeze(0)
			tt_shift = tt_shift.detach().cpu().numpy()
			video_dict["mask_shift"].append_data(tt_shift)

		break



def blend(args):

	device = "cuda"

	idx = args.savepath.split("/")[-1]
	print("idx: ", idx)

	# save_imgpath = os.path.join(args.savepath, "imgs")
	save_testpath = os.path.join(args.savepath, f"blend")

	if not os.path.exists(save_testpath):
		os.makedirs(save_testpath)

	print("save_testpath: ", save_testpath)

	# if not os.path.exists(save_modelpath):
	# 	os.makedirs(save_modelpath)

	h_res = args.resolution[0]
	w_res = args.resolution[1]
	L_tag = torch.tensor([0]).cuda()
	R_tag = torch.tensor([1]).cuda()

	# # ---------------------- Dataloader ----------------------
	# Myset = DataLoader_helper(args.datapath, h_res, w_res)
	Myset = DataLoader_helper_test(os.path.join(args.datapath, idx), h_res, w_res)
	Mydata = DataLoader(dataset=Myset, batch_size=1, shuffle=True, drop_last=True)

	with torch.no_grad():
		# ----------------------- define network --------------------------------------
		net_list = []
		for k in range(6):

			net_scene = CondSIREN(n_emb = 2, norm_p = 1, inter_fn=linterp, D=args.n_layer, z_dim = 128, in_feat=2, out_feat=4, W=args.n_c, with_res=False, with_norm=args.use_norm, use_sig=args.use_sigmoid).cuda()
			net_list.append(net_scene)

			save_modelpath = os.path.join(args.savepath, f"ckpt{k}")

			# load network
			saved_dict = torch.load("%s/model.ckpt"%(save_modelpath), map_location='cuda:0')
			net_list[k].load_state_dict(saved_dict['mlp'])

			print("finish loading net_scene: ", k)

		# ----------------------- blending --------------------------------------

		xp = [0, 1]
		fp = [-2, 2]

		dist = 1/20
		inter_list = [np.float32(i*dist) for i in range(21)]
		# inter_list = [0]

		video = imageio.get_writer(os.path.join(save_testpath, f"video.mp4"), mode='I', fps=6, codec='libx264')
		# video_dict = {}
		# video_dict[f"layer_shift"] = imageio.get_writer(os.path.join(save_testpath, f"layer_shift.mp4"), mode='I', fps=6, codec='libx264')
		# video_dict[f"layer_shift"] = imageio.get_writer(os.path.join(save_testpath, f"layer_shift.mp4"), mode='I', fps=6, codec='libx264')
		# video_dict[f"mask_shift"] = imageio.get_writer(os.path.join(save_testpath, f"mask_shift.mp4"), mode='I', fps=6, codec='libx264')

		# this is for outerloop
		for i,data in enumerate(Mydata):
			disp_b = data

			# --------------- fetch --------------------
			disp = disp_b[0].to(device)

			# compute offsets, baseline (scene-dependent)
			disp = torch.abs(disp[:,:,0])
			max_disp = torch.max(disp)
			min_disp = torch.min(disp)
			planes = torch.round(torch.linspace(min_disp, max_disp, args.num_planes+1)/2)*2
			base_shift = int(max_disp//2)
			offsets = [ int((planes[i]/2+planes[i+1]/2)//2) for i in range(args.num_planes)]
			print("max_disp: ", max_disp, "min_disp: ", min_disp)

			print(f"baseshift: {base_shift}")

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


			# --------------- end fetch ------------------

			for inter in inter_list:
				print("inter: ", inter)

				rgb_list = []
				mask_list = []

				for n_l in range(6):
					net_scene = net_list[n_l]
					# inter = torch.from_numpy(inter)
					z0 = net_scene.ret_z(torch.LongTensor([0.]).cuda()).squeeze()
					z1 = net_scene.ret_z(torch.LongTensor([1.]).cuda()).squeeze()

					zi = linterp(inter, z0, z1).unsqueeze(0)
					out = net_scene.forward_with_z(grid_inp, zi)
					out = out.reshape(1, h_res, -1, 4).permute(0,3,1,2)
					off_scl = np.interp(inter, xp, fp)

					meshxw = meshx + (base_shift * 2 + off_scl * offsets[n_l]) / (w_res + base_shift * 2)
					grid = torch.stack((meshxw, meshy), 2)[None].to(device).to(torch.float32)
					out_shift = grid_sample(out, grid, mode='bilinear', align_corners=True)[:, :, :, :-base_shift*2]

					rgb_list.append(out_shift[:, :3].permute(0, 2, 3, 1).squeeze(0).clamp(0, 1))
					mask = out_shift[:, 3:4].permute(0, 2, 3, 1).squeeze(0).repeat(1,1,3).clamp(0, 1)

					mask[mask<=0.1] = 0
					mask[mask>=0.9] = 1

					mask_list.append(mask)

				for n_l in reversed(range(6)):

					print(n_l)
					rgb = rgb_list[n_l]
					mask = mask_list[n_l]

					if n_l!=5:
						mask_final = mask*(1-remainder_mask)
						remainder_mask = torch.logical_or(mask_final, remainder_mask).float()
					else:
						remainder_mask = mask
						mask_final = mask

					if n_l==5:
						final = mask_final*rgb
					else:
						final += mask_final*rgb


					# remainder_mask_vis = (remainder_mask*255).clamp(0, 255).to(torch.uint8)
					# save_path = os.path.join(save_testpath,f"{n_l}_remainder_mask.png")
					# Image.fromarray(remainder_mask_vis.cpu().numpy()).save(save_path)



					mask_vis = (mask*255).clamp(0, 255).to(torch.uint8)
					save_path = os.path.join(save_testpath,f"{n_l}_mask1.png")
					Image.fromarray(mask_vis.cpu().numpy()).save(save_path)

					mask_vis = (mask_final*255).clamp(0, 255).to(torch.uint8)
					save_path = os.path.join(save_testpath,f"{n_l}_mask.png")
					Image.fromarray(mask_vis.cpu().numpy()).save(save_path)

					rgb_vis = (rgb*255).clamp(0, 255).to(torch.uint8)
					save_path = os.path.join(save_testpath,f"{n_l}_rgb.png")
					Image.fromarray(rgb_vis.cpu().numpy()).save(save_path)
			
					final_vis = (final*255).clamp(0, 255).to(torch.uint8)
					save_path = os.path.join(save_testpath,f"{n_l}_final.png")
					Image.fromarray(final_vis.cpu().numpy()).save(save_path)

				# tt_shift = (out_shift[:, 3:4].repeat(1,3,1,1).permute(0, 2, 3, 1) * 255).clamp(0, 255).to(torch.uint8).squeeze(0)
				final_vis = final_vis.detach().cpu().numpy()
				video.append_data(final_vis)

			# break

			# 	rgb = ( * 255).clamp(0, 255).to(torch.uint8).squeeze(0)



	
if __name__=='__main__':
	

	parser = argparse.ArgumentParser()

	parser.add_argument('--nfg',type=int,help='capacity multiplier',default = 8)
	parser.add_argument('--num_planes',type=int,help='number of planes',default = 6)
	parser.add_argument('--num_freqs_pe', type=int,help='#frequencies for positional encoding',default = 5)
	parser.add_argument('--num_iters',type=int,help='num of iterations',default = 500000)
	parser.add_argument('--progress_iter',type=int,help='update frequency',default = 5000)
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
	parser.add_argument('--add_mask',help='regular training, for debugging',action='store_true')
	parser.add_argument('--use_viinter',help='use viinter network or not',action='store_true')
	parser.add_argument('--hypernet',help='use viinter network or not',action='store_true')
	parser.add_argument('--mask_only',help='use viinter network or not',action='store_true')
	parser.add_argument('--debug',help='use viinter network or not',action='store_true')
	parser.add_argument('--n_c',help='number of channel in netMet',type=int,default = 256)
	parser.add_argument('--separate_layer',help='seperate layers',type=int,default = -1)
	parser.add_argument('--batch_size',help='batch size ',type=int,default = 2)
	parser.add_argument('--datapath',help='the path of training dataset',type=str,default="../../Dataset/SynData_s1_all")
	parser.add_argument('--blend',help='blending test',action='store_true')

	args = parser.parse_args()



	save_args(args)

	if not os.path.exists(args.savepath):
		os.makedirs(args.savepath)


	if args.blend:
		blend(args)
	else:

		if args.hypernet:
			hyper_test(args)
		else:
			if args.separate_layer>=0:
				separate_train(args)
			else:
				regular_train(args)