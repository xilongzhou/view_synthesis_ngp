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
import imageio


from utils import VGGLoss, VGGpreprocess, saveimg

from network import VIINTER, linterp, Unet_Blend, MLP_blend, CondSIREN, CondSIREN_meta
import torch.nn.functional as F

import random

# from torchmeta.utils.gradient_based import gradient_update_parameters
from collections import OrderedDict

from my_dataloader import DataLoader_helper, DataLoader_helper2

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

	save_testpath = os.path.join(args.savepath, "test")
	save_modelpath = os.path.join(args.savepath, "ckpt")

	if not os.path.exists(save_testpath):
		os.makedirs(save_testpath)

	if not os.path.exists(save_modelpath):
		os.makedirs(save_modelpath)

	h_res = args.resolution[0]
	w_res = args.resolution[1]

	L_tag = torch.tensor([0]).cuda()
	R_tag = torch.tensor([1]).cuda()

	# # ---------------------- Dataloader ----------------------
	# Myset = DataLoader_helper(args.datapath, h_res, w_res)
	Myset = DataLoader_helper2(args.datapath, h_res, w_res, one_scene=args.load_one, load_model=True)
	Mydata = DataLoader(dataset=Myset, batch_size=args.batch_size, shuffle=True, drop_last=True)

	total_data = Myset.get_total_num()

	print(f"length of data is {total_data}")


	# ----------------------- define network --------------------------------------
	# common net
	if args.blend_type=="mlp":
		net_blend = MLP_blend(D=args.mlp_d, in_feat=3*args.num_planes, out_feat=3, W=args.mlp_w, with_res=False, with_norm=args.use_norm).cuda()
	elif args.blend_type=="unet":
		net_blend = Unet_Blend(3*args.num_planes, args.num_planes if args.mask_blend else 3, 4, (h_res, w_res)).cuda()

	
	net_blend.load_state_dict(torch.load(save_modelpath+"/model.ckpt")['nn_blend'])
	print("finish loading netblending")

	if args.use_viinter:
		net_scene = VIINTER(n_emb = 2, norm_p = 1, inter_fn=linterp, D=args.n_layer, z_dim = 128, in_feat=2, out_feat=3*args.num_planes, W=args.n_c, with_res=False, with_norm=True).cuda()
	else:
		net_scene = CondSIREN(n_emb = 2, norm_p = 1, inter_fn=linterp, D=args.n_layer, z_dim = 128, in_feat=2, out_feat=3*args.num_planes, W=args.n_c, with_res=False, with_norm=args.use_norm, use_sig=args.use_sigmoid).cuda()
	
	print("net_scene: ", net_scene)

	xp = [0, 1]
	fp = [-2, 2]

	shuffle = True

	inter_list = np.linspace(0,1,50)
	print("inter_list: ", inter_list)

	while True:

		# this is for outerloop
		for j,data in enumerate(Mydata):
			imgL_b, imgR_b, imgtest_b, inter_val_b, index_b, all_maskL_b, all_maskR_b, disp_b, model_b = data
			# imgL_b, imgR_b, imgtest_b, inter_val_b, index_b = data

			mseloss = torch.tensor(0.).cuda()
			vgloss = torch.tensor(0.).cuda()
			scene_loss = torch.tensor(0.).cuda()


			# --------------- fetch --------------------
			imgtest = imgtest_b[0:1].to(device)
			inter = inter_val_b[0:1]
			imgL = imgL_b[0:1].to(device)
			imgR = imgR_b[0:1].to(device)
			index = index_b[0]
			all_maskL = all_maskL_b[0].to(device)
			all_maskR = all_maskR_b[0].to(device)
			disp = disp_b[0].to(device)
			model_path = model_b[0]

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
			
			# load model
			saved_dict = torch.load(model_path)
			net_scene.load_state_dict(saved_dict['mlp'])
			# print("finsih loading network")

			# inter = torch.from_numpy(inter)
			video_out = imageio.get_writer(os.path.join(save_testpath, f"{j}_inter.mp4"), mode='I', fps=12, codec='libx264')

			video_dict = {}
			for l in range(args.num_planes):
				print("................", l)
				video_dict[f"layer_{l}"] = imageio.get_writer(os.path.join(save_testpath, f"{j}_layer{l}.mp4"), mode='I', fps=12, codec='libx264')


			for index, inter in enumerate(inter_list):
				z0 = net_scene.ret_z(torch.LongTensor([0.]).cuda()).squeeze()
				z1 = net_scene.ret_z(torch.LongTensor([1.]).cuda()).squeeze()

				zi = linterp(inter, z0, z1).unsqueeze(0)
				out = net_scene.forward_with_z(grid_inp, zi)
				out = out.reshape(1, h_res, -1, 3*args.num_planes).permute(0,3,1,2)

				off_scl = np.interp(inter, xp, fp)
				multi_out_list=[]
				for i in range(args.num_planes):
					meshxw = meshx + (base_shift * 2 + off_scl * offsets[i]) / (w_res + base_shift * 2)
					grid = torch.stack((meshxw, meshy), 2)[None].to(device).to(torch.float32)
					multi_out = grid_sample(out[:, 3*i:3*i+3], grid, mode='bilinear', align_corners=True)[:, :, :, :-base_shift*2]
					multi_out_list.append(multi_out)

					# if index%10==0:
						# saveimg(multi_out, f"{save_testpath}/{j}_{inter}_out{i}.png")
					tt = (multi_out.permute(0, 2, 3, 1) * 255).clamp(0, 255).to(torch.uint8).squeeze(0)
					tt = tt.detach().cpu().numpy()
					video_dict[f"layer_{i}"].append_data(tt)


				multi_out = torch.cat(multi_out_list,dim=1)
				blend_out = net_blend(2*multi_out-1) # multi out [0,1]-->[-1,1]


				if args.mask_blend:
					for k in range(args.num_planes):
						if k==0:
							blend_test = blend_out[:,0:1]*multi_out[:,0:3]
						else:
							blend_test += blend_out[:,k:k+1]*multi_out[:,3*k:3*k+3]
				else:
					blend_test = blend_out


				if index%10==0:
					saveimg(blend_test, f"{save_testpath}/{j}_{inter}_blendtest.png")

				blend_test = (blend_test.permute(0, 2, 3, 1) * 255).clamp(0, 255).to(torch.uint8).squeeze(0)
				blend_test = blend_test.detach().cpu().numpy()
				video_out.append_data(blend_test)

			video_out.close()

		break

	
if __name__=='__main__':
	

	parser = argparse.ArgumentParser()

	parser.add_argument('--nfg',type=int,help='capacity multiplier',default = 8)
	parser.add_argument('--num_planes',type=int,help='number of planes',default = 6)
	parser.add_argument('--num_freqs_pe', type=int,help='#frequencies for positional encoding',default = 5)
	parser.add_argument('--num_iters',type=int,help='num of iterations',default = 500000)
	parser.add_argument('--progress_iter',type=int,help='update frequency',default = 5000)
	parser.add_argument('--lr',type=float,help='learning rate',default = 0.0001)
	parser.add_argument('--savepath',type=str,help='saving path',default = 'resultsTest1')
	parser.add_argument('--blend_type',type=str,help='mlp || unet',default = 'mlp')
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
	parser.add_argument('--load_one',help='load one scene only',action='store_true')
	parser.add_argument('--reg_train',help='regular training, for debugging',action='store_true')
	parser.add_argument('--use_viinter',help='use viinter network or not',action='store_true')
	parser.add_argument('--mask_blend',help='use mask_blend',action='store_true')
	parser.add_argument('--no_blend',help='no blend network',action='store_true')
	parser.add_argument('--n_c',help='number of channel in netMet',type=int,default = 256)
	parser.add_argument('--batch_size',help='batch size ',type=int,default = 2)
	parser.add_argument('--datapath',help='the path of training dataset',type=str,default="../../Dataset/SynData_s1_all")
	args = parser.parse_args()



	save_args(args)

	if not os.path.exists(args.savepath):
		os.makedirs(args.savepath)

	regular_train(args)