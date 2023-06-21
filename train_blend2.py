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

from network import VIINTER, linterp, Unet_Blend, MLP_blend, CondSIREN
import torch.nn.functional as F

import random

# from torchmeta.utils.gradient_based import gradient_update_parameters
from collections import OrderedDict

from my_dataloader import DataLoader_helper, DataLoader_helper2, DataLoader_helper_blend, DataLoader_helper_blend_sep

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

    # # ---------------------- Dataloader ----------------------
    # Myset = DataLoader_helper(args.datapath, h_res, w_res)
    # Myset = DataLoader_helper2(args.datapath, h_res, w_res, one_scene=args.load_one, load_model=True)
    Myset = DataLoader_helper_blend_sep(args.datapath, h_res, w_res)
    Mydata = DataLoader(dataset=Myset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    total_data = Myset.get_total_num()

    print(f"length of data is {total_data}")


    # ----------------------- define network --------------------------------------
    net_blend = Unet_Blend(4*args.num_planes, args.num_planes if args.mask_blend else 3, (h_res, w_res), layer_n=args.blend_nl, n_c=32).cuda()
    
    with torch.no_grad():
        if args.batch_size==1:
            net_scene = CondSIREN(n_emb = 2, norm_p = 1, inter_fn=linterp, D=args.n_layer, z_dim = 128, in_feat=2, out_feat=3*args.num_planes, W=args.n_c, with_res=False, with_norm=args.use_norm, use_sig=args.use_sigmoid).cuda()
        else:
            # net_mask_list = []
            # net_rgb_list = []

            net_mask = CondSIREN(n_emb = 2, norm_p = 1, inter_fn=linterp, D=args.n_layer, z_dim = 128, in_feat=2, out_feat=1, W=args.n_c, with_res=False, with_norm=args.use_norm, use_sig=args.use_sigmoid).cuda()
            net_rgb = CondSIREN(n_emb = 2, norm_p = 1, inter_fn=linterp, D=args.n_layer, z_dim = 128, in_feat=2, out_feat=3, W=args.n_c, with_res=False, with_norm=args.use_norm, use_sig=args.use_sigmoid).cuda()

            # net_mask_list1.append(net_mask)
            # net_rgb_list1.append(net_rgb)


    optimizer_blend = torch.optim.Adam(net_blend.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_blend, T_max=args.num_iters, eta_min=args.lr*0.1)


    # print("net_scene: ", net_scene)

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
    epsilon=1e-2

    while iter < args.num_iters:

        for i,data in enumerate(Mydata):

            # _, _, imgtest_b, inter_val_b, index_b, _, _, disp_b, model_b = data
            imgtest_b, inter_val_b, index_b, max_disp_b, min_disp_b,model_b = data
            # imgL_b, imgR_b, imgtest_b, inter_val_b, index_b = data

            mseloss = torch.tensor(0.).cuda()
            vgloss = torch.tensor(0.).cuda()
            scene_loss = torch.tensor(0.).cuda()

            for task_id in range(args.batch_size):

                # --------------- fetch --------------------
                with torch.no_grad():
                    imgtest = imgtest_b[task_id:task_id+1].to(device)
                    inter = inter_val_b[task_id:task_id+1]

                    max_disp = max_disp_b[task_id].item()
                    min_disp = min_disp_b[task_id].item()
                    model_path = model_b[task_id]

                    planes = torch.round(torch.linspace(min_disp-epsilon, max_disp+epsilon, args.num_planes+1)/2)*2
                    max_shift = int(max_disp//2)
                    offsets = [ int((planes[i]/2+planes[i+1]/2)//2) for i in range(args.num_planes)]

                    # load rgb model
                    # print("model_path: ", model_path)
                    # print("max_shift: ", max_shift)

                    multi_out_list = []

                    for layer_n in range(6):

                        base_shift = offsets[layer_n]

                        # print(f"offsets: {offsets}, baseshift: {base_shift}")

                        coords_h = np.linspace(-1, 1, h_res, endpoint=False)
                        coords_w = np.linspace(-1, 1,  w_res + base_shift * 2, endpoint=False)
                        xy_grid = np.stack(np.meshgrid(coords_w, coords_h), -1)
                        xy_grid = torch.FloatTensor(xy_grid).cuda()
                        grid_inp = xy_grid.view(-1, 2).contiguous().unsqueeze(0)
                        dx = torch.from_numpy(coords_w).float()
                        dy = torch.from_numpy(coords_h).float()
                        meshy, meshx = torch.meshgrid((dy, dx))
                        

                        # load rgb

                        ckpt_path = os.path.join(model_path, f"rgb{layer_n}.ckpt")
                        saved_dict = torch.load(ckpt_path, map_location='cuda:0')
                        net_rgb.load_state_dict(saved_dict['mlp'])

                        z0 = net_rgb.ret_z(torch.LongTensor([0.]).cuda()).squeeze()
                        z1 = net_rgb.ret_z(torch.LongTensor([1.]).cuda()).squeeze()
                        zi = linterp(inter.to(device), z0, z1).unsqueeze(0)
                        out = net_rgb.forward_with_z(grid_inp, zi)
                        out_rgb = out.reshape(1, h_res, -1, 3).permute(0,3,1,2)
                        # print("out_rgb: ", out_rgb.shape)

                        del zi,z0,z1


                        # load mask
                        ckpt_path = os.path.join(model_path, f"mask{layer_n}.ckpt")
                        saved_dict = torch.load(ckpt_path, map_location='cuda:0')
                        net_mask.load_state_dict(saved_dict['mlp'])

                        z0 = net_mask.ret_z(torch.LongTensor([0.]).cuda()).squeeze()
                        z1 = net_mask.ret_z(torch.LongTensor([1.]).cuda()).squeeze()
                        zi = linterp(inter.to(device), z0, z1).unsqueeze(0)
                        out = net_mask.forward_with_z(grid_inp, zi)
                        out_mask = out.reshape(1, h_res, -1, 1).permute(0,3,1,2)
                        # print("out_mask: ", out_mask.shape)

                        del zi,z0,z1


                        # shifting
                        off_scl = np.interp(inter, xp, fp)

                        # rgb
                        meshxw = meshx + (base_shift * 2 + off_scl * base_shift) / (w_res + base_shift * 2)
                        grid = torch.stack((meshxw, meshy), 2)[None].to(device).to(torch.float32)
                        multi_out_rgb = grid_sample(out_rgb[:, 0:3], grid, mode='bilinear', align_corners=True)[:, :, :, :-base_shift*2]
                        multi_out_mask = grid_sample(out_mask[:, 0:1], grid, mode='bilinear', align_corners=True)[:, :, :, :-base_shift*2]
                        # print(layer_n, "multi_out_mask: ", multi_out_mask.shape)
                        multi_out_tmp = torch.cat([multi_out_rgb, multi_out_mask], dim=1)

                        multi_out_list.append(multi_out_tmp)

                multi_out = torch.cat(multi_out_list,dim=1)

                # print("multi_out: ", multi_out.shape)



                ## ======================= start train blending network ==========================================

                blend_out = net_blend(2*multi_out-1) # multi out [0,1]-->[-1,1]

                if args.mask_blend:
                    for k in range(args.num_planes):
                        if args.add_mask:
                            if k==0:
                                blend_test = blend_out[:,0:1]*multi_out[:,0:3]
                            else:
                                blend_test += blend_out[:,k:k+1]*multi_out[:,4*k:4*k+3]                         
                        else:
                            if k==0:
                                blend_test = blend_out[:,0:1]*multi_out[:,0:3]
                            else:
                                blend_test += blend_out[:,k:k+1]*multi_out[:,3*k:3*k+3]
                else:
                    blend_test = blend_out

                mseloss = mseloss + metric_mse(blend_test, imgtest)
                if args.w_vgg>0:
                    blend_test_vg = VGGpreprocess(blend_test)
                    imgtest_vg = VGGpreprocess(imgtest)
                    vgloss = vgloss + metric_vgg(blend_test_vg, imgtest_vg) * args.w_vgg
                
            mseloss.div_(args.batch_size)
            vgloss.div_(args.batch_size)

            blend_loss = mseloss + vgloss

            optimizer_blend.zero_grad()
            blend_loss.backward()
            optimizer_blend.step()

            torch.cuda.empty_cache()

            # print("iter: ", iter, model_path, base_shift)

            if (iter % args.progress_iter) ==0:

                print(f"iterations: {iter}, interval: {inter}, blend_loss: {blend_loss}, mseloss: {mseloss}, vgloss: {vgloss}")
                print(f"offset: {offsets}, base_shift: {base_shift}, planes: {planes},")

                save_dict = { 
                            'nn_blend':net_blend.state_dict() if not args.no_blend else None, \
                            }
                torch.save(save_dict, "%s/blend.ckpt"%(save_modelpath))

                # visualize layer
                for i in range(args.num_planes):
                    saveimg(multi_out[:, 4*i:4*i+3], f"{save_imgpath}/{iter}_outrgb_{i}.png")
                    saveimg(multi_out[:, 4*i+3:4*i+4].repeat(1,3,1,1), f"{save_imgpath}/{iter}_outmask_{i}.png")

                    if args.mask_blend:
                        saveimg(blend_out[:, i:i+1].repeat(1,3,1,1), f"{save_imgpath}/{iter}_blendmask_{i}.png")


                saveimg(blend_test, f"{save_imgpath}/{iter}_blendtest.png")
                saveimg(imgtest, f"{save_imgpath}/{iter}_gttest.png")

            iter +=1



    
if __name__=='__main__':
    

    parser = argparse.ArgumentParser()

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
    parser.add_argument('--n_layer',help='layer number of meta MLP',type=int,default = 5)
    parser.add_argument('--blend_nl',help='layer number of blending net',type=int,default = 4)
    parser.add_argument('--max_disp',help='max_disp for shifring',type=int,default = 10)
    parser.add_argument('--resolution',help='resolution [h,w]',nargs='+',type=int,default = [270, 480])
    parser.add_argument('--add_mask',help='add mask for training',action='store_true')
    parser.add_argument('--use_norm',help='use my network for debugging',action='store_true')
    parser.add_argument('--use_sigmoid',help='add sigmoid to the end of CondSiren',action='store_true')
    parser.add_argument('--load_one',help='load one scene only',action='store_true')
    parser.add_argument('--reg_train',help='regular training, for debugging',action='store_true')
    parser.add_argument('--use_viinter',help='use viinter network or not',action='store_true')
    parser.add_argument('--no_blend',help='no blend network',action='store_true')
    parser.add_argument('--mask_blend',help='output alpha mask for blending',action='store_true')
    parser.add_argument('--n_c',help='number of channel in netMet',type=int,default = 256)
    parser.add_argument('--batch_size',help='batch size ',type=int,default = 2)
    parser.add_argument('--datapath',help='the path of training dataset',type=str,default="../../Dataset/SynData_s1_all")
    args = parser.parse_args()



    save_args(args)

    if not os.path.exists(args.savepath):
        os.makedirs(args.savepath)

    regular_train(args)