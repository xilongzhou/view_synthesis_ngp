from sched import scheduler
import torch
import torch.nn as nn
from torch.nn.functional import interpolate, grid_sample
from torch.optim import lr_scheduler, Adam
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.transforms.functional import gaussian_blur

import numpy as np
import cv2,os,time
from tqdm import tqdm

# import flow_vis
from xfields_blending import Blending_train_stereo_video_only as Blend
from dataloaderSVPE import Video
from models.xfields import XfieldsFlow
from util.args import TrainingArguments


from PIL import Image

from utils import VGGLoss, VGGpreprocess

import imageio
from network import VIINTER, linterp, Unet_Blend, MLP_blend

import torch.nn.functional as F

torch.set_num_threads(1)


def init(args):

    torch.manual_seed(1234)
    np.random.seed(1234)
    savedir = os.path.join(args.savepath,os.path.split(args.dataset)[1])
    os.makedirs(os.path.join(savedir,"test"), exist_ok=True)
    
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


def test(args):
    def save_flow(flows_out, txt, max_mot, shift_mask=None):
        disp = torch.clamp(-flows_out, torch.Tensor([0]).cuda(), max_mot).permute(1, 2, 0).detach().cpu().numpy()[:, :, 0]
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
        Image.fromarray(out_img, 'RGB').save("%s/test/%s.png"%(savedir, txt))

        return
        
    # def save_init():
    #     out = Blending_train_stereo(input_coords, images, gtflows, h_res, w_res)
    #     [save_flow((gtflows*masks)[idx], "GT{}".format(idx)) for idx in range(2)]
    #     [save_img(out[idx], "GT{}".format(idx)) for idx in range(2)]
    #     [save_img((torch.flip(images, [0])*masks)[idx], "GTM{}".format(idx)) for idx in range(2)]
    #     return

    def save_progress():
        save_img(out[0], "train")
        save_img(images[0], "trainGT")
        save_img(mask[0], "trainmask")
        save_img(shift_mask[0][0], "trainmask_s")
        save_flow(flow[0], "train", max_mot=max_motion)
        save_flow(gtflow[0], "trainGT", max_mot=max_motion)
        with torch.no_grad():
            model_out = flow_model(input_coordsV[:1, 1:])

            if args.out_rgb:
                if input_coords[:1, -1] == 0.03:
                    planes = [model_out[:, 4*i:4*i+1, :, base_shift - offsets[i]:base_shift + w_res - offsets[i]] for i in range(len(offsets))]
                    rgb_layer = [model_out[:, 4*i+1:4*i+4, :, base_shift - offsets[i]:base_shift + w_res - offsets[i]] for i in range(len(offsets))]
                elif input_coords[:1, -1] == 0.06:
                    planes = [model_out[:, 4*i:4*i+1, :, base_shift + offsets[i]:base_shift + w_res + offsets[i]] for i in range(len(offsets))]
                    rgb_layer = [model_out[:, 4*i+1:4*i+4, :, base_shift - offsets[i]:base_shift + w_res - offsets[i]] for i in range(len(offsets))]
                else:
                    exit()             
            else:
                if input_coordsV[:1, -1] == 0.03:
                    planes = [model_out[:, i:i+1, :, base_shift - offsets[i]:base_shift + w_res - offsets[i]] for i in range(len(offsets))]
                elif input_coordsV[:1, -1] == 0.06:
                    planes = [model_out[:, i:i+1, :, base_shift + offsets[i]:base_shift + w_res + offsets[i]] for i in range(len(offsets))]
                else:
                    exit()
            jacobianX = merge_planes(planes, gtflowV)
            
            jacobianY = torch.zeros_like(jacobianX).cuda()
            jacobian  = torch.cat((jacobianX, jacobianY), dim=1)
            abs_gt = torch.abs(gtflowV[:, :1])
            shift_maskV = [torch.logical_and(abs_gt >= bounds[i][0], abs_gt <= bounds[i][1]).detach() for i in range(len(offsets))]
            outV, flowV, _ = Blend(indicesV, imagesV[1:], jacobian)

            if args.out_rgb:
                rgb_wrap = None
                for tmp_rgb, tmp_mask in zip(rgb_layer, shift_mask):
                    if rgb_wrap is None:
                        rgb_wrap = tmp_mask*tmp_rgb
                    else:
                        rgb_wrap += tmp_mask*tmp_rgb

                save_img(rgb_wrap[0], "rgb_wrap")

        save_img(outV[0], "")
        save_img(imagesV[0], "GT")
        save_img(maskV[0], "mask")
        # save_img(shift_maskV[0][0], "mask_s")
        save_flow(flowV[0], "", max_mot=max_motion_v)
        save_flow((gtflowV)[0], "GT", max_mot=max_motion_v)
        [save_flow((shift_maskV[i]*gtflowV)[0], "GTSM0{}".format(i), shift_mask=shift_maskV[i][0, 0].detach().cpu().numpy(), max_mot=max_motion_v) for i in range(len(offsets))]
        [save_flow((shift_maskV[i]*gtflowV)[0], "GTSMN0{}".format(i), max_mot=max_motion_v) for i in range(len(offsets))]
        [save_flow((-planes[i])[0], "SM0{}".format(i), max_mot=max_motion_v) for i in range(len(offsets))]
        [save_img((shift_maskV[i].expand(-1, 3, h_res, w_res))[0], "mask_s0{}".format(i)) for i in range(len(offsets))]
        return

    def save_progress_debug(rgb_warp, rgb_gt, iter):
        save_img(rgb_warp[0], f"train_{iter}")
        save_img(rgb_gt[0], f"trainGT")


    savedir = init(args)

    # load image and mask
    imgR_path = "./fall/0001R.png"
    imgL_path = "./fall/0001L.png"
    imgR_pil = Image.open(imgR_path).convert("RGB")
    imgL_pil = Image.open(imgL_path).convert("RGB")
    imgR_pil = imgR_pil.resize((480, 270), Image.LANCZOS )
    imgL_pil = imgL_pil.resize((480, 270), Image.LANCZOS )
    w, h = imgR_pil.size
    w_res, h_res = w, h
    imgR = torch.from_numpy(np.array(imgR_pil)).cuda().permute(2,0,1)/255.
    imgL = torch.from_numpy(np.array(imgL_pil)).cuda().permute(2,0,1)/255.


    # load model

    saved_dict = torch.load("%s/trained model/model.ckpt"%(savedir))
    offsets = saved_dict['offsets']
    base_shift = saved_dict['base_shift']
    num_planes = saved_dict['num_planes']

    video_pairs = Video(args, args.dataset)
    dataloader = DataLoader(video_pairs, batch_size=1, shuffle=True)
    print(video_pairs)
    
    imagesV, input_coordsV, gtflowV, maskV, indicesV, gtflowV1, max_motion = video_pairs.__getitem__(0)
    h_res, w_res = h, w
    print(torch.max(gtflowV), torch.min(gtflowV), torch.max(gtflowV1), torch.min(gtflowV1))
    # exit()
    # save_init()
    max_motion_v = torch.min(torch.max(torch.abs(gtflowV)), torch.max(torch.abs(gtflowV1)))
    min_motion = torch.min(torch.min(torch.abs(gtflowV)), torch.min(torch.abs(gtflowV1)))
    print(max_motion, min_motion)
    # exit()
    
    offsets, bounds, base_shift = get_planes(max_motion, num_planes, w_res)


        # saved_dict = torch.load("%s/net.pth"%(savedir))
        # base_shift = 0
        # num_planes = 1

    # print(offsets * 2, bounds, bounds * w_res, base_shift, max_motion * w_res)
    # print([[i, i+1, base_shift + offsets[i], base_shift + w_res + offsets[i]] for i in range(len(offsets))])
    # print([[bounds[i][0], bounds[i][1]] for i in range(len(offsets))])

    # load flow for L and R
    if not args.nn_blend:
        flow_L = torch.from_numpy(np.load("./fall/disp_0_1.npy")).cuda().permute(2, 0, 1)[None]
        flow_R = torch.from_numpy(np.load("./fall/disp_1_0.npy")).cuda().permute(2, 0, 1)[None]
        flow_L = F.interpolate(flow_L, size=(h,w), mode='bilinear', align_corners=True)   
        flow_R = F.interpolate(flow_R, size=(h,w), mode='bilinear', align_corners=True)  

        flow_L = flow_L[:,:1,:,:]
        flow_R = flow_R[:,:1,:,:]
        flow_L = torch.abs(flow_L)
        shift_mask_L = [torch.logical_and(flow_L >= bounds[i][0], flow_L <= bounds[i][1]).detach() for i in range(len(offsets))]
        flow_R = torch.abs(flow_R)
        shift_mask_R = [torch.logical_and(flow_R >= bounds[i][0], flow_R <= bounds[i][1]).detach() for i in range(len(offsets))]


        # load flow for interpolated 
        inter_list = ["0", "1"]
        for file in os.listdir("./fall"):
            if not file.endswith(".npy"):
                continue
            if "disp_0_1.npy" == file or "disp_1_0.npy"==file:
                continue
            tmp = file.split(".npy")[0]
            index = tmp.split("_")[-1]
            inter_list.append(index)
        inter_list.sort()


    if args.net=="conv":
        L_tag = torch.tensor([0.03]).cuda().unsqueeze(0).unsqueeze(0).unsqueeze(0).to(torch.float32)
        R_tag = torch.tensor([0.06]).cuda().unsqueeze(0).unsqueeze(0).unsqueeze(0).to(torch.float32)

        flow_model = XfieldsFlow(h, w + base_shift * 2, inChannels=1, ngf=args.nfg, outChannels=3*num_planes).cuda()
        flow_model.load_state_dict(saved_dict['state_dict'])

    elif args.net=="mlp":

        L_tag = torch.tensor([0]).cuda()
        R_tag = torch.tensor([1]).cuda()

        flow_model = VIINTER(n_emb = 2, norm_p = 1, inter_fn=linterp, D=8, z_dim = 128, in_feat=2, out_feat=3*args.num_planes, W=512, with_res=False, with_norm=True).cuda()
        flow_model.load_state_dict(saved_dict['state_dict'])
        if args.nn_blend:
            if args.blendnn_type=="unet":
                net_blend = Unet_Blend(3*args.num_planes, 3, 4, (h_res, w_res)).cuda()
            elif args.blendnn_type=="mlp":
                net_blend = MLP_blend(D=args.mlp_d, in_feat=3*args.num_planes, out_feat=3, W=args.mlp_w, with_res=False, with_norm=True).cuda()

            net_blend.load_state_dict(saved_dict['nn_blend'])
            # flow_model.load_state_dict(saved_dict)

        coords_h = np.linspace(-1, 1, h_res, endpoint=False)
        coords_w = np.linspace(-1, 1,  w_res + base_shift * 2, endpoint=False)
        xy_grid = np.stack(np.meshgrid(coords_w, coords_h), -1)
        xy_grid = torch.FloatTensor(xy_grid).cuda()
        # if not args.rowbatch:
        grid_inp = xy_grid.view(-1, 2).contiguous().unsqueeze(0)

        dx = torch.from_numpy(coords_w).float().cuda()
        dy = torch.from_numpy(coords_h).float().cuda()
        meshy, meshx = torch.meshgrid((dy, dx))

    nparams_decoder = sum(p.numel() for p in flow_model.parameters() if p.requires_grad)
    print('Number of learnable parameters (decoder): %d' %(nparams_decoder))
    print('Number of planes: %d' %(args.num_planes), num_planes)

    # ----------------------------  starts --------------------------------------


    if args.net=="conv":
        # for left tag
        out = (1 + flow_model(L_tag))*0.5
        multi_out = [out[:, 3*i:3*i+3, :, base_shift - offsets[i]:base_shift + w_res - offsets[i]] for i in range(len(offsets))]
        blend = None
        for l in range(num_planes):
            if blend is None:
                blend = multi_out[l]*shift_mask_L[l]
            else:
                blend += multi_out[l]*shift_mask_L[l]
            save_img(multi_out[l], f"rgbL_{l}")
            save_img(shift_mask_L[l], f"maskL_{l}")
        save_img(blend, f"L")

        # for right tag
        out = (1 + flow_model(R_tag))*0.5
        multi_out = [out[:, 3*i:3*i+3, :, base_shift - offsets[i]:base_shift + w_res - offsets[i]] for i in range(len(offsets))]
        blend = None
        for l in range(num_planes):
            if blend is None:
                blend = multi_out[l]*shift_mask_R[l]
            else:
                blend += multi_out[l]*shift_mask_R[l]
            save_img(multi_out[l], f"rgbR_{l}")
            save_img(shift_mask_R[l], f"maskR_{l}")
        save_img(blend, f"R")

    elif args.net=="mlp":

        with torch.no_grad():

            # left tag
            out = flow_model(grid_inp, L_tag)
            out = out.reshape(1, h_res, -1, 3*args.num_planes).permute(0,3,1,2)
            multi_out = [out[:, 3*i:3*i+3, :, base_shift - offsets[i]:base_shift + w_res - offsets[i]] for i in range(len(offsets))]
            if args.nn_blend:
                for l in range(args.num_planes):
                    save_img(out[:,3*l:3*l+3,:,:], f"L_layer{l}")
                    save_img(multi_out[l], f"rgbL_{l}")
                multi_out = torch.cat(multi_out,dim=1)
                blend = net_blend(2*multi_out-1)
            else:
                blend = None
                for l in range(args.num_planes):
                    if blend is None:
                        blend = multi_out[l]*shift_mask_L[l]
                    else:
                        blend += multi_out[l]*shift_mask_L[l]
                    save_img(out[:,3*l:3*l+3,:,:], f"L_layer{l}")
                    save_img(multi_out[l], f"rgbL_{l}")
                    save_img(shift_mask_L[l], f"maskL_{l}")
            save_img(blend, f"L")

            # right tag
            out = flow_model(grid_inp, R_tag)
            out = out.reshape(1, h_res, -1, 3*args.num_planes).permute(0,3,1,2)
            multi_out = [out[:, 3*i:3*i+3, :, base_shift + offsets[i]:base_shift + w_res + offsets[i]] for i in range(len(offsets))]
            
            if args.nn_blend:
                for l in range(args.num_planes):
                    save_img(out[:,3*l:3*l+3,:,:], f"R_layer{l}")
                    save_img(multi_out[l], f"rgbR_{l}")
                multi_out = torch.cat(multi_out,dim=1)
                blend = net_blend(2*multi_out-1)
            else:
                blend = None
                for l in range(args.num_planes):
                    if blend is None:
                        blend = multi_out[l]*shift_mask_R[l]
                    else:
                        blend += multi_out[l]*shift_mask_R[l]
                    save_img(out[:,3*l:3*l+3,:,:], f"R_layer{l}")
                    save_img(multi_out[l], f"rgbR_{l}")
                    save_img(shift_mask_L[l], f"maskR_{l}")
            save_img(blend, f"R")


    # --------------------------------------------------------- interpolation -------------------------------------------
    # load from npy
    video_out = imageio.get_writer(os.path.join(savedir, "test/inter.mp4"), mode='I', fps=12, codec='libx264')
    video_dict = {}
    for l in range(num_planes):
        print("................", l)
        video_dict[f"video_layer_{l}"] = imageio.get_writer(os.path.join(savedir, f"test/inter_layer{l}.mp4"), mode='I', fps=12, codec='libx264')

    if args.net=="mlp":
        z0 = flow_model.ret_z(torch.LongTensor([0]).cuda()).squeeze()
        z1 = flow_model.ret_z(torch.LongTensor([1]).cuda()).squeeze()

    xp = [0, 1]
    fp = [-2, 2]

    if args.nn_blend:
        inter_list = [j for j in np.linspace(0,1,num=10)]
    print("inter_list: ", inter_list)

    for inter in inter_list:


        if not args.nn_blend:
            if inter=="0":
                flow_inter = flow_L
            elif inter=="1":
                flow_inter = flow_R
            else:
                # load the interpolated disp
                flow_inter = torch.from_numpy(np.load(f"./fall/disp_{inter}.npy")).cuda()

            flow_inter = F.interpolate(flow_inter, size=(h,w), mode='bilinear', align_corners=True)   

            print("flow_inter: ",flow_inter.shape)
            flow_inter = flow_inter[:,:1,:,:]
            flow_inter = torch.abs(flow_inter)
            shift_mask_inter = [torch.logical_and(flow_inter >= bounds[i][0], flow_inter <= bounds[i][1]).detach() for i in range(len(offsets))]

        inter = float(inter)

        if args.net=="conv":

            # interpolation
            inter = float(inter)
            tmp_tag = (1- inter)*L_tag + inter*R_tag

            out = (1 + flow_model(tmp_tag))*0.5
            multi_out = [out[:, 3*i:3*i+3, :, base_shift - offsets[i]:base_shift + w_res - offsets[i]] for i in range(len(offsets))]
            blend = None
            for l in range(num_planes):

                if blend is None:
                    blend = multi_out[l]*shift_mask_inter[l]
                else:
                    blend += multi_out[l]*shift_mask_inter[l]

                tt = (multi_out[l].permute(0, 2, 3, 1) * 255).clamp(0, 255).to(torch.uint8).squeeze(0)
                tt = tt.detach().cpu().numpy()
                video_dict[f"video_layer_{l}"].append_data(tt)

            save_img(blend, f"inter_{inter:0.2f}")
            blend = (blend.permute(0, 2, 3, 1) * 255).clamp(0, 255).to(torch.uint8).squeeze(0)
            blend = blend.detach().cpu().numpy()
            video_out.append_data(blend)

        elif args.net=="mlp":

            with torch.no_grad():
                print("inter: ", inter)
                print("len(offsets): ", len(offsets))

                zi = linterp(inter, z0, z1).unsqueeze(0)
                # print("zi: ", zi)

                out = flow_model.forward_with_z(grid_inp, zi)
                out = out.reshape(1, h_res, -1, 3*args.num_planes).permute(0,3,1,2)
                off_scl = np.interp(inter, xp, fp)

                if args.nn_blend:

                    multi_out_list=[]
                    for i in range(num_planes):
                        meshxw = meshx + (base_shift * 2 + off_scl * offsets[i]) / (w_res + base_shift * 2)
                        grid = torch.stack((meshxw, meshy), 2)[None]
                        multi_out = grid_sample(out[:, 3*i:3*i+3], grid, mode='bilinear', align_corners=True)[:, :, :, :-base_shift*2]
                        multi_out_list.append(multi_out)
                        tt = (multi_out.permute(0, 2, 3, 1) * 255).clamp(0, 255).to(torch.uint8).squeeze(0)
                        tt = tt.detach().cpu().numpy()
                        video_dict[f"video_layer_{i}"].append_data(tt)

                    multi_out = torch.cat(multi_out_list,dim=1)
                    blend = net_blend(2*multi_out-1)

                else:
                    blend = None
                    for i in range(num_planes):

                        print("out: ", out.shape)
                        save_img(out[:,3*i:3*i+3], f"out_{i}_{inter}")
                        print("i: ", i)
                        meshxw = meshx + (base_shift * 2 + off_scl * offsets[i]) / (w_res + base_shift * 2)
                        grid = torch.stack((meshxw, meshy), 2)[None]
                        multi_out = grid_sample(out[:, 3*i:3*i+3], grid, mode='bilinear', align_corners=True)[:, :, :, :-base_shift*2]

                        tt = (multi_out.permute(0, 2, 3, 1) * 255).clamp(0, 255).to(torch.uint8).squeeze(0)
                        tt = tt.detach().cpu().numpy()
                        video_dict[f"video_layer_{i}"].append_data(tt)

                        if blend is None:
                            blend = multi_out*shift_mask_inter[i]
                        else:
                            blend += multi_out*shift_mask_inter[i]

                print("blend: ", blend.shape)

                save_img(blend, f"inter_{inter:0.2f}")
                blend = (blend.permute(0, 2, 3, 1) * 255).clamp(0, 255).to(torch.uint8).squeeze(0)
                blend = blend.detach().cpu().numpy()
                video_out.append_data(blend)


    for key in video_dict:
        video_dict[key].close()

    video_out.close()


if __name__=='__main__':
    
    args = TrainingArguments().parser.parse_args()
    print(args)

    test(args)