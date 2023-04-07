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

# import flow_vis
from xfields_blending import Blending_train_stereo_video_only as Blend
from dataloaderSVPE import Video
from models.xfields import XfieldsFlow
from util.args import TrainingArguments


from PIL import Image

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
        # print("plane ", index, planes[index].shape)
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
        disp = torch.clamp(-flows_out, torch.Tensor([0]).cuda(), max_mot).permute(1, 2, 0).detach().cpu().numpy()[:, :, 0]
        disp_vis = (disp - min_motion.detach().cpu().numpy()) / (max_mot.detach().cpu().numpy() - min_motion.detach().cpu().numpy()) * 255.0
        disp_vis = disp_vis.astype("uint8")
        disp_vis = cv2.applyColorMap(disp_vis, cv2.COLORMAP_INFERNO)
        
        if shift_mask is not None:
            disp_vis[~shift_mask] = [255, 255, 255]
        cv2.imwrite("%s/saved training/flow%s.png"%(savedir, txt),np.uint8(disp_vis))
        return
    
    def save_img(out, txt):
        out_img = np.minimum(np.maximum(out.permute(1, 2, 0).detach().cpu().numpy(),0.0),1.0)
        # cv2.imwrite("%s/saved training/recons%s.png"%(savedir, txt),np.uint8(out_img*255))
        if out_img.shape[-1]==1:
            out_img = np.repeat(out_img,3, axis=-1)

        # print(out_img.shape)
        Image.fromarray(np.uint8(out_img*255), 'RGB').save("%s/saved training/recons%s.png"%(savedir, txt))

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
                    rgb_layer = [model_out[:, 4*i+1:4*i+4, :, base_shift + offsets[i]:base_shift + w_res + offsets[i]] for i in range(len(offsets))]
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

    savedir = init(args)

    # sample = load_imgs(args, args.dataset)
    # images, input_coords, val_coords, gtflows, masks = [x.cuda() for x in sample[:-2]]
    # h_res, w_res = sample[-2:]

    video_pairs = Video(args, args.dataset)
    dataloader = DataLoader(video_pairs, batch_size=1, shuffle=True)
    print(video_pairs)
    
    imagesV, input_coordsV, gtflowV, maskV, indicesV, gtflowV1, max_motion = video_pairs.__getitem__(0)
    _, _, h_res, w_res = imagesV.shape
    print(torch.max(gtflowV), torch.min(gtflowV), torch.max(gtflowV1), torch.min(gtflowV1))
    # exit()
    # save_init()
    max_motion_v = torch.min(torch.max(torch.abs(gtflowV)), torch.max(torch.abs(gtflowV1)))
    min_motion = torch.min(torch.min(torch.abs(gtflowV)), torch.min(torch.abs(gtflowV1)))
    print(max_motion, min_motion)
    # exit()
    
    offsets, bounds, base_shift = get_planes(max_motion, args.num_planes, w_res)

    # print(int(scale*2), int(scale*edge))
    # exit()
    # 30, 4, 7.5 - oscilamp
    # 60, 4, 15 - ambush

    
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

    flow_model  = XfieldsFlow(h_res, w_res + base_shift * 2, inChannels=args.num_freqs_pe*2+1, ngf=args.nfg, outChannels=args.num_planes if not args.out_rgb else args.num_planes*4).cuda()

    nparams_decoder = sum(p.numel() for p in flow_model.parameters() if p.requires_grad)
    print('Number of learnable parameters (decoder): %d' %(nparams_decoder))

    optimizer = Adam(params=flow_model.parameters(), lr=args.lr)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[args.num_iters//3, (2*args.num_iters)//3], gamma=0.4)

    st = time.time()
    loss_t, loss_w_t, loss_f_t, loss_af_t = 0, 0, 0, 0
    coeff, iter = 20, 0

    for _ in (range(args.num_iters//len(dataloader))):
        for data in dataloader:
            images, input_coords, gtflow, mask, indices, _, _ = [x[0] for x in data]

            # print("indices: ", indices)
            # print("indices: ", indices.shape)
            # print("images: ", images.shape)
            # print("gtflow: ", gtflow.shape)
            # print("mask: ", mask.shape)

            optimizer.zero_grad()
            

            model_out  = flow_model(input_coords[:1, 1:])
            # print("model_out ", model_out.shape)

            if args.out_rgb:
                if input_coords[:1, -1] == 0.03:
                    planes = [model_out[:, 4*i:4*i+1, :, base_shift - offsets[i]:base_shift + w_res - offsets[i]] for i in range(len(offsets))]
                    rgb = [model_out[:, 4*i+1:4*i+4, :, base_shift - offsets[i]:base_shift + w_res - offsets[i]] for i in range(len(offsets))]
                elif input_coords[:1, -1] == 0.06:
                    planes = [model_out[:, 4*i:4*i+1, :, base_shift + offsets[i]:base_shift + w_res + offsets[i]] for i in range(len(offsets))]
                    rgb = [model_out[:, 4*i+1:4*i+4, :, base_shift - offsets[i]:base_shift + w_res - offsets[i]] for i in range(len(offsets))]
                else:
                    exit()                
            else:
                rgb = None
                if input_coords[:1, -1] == 0.03:
                    # print("L")
                    planes = [model_out[:, i:i+1, :, base_shift - offsets[i]:base_shift + w_res - offsets[i]] for i in range(len(offsets))]
                elif input_coords[:1, -1] == 0.06:
                    # print("R")
                    planes = [model_out[:, i:i+1, :, base_shift + offsets[i]:base_shift + w_res + offsets[i]] for i in range(len(offsets))]
                else:
                    exit()


            jacobianX = merge_planes(planes, gtflow)
            
            
            jacobianY = torch.zeros_like(jacobianX).cuda()
            jacobian  = torch.cat((jacobianX, jacobianY), dim=1)
            
            
            abs_gt = torch.abs(gtflow[:, :1])
            # print("abs_gt: ", abs_gt.shape)
            shift_mask = [torch.logical_and(abs_gt >= bounds[i][0], abs_gt <= bounds[i][1]).detach() for i in range(len(offsets))]
            
            # print(jacobianY.shape, jacobian.shape, jacobianX.shape, model_out.shape)
            # exit()
            # for tmp_mask in shift_mask:
            #     print("tmp mask: ", tmp_mask.shape)


            out, flow, delta = Blend(indices, images[1:], jacobian)
            # loss, warp_loss, flow_loss = ms_loss(out, images[:1], mask, flow, gtflow, coeff, max(1, (4*iter)/args.num_iters))
            # print(torch.max(shift_mask), torch.min(shift_mask))
            # print(torch.histogram(gtflow.cpu()[:, :1], bins=100))
            # exit()
            # save_progress()
            # exit()
            loss, warp_loss, flow_loss, aux_flow_loss = ms_loss(out, images[:1], mask, flow, gtflow, coeff, 1, shift_mask, planes, delta, rgb_layer=rgb)
            
            loss.backward()
            optimizer.step()
            scheduler.step()

            loss_t += loss.item()
            loss_w_t += warp_loss.item()
            loss_f_t += flow_loss.item()
            loss_af_t += aux_flow_loss.item()
            iter += 1

            # print('\r Iteration %6d Cumulative -> Loss = %3.3f WLoss = %3.3f FLoss = %3.3f AFLoss = %3.3f coeff = %3.3f '%(iter+1, loss_t, loss_w_t, loss_f_t, loss_af_t, coeff),end=" " )

            # if ((iter % args.progress_iter) == (args.progress_iter - 1)):
            if (iter % args.progress_iter) ==0:
                
                print(" elapsed time %3.1f m  Averaged -> Loss = %3.5f WLoss = %3.5f FLoss = %3.5f AFLoss = %3.5f"%((time.time()-st)/60, loss_t/args.progress_iter, loss_w_t/args.progress_iter, loss_f_t/args.progress_iter, loss_af_t/args.progress_iter))
                save_dict = {'state_dict':flow_model.state_dict(), 'offsets':offsets, 'base_shift':base_shift, 'num_planes':args.num_planes}
                torch.save(save_dict, "%s/trained model/model.ckpt"%(savedir))
                save_progress()
                loss_t, loss_w_t, loss_f_t, loss_af_t = 0, 0, 0, 0
                # coeff = 2**((iter*7)/args.num_iters)


if __name__=='__main__':
    
    args = TrainingArguments().parser.parse_args()
    print(args)

    run_training(args)