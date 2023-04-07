import torch
import torch.nn as nn
from torch.nn.functional import interpolate, grid_sample
from torch.utils.data import DataLoader
from torchvision.transforms import Pad

import numpy as np
import cv2,os,time
from tqdm import tqdm
from kornia.morphology import dilation, erosion
from kornia.filters import spatial_gradient
import sys
sys.path.append('../RAFTW')

import flow_vis
from xfields_blending import Blending_test_stereo_video as Blend
from models.xfields import XfieldsFlow
from util.args import TestArguments
from dataloaderSVPE import VideoTest
from models.maskmodel import Mask
from backWarp import backWarp

torch.set_num_threads(1)
torch.autograd.set_grad_enabled(False)
epsilon = 0.0000001

def init(args, w_res, h_res):

    torch.manual_seed(1234)
    np.random.seed(1234)
    savedir = os.path.join(args.savepath,os.path.split(args.dataset)[1])
    os.makedirs(os.path.join(savedir,"rendered videos"), exist_ok=True)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('%s/rendered videos/rendered.mp4'%(savedir),fourcc, args.fps, (w_res,h_res))
    outF = cv2.VideoWriter('%s/rendered videos/renderedF.mp4'%(savedir),fourcc, args.fps, (w_res,h_res))
    
    return savedir, out, outF

def merge_planes(planes):

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

    out = planes[0]
    # print(torch.max(planes[0]), torch.min(planes[0]))

    # print("=======")
    for index in range(1, len(planes)):
        out = torch.maximum(out, planes[index])
        # print(torch.max(planes[index]), torch.min(planes[index]))
    # print("=======")
    # out = sum(planes)
    
    return out

def run_training(args):
    def save_flow(flows_out, index, writeIndex=[], max_motion=None, min_motion=None):
        # max_motion, min_motion = args.max_motion, 0
        max_motion, min_motion = 1, 0
        
        disp = torch.clamp(flows_out[0], min_motion, max_motion).permute(1, 2, 0).detach().cpu().numpy()[:, :, 0]
        max_motion, min_motion = np.max(disp), np.min(disp)
        disp_vis = (disp - min_motion) / (max_motion - min_motion) * 255.0
        disp_vis = disp_vis.astype("uint8")
        disp_vis = cv2.applyColorMap(disp_vis, cv2.COLORMAP_INFERNO)

        # print(max_motion, min_motion)

        if index in writeIndex:
            # np.save("%s/rendered videos/flow%d.npy"%(savedir, index), flow)
            cv2.imwrite("%s/rendered videos/flow%s.png"%(savedir, index),disp_vis)
        return disp_vis
    
    def save_img(out, index, writeIndex=[]):
        out_img = np.minimum(np.maximum(out[0].permute(1, 2, 0).detach().cpu().numpy(),0.0),1.0)
        out_img = np.uint8(out_img*255)[:, :, [2, 1, 0]]  if out_img.shape[2] == 3 else np.uint8(out_img*255)
        if index in writeIndex:
            cv2.imwrite("%s/rendered videos/recons%04d.png"%(savedir, index),out_img)
        return out_img
        
    def morph(mask, flow):
        maskG = spatial_gradient(flow)[0]+1
        maskBE = torch.where(maskG > 0.95, 1, 0)[:, :1]
        maskBE = erosion(maskBE, kernel)
        maskWD = torch.where(maskG < 1.1, 0, 1)[:, :1]
        maskWD = dilation(maskWD, kernel)
        maskOut = torch.maximum(mask, maskWD)
        maskOut = torch.minimum(maskOut, maskBE)

        return maskOut

    video_pairs = VideoTest(args, args.dataset, interp_coord=0.5)
    dataloader = DataLoader(video_pairs, batch_size=1, shuffle=False)
    print(video_pairs)

    images, _, _ = video_pairs.__getitem__(0)
    _, _, h_res, w_res = images.shape
    
    savedir, out, outF = init(args, w_res, h_res)
    
    
    saved_dict = torch.load("%s/trained model/model.ckpt"%(savedir))
    
    offsets = saved_dict['offsets']
    base_shift = saved_dict['base_shift']
    num_planes = saved_dict['num_planes']

    flow_model = XfieldsFlow(h_res, w_res + base_shift * 2, inChannels=args.num_freqs_pe*2+1, ngf=args.nfg, outChannels=num_planes).cuda()
    flow_model.load_state_dict(saved_dict['state_dict'])


    mask_model = Mask(16, 1)
    # mask_model = Mask(14, 1)
    mask_model.cuda()
    mask_model.load_state_dict(torch.load(args.restore_ckpt))

    nparams_decoder = sum(p.numel() for p in flow_model.parameters() if p.requires_grad)
    print('Number of learnable parameters (decoder): %d' %(nparams_decoder))

    dx = torch.linspace(-1, 1, w_res + base_shift * 2).cuda()
    dy = torch.linspace(-1, 1, h_res).cuda()
    meshy, meshx = torch.meshgrid((dy, dx))

    # grid = torch.stack((meshy, meshx), 2)[None]
    #warped = F.grid_sample(image, grid, mode='bilinear', align_corners=False)

    kernel = torch.ones(1, 5).cuda()
    # kernel = torch.ones(7, 7).cuda()

    size = len(video_pairs)
    # outlist = [index for index in range(len(dataloader))]
    outlist = [0, 20, size//2, size-1]

    warper = None#backWarp(w_res, h_res)
    # padder = Pad((4, 0))
    padder = Pad((0, 12))

    # xp = [-1, 2]
    # fp = [-6, 6]
    xp = [0, 1]
    fp = [-2, 2]

    def render(index, input_coords, indices, video=False):
        a = time.time()
        model_out = flow_model(input_coords[:1, 1:])
        b = time.time()-a
        # print(input_coords[:, 1:])
        # exit()
        
        # jacobiansX = sum([model_out[:, s-2:s-1:, :, shift//2:-shift//2] for s in range(2, 5)])
        # planes = [model_out[:1, i:i+1, :, base_shift:base_shift + w_res] for i in range(len(offsets))]
        
        # planes = [model_out[:1, i:i+1, :, :] for i in range(len(offsets))]
        planes = list(torch.split(model_out, 1, 1))

        # [save_flow((planes[i][:, :, :, base_shift:-base_shift]), "_mp{}".format(i), ["_mp{}".format(i)]) for i in range(len(offsets))]

        planes[0] = planes[0][:, :, :, base_shift:-base_shift]
        point = indices[:1, -1:].item()
        off_scl = np.interp(point, xp, fp)
        # print(point, off_scl)
        
        for i in range(1, len(offsets)):
            meshxw = meshx + (base_shift * 2 + off_scl * offsets[i]) / (w_res + base_shift * 2)
            grid = torch.stack((meshxw, meshy), 2)[None]
            planes[i] = grid_sample(planes[i], grid, mode='bilinear', align_corners=True)[:, :, :, :-base_shift*2]
        
        # [save_flow((planes[i]), "_sp{}".format(i), ["_sp{}".format(i)]) for i in range(len(offsets))]
        # scaling = np.interp(point, xp, fp2)
        # meshxw = meshx + (scaling * shift) / (w_res + shift)
        # grid = torch.stack((meshxw, meshy), 2)[None]
        # planes[2] = grid_sample(planes[2], grid, mode='bilinear', align_corners=True)[:, :, :, :-shift]

        jacobiansX = merge_planes(planes)

        c = time.time()-a

        # jacobiansX = model_out[:, 2:3:, :, shift//2:-shift//2]
        # jacobiansX = model_out[:, 1:2:, :, shift//2:-shift//2]
        # jacobiansX = model_out[:, 0:1:, :, shift//2:-shift//2]
        jacobiansY = torch.zeros_like(jacobiansX).cuda()
        jacobiansT  = torch.cat((jacobiansX, jacobiansY), dim=1)
        jacobians = jacobiansT
        ## temp = jacobians[:, :, :, shift//2:-shift//2]
        # temp0 = jacobiansT[0:1, :, :, shift//2:-shift//2]
        # temp1 = jacobiansT[1:2, :, :, :-shift]
        # temp2 = jacobiansT[2:3, :, :, shift:]
        # jacobians = torch.cat([temp0, temp1, temp2])
        out_jacobians = jacobians

        d = time.time()-a

        # print(input_coords[:, 1:])
        # print(jacobians.shape)
        # exit()
        out_img, flows, wpds, wgts, t = Blend(indices, images, jacobians, w_res, args, warper)


        e = time.time()-a

        # print(torch.amin(flows, (1, 2, 3), keepdims=True))
        # exit()
        div = torch.amax(torch.abs(flows), (1, 2, 3), keepdims=True) + epsilon
        # print(torch.amax(torch.abs(flows), (1, 2, 3), keepdims=True))
        # exit()
        # aa = 0.04
        # flows[:1], flows[1:]= flows[:1]-aa, flows[1:]+aa
        # print(torch.amax(torch.abs(flows), (1, 2, 3), keepdims=True))
        # print(torch.amax(flows, (1, 2, 3), keepdims=True))
        # flows = flows / (div * 0.5) - 1
        flows = flows / (div)# * 0.2)
        flows = (torch.pow(flows, 3)) / 0.5
        # print(torch.max(flows[:1]), torch.min(torch.sigmoid(flows[:1])), div)
        outs = [images[:1], images[1:2], wpds[:1], wpds[1:]]
        mask_in = [padder(x) for x in ([flows[:1], -flows[1:]] + outs)]
        # mask_in = [(x) for x in ([flows[:1], -flows[1:]] + outs)]
        # mask_in = [padder(x) for x in ([flows[:1]] + outs)]

        f = time.time()-a

        # maskT1 = mask_model(torch.cat(mask_in, 1))[:, :, :, 4:-4]# + outs[1]
        maskT1 = mask_model(torch.cat(mask_in, 1))[:, :, 12:-12]# + outs[1]

        g = time.time()-a

        # maskT1 = outs[1]
        maskT1 = torch.clamp(maskT1, 0, 1)
        maskT1 = morph(maskT1, flows[:1])
        maskT2 = 1 - maskT1

        # if (t[:1]) < 0:
        #     im_out =     
        t = abs(t)
        im_out = (maskT1 * outs[2] * t[:1] + maskT2 * outs[3] * t[1:]) / (maskT1*t[:1] + maskT2*t[1:])

        h = time.time()-a

        # print(time.time()-a, b, c, d, e, f, g, h)
        # print("=================================")

        # Flows and Interpolated images for debugging
        out_img    = save_img(im_out, index, [index])#[0, size//2, size-1])
        # flow_color = save_flow(flows*div/t, index, [index])#[0, size//2, size-1])
        flow_color = save_flow(jacobians, index, outlist)#[0, size//2, size-1])
        # _ = save_img(wpds[:1], index+1000, [x+1000 for x in [index]])
        # _ = save_img(wpds[1:], index+2000, [x+2000 for x in [index]])
        # _ = save_img(wgts[:1], index+3000, [x+3000 for x in outlist])
        # _ = save_img(maskT1[:1], index+4000, [x+4000 for x in outlist])
        # _ = save_img(maskT11[:1], index+6000, [x+6000 for x in outlist])
        # _ = save_img(maskE[:1], index+5000, [x+5000 for x in outlist])
        # Interpolated flow and frame videos
        # exit()
        if video:
            out.write(out_img)
            outF.write(flow_color)

        return out_img, flow_color, out_jacobians
    
    def viewsyn(id, input_coords, indices):
        # v_coords  = [args.distV * (1 + 3*i/30 - 1) for i in range(31)]
        # v_indices = [3*i/30 - 1 for i in range(31)]
        v_coords  = [args.distV * (1 + i/30) for i in range(31)]
        v_indices = [i/30 for i in range(31)]
        v_coords  = v_coords [::-1][15:-1] + v_coords[1:-1]  + v_coords [::-1][1:16]
        v_indices = v_indices[::-1][15:-1] + v_indices[1:-1] + v_indices[::-1][1:16]

        for v_coord, v_ind in zip(v_coords, v_indices):
            print( "v_ind: ", v_ind)
            input_coords[:1, -1:] = v_coord
            indices[:1, -1:] = v_ind
            _,_,out_disp = render(id, input_coords, indices, video=True)
            ## this is for saving npy
            np.save(f"disp_{v_ind:.3f}", out_disp.cpu().numpy())


            # print("out_disp shape: ", out_disp.shape)
            # out_disp[:1]/v_ind == out_disp[1:]/(1-v_ind)
            # exit()

            # img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)

            # PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{outdir}/seed{seed:04d}.png')
            # Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{path}/{name}.png')

    viewsynlist = args.viewind
    print("viewsynlist ", viewsynlist)


    for index, data in enumerate(tqdm(dataloader)):
        images, input_coords, indices = [x[0] for x in data]
        out_img, flow_color, _ = render(index, input_coords, indices)

        if index in viewsynlist:
            [out.write(out_img) for x in range(25)]
            # [outF.write(flow_color) for x in range(25)]
            viewsyn(index, input_coords, indices)
            [out.write(out_img) for x in range(25)]
            # [outF.write(flow_color) for x in range(25)]
        

    out.release()
    outF.release()

if __name__=='__main__':
    
    parser = TestArguments().parser
    parser.add_argument('--max_motion',      type=float,
                            help='number of frames to interpolate',  default = 0.0)
    args = parser.parse_args()
    print(args)
    run_training(args)