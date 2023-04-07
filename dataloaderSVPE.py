import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

import numpy as np
from PIL import Image
import glob

from posenc import get_embedder
from xfields_bilinear_sampler import bilinear_sampler_2d

def _pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def gen_mask(gt_flows):
    gt_flow1, gt_flow2 = gt_flows
    # print(".........................flow.................................",gt_flow1.shape, gt_flow2.shape)
    flow_out = bilinear_sampler_2d(-gt_flow1, gt_flow2)
    
    mask = 1-torch.mean(torch.abs(flow_out-gt_flow2),1,keepdims=True)
    mask = (mask > 0.999)*1

    return mask

class Video(Dataset):
    def __init__(self, args, dataset):
        def save_flow():
            ind = training_pairs[idx]
            key = "{}_{}".format(ind[0], ind[1])
            # if ind[0] > ind[1]:
            #     flow = torch.load(dataset+"/f{}.pt".format(key)).cuda()
                
            # else:
            # flow = torch.load(dataset+"/f{}.pt".format(key)).cuda()
            # flow[:, 1:] = 0
            flow = tT(np.load(dataset+"/disp_{}.npy".format(key))).cuda().permute(2, 0, 1)[None]
            self.max_motion = torch.max(torch.max(torch.abs(flow)), self.max_motion)
            self.flows[key] = flow
        
        embed_fn = get_embedder(args.num_freqs_pe)
        self.dataset = dataset
        path_images = glob.glob(dataset+"/*.png")+glob.glob(dataset+"/*.jpg")
        path_images.sort()
        print(len(path_images))
        path_images = path_images[:200]
        print(len(path_images))
        args = args
        
        img =  _pil_loader(path_images[0])
        w_res, h_res = self.size = img.size
        h_res = h_res//args.factor
        w_res = w_res//args.factor
        self.factor = args.factor
        
        tT = torch.Tensor
        self.num_images = num_images = len(path_images)
            
        images = [_pil_loader(path_images[index]) for index in range(num_images)]
        images = [TF.resize(image, (h_res, w_res)) for image in images]
        images = [TF.to_tensor(image) for image in images]

        # coordinates = [tT(embed_fn(index/(num_images-1)))[:, None, None] for index in range(num_images)]
        div = (num_images//2-1)

        '''TEMP'''
        if div==0:
            div += 1
            print("***example with one pair of stereo frames***")
        '''TEMP'''

        coordinates = [tT([embed_fn((i//2)/div).tolist() + [((i%2)+1) * args.distV]])[:, :, None, None] for i in range(num_images)]
        indices = [tT([(i//2), (i%2)])[:, None, None] for i in range(num_images)]
        
        self.images = torch.stack(images).cuda()
        self.coordinates = torch.cat(coordinates).cuda()
        self.indices = torch.stack(indices).cuda()

        print("self indices: ", self.indices)

        num_pairs_lr = num_images//2
        training_pairs = np.array([[2*i, 2*i+1] for i in range(num_pairs_lr)] + [[2*i+1, 2*i] for i in range(num_pairs_lr)])
        self.training_pairs = tT(training_pairs).to(torch.int32).cuda()

        self.flows = {}
        self.max_motion = torch.Tensor([0]).cuda()
        for idx in range(len(training_pairs)):
            save_flow()

    def __len__(self):
        return len(self.training_pairs)

    def __getitem__(self, index):
        pair   = self.training_pairs[index]
        # print("pair, ", pair)
        images = torch.index_select(self.images, 0, pair)
        coordinates = torch.index_select(self.coordinates, 0, pair)
        indices = torch.index_select(self.indices, 0, pair)
        
        flow12, flow21 = [self.flows["{}_{}".format(pair[0], pair[1])], 
                          self.flows["{}_{}".format(pair[1], pair[0])]]

        # print(flow12.shape, flow21.shape, pair)
        mask = gen_mask([flow21, flow12])

        return images, coordinates, flow12, mask, indices, flow21, self.max_motion

    def __repr__(self):
        
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Root Location: {}\n'.format(self.dataset)
        fmt_str += '    Output Resolution: {}\n'.format(self.size)
        fmt_str += '    Downsampling factor: {}\n'.format(self.factor)
        fmt_str += '    Number of training pairs: {}\n'.format(self.__len__())
        return fmt_str

    def return_indices(self,):
        return self.indices

    def return_mask(self):
        flow12, flow21 = [self.flows["{}_{}".format(0, 1)], 
                          self.flows["{}_{}".format(1, 0)]]

        # print(flow12.shape, flow21.shape, pair)
        mask = gen_mask([flow21, flow12])

        return mask

class VideoTest(Dataset):
    def __init__(self, args, dataset, interp_coord):

        embed_fn = get_embedder(args.num_freqs_pe)
        self.dataset = dataset
        path_images = glob.glob(dataset+"/*.png")+glob.glob(dataset+"/*.jpg")
        path_images.sort()
        print(len(path_images))
        path_images = path_images[:200]
        print(len(path_images))
        self.args = args
        
        img =  _pil_loader(path_images[0])
        w_res, h_res = self.size = img.size
        h_res = h_res//args.factor
        w_res = w_res//args.factor
        self.factor = args.factor
        self.interp_coord = interp_coord
        
        tT = torch.Tensor
        self.num_images = num_images = len(path_images)
            
        images = [_pil_loader(path_images[index]) for index in range(num_images)]
        images = [TF.resize(image, (h_res, w_res)) for image in images]
        images = [TF.to_tensor(image) for image in images]

        # coordinates = [tT(embed_fn(index/(len(path_images)-1)))[:, None, None] for index in range(num_images)]
        div = (num_images//2-1)

        '''TEMP'''
        if div==0:
            div += 1
        '''TEMP'''

        coordinates = [tT([embed_fn((i//2)/div).tolist() + [((i%2)+1) * args.distV]])[:, :, None, None] for i in range(num_images)]
        indices = [tT([(i//2), (i%2)])[:, None, None] for i in range(num_images)]
        
        self.images = torch.stack(images).cuda()
        self.coordinates = torch.cat(coordinates).cuda()
        self.indices = torch.stack(indices).cuda()

        num_pairs_lr = num_images//2
        training_pairs = np.array([[2*i, 2*i+1] for i in range(num_pairs_lr)])
        self.training_pairs = tT(training_pairs).to(torch.int32).cuda()

    def __len__(self):
        return len(self.training_pairs)

    def __getitem__(self, index):
        pair   = self.training_pairs[index]
        images = torch.index_select(self.images, 0, pair)

        coordinates = torch.index_select(self.coordinates, 0, pair)
        interp_coords = torch.mean(coordinates, 0, keepdim=True)
        interp_coords[:, -1:] = (self.interp_coord + 1) * self.args.distV
        coordinates = torch.cat((interp_coords, coordinates))

        indices = torch.index_select(self.indices, 0, pair)
        interp_indices = torch.mean(indices, 0, keepdim=True)
        interp_indices[:, 1:] = self.interp_coord
        indices = torch.cat((interp_indices, indices))

        return images, coordinates, indices

    def __repr__(self):
        
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Root Location: {}\n'.format(self.dataset)
        fmt_str += '    Output Resolution: {}\n'.format(self.size)
        fmt_str += '    Downsampling factor: {}\n'.format(self.factor)
        fmt_str += '    Number of training pairs: {}\n'.format(self.__len__())
        return fmt_str