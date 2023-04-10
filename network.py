
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

import torch.nn as nn

from torchmeta.modules import (MetaModule, MetaSequential, MetaConv2d, MetaBatchNorm2d, MetaLinear, MetaLayerNorm, MetaEmbedding)
from torchmeta.utils.gradient_based import gradient_update_parameters

from collections import OrderedDict

import sys

### ------------------------- copying from VIINTER ---------------------------


def linterp(val, low, high):
    res = (1 - val) * low + val * high
    return res

class SineLayer(nn.Module):    
    def __init__(self, in_features, out_features, bias=True, is_first=False, is_res=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.is_res = is_res
        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights(self.linear)
    
    def init_weights(self, layer):
        with torch.no_grad():
            if self.is_first:
                layer.weight.uniform_(-1 / self.in_features, 1 / self.in_features)      
            else:
                layer.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input):
        if self.is_res:
            return input + torch.sin(self.omega_0 * self.linear(input))
        else:
            return torch.sin(self.omega_0 * self.linear(input))

class CondSIREN(nn.Module):
    def __init__(self, n_emb, norm_p = None, inter_fn = None, first_omega_0=30, hidden_omega_0=30, D=8, z_dim = 64, in_feat=2, out_feat=3, W=256, with_res=True, with_norm=True, use_sig=False):
        super().__init__()
        self.norm_p = norm_p
        if self.norm_p is None or self.norm_p == -1:
            self.emb = nn.Embedding(num_embeddings = n_emb, embedding_dim = z_dim)
        else:
            self.emb = nn.Embedding(num_embeddings = n_emb, embedding_dim = z_dim, max_norm=1.0, norm_type=norm_p)

        for i in range(D+1):
            if i == 0:
                layer = SineLayer(in_feat + z_dim, W, is_first=True, is_res=False, omega_0=first_omega_0)
            else:
                layer = SineLayer(W, W, is_first=False, is_res=with_res, omega_0=hidden_omega_0)
            if with_norm:
                layer = nn.Sequential(layer, nn.LayerNorm(W))
            setattr(self, f"layer_{i+1}", layer)

        final_linear = nn.Linear(W, out_feat, bias=True)
        with torch.no_grad():
            final_linear.weight.uniform_(-np.sqrt(6 / W) / hidden_omega_0,  np.sqrt(6 / W) / hidden_omega_0)
        self.final_rgb = nn.Sequential(final_linear, nn.Identity() if not use_sig else nn.Sigmoid())
        self.D = D
        self.inter_fn = inter_fn

    def normalize_z(self, z):
        if self.norm_p == -1:
            z = z / torch.max(z, dim = -1)[0]
        elif self.norm_p is not None:
            z = F.normalize(z, p=self.norm_p, dim=-1)
        else:
            z = z
        return z

    def forward_with_z(self, x, z):
        # print("x , ", x.shape)
        # print("z , ", z.shape)
        xyz_ = torch.cat([x, z.unsqueeze(1).repeat(1, x.shape[1], 1)], dim = -1)
        # print("input shape, ", xyz_.shape)
        for i in range(self.D):
            xyz_ = getattr(self, f'layer_{i+1}')(xyz_)
        rgb = self.final_rgb(xyz_)
        return rgb

    def ret_z(self, ind):
        z = self.emb(ind)
        z = self.normalize_z(z)
        return z

    def get_all_Z_mat(self):
        Z_mat = self.emb.weight
        return self.normalize_z(Z_mat)

    def forward(self, x, ind, ret_z=False):
        # print("ind ", ind.shape)
        # print("x ", x.shape)
        # print("z ", z.shape)

        z = self.emb(ind)
        z = self.normalize_z(z)
        # print("z ", z.shape)
        rgb = self.forward_with_z(x, z)

        if ret_z:
            return rgb, z

        return rgb

class VIINTER(CondSIREN):
    def mix_forward(self, xy_grid_flattened, batch_size=4, chunked=False):
        N = self.emb.num_embeddings
        all_inds = torch.arange(0, N).type(torch.LongTensor).to(xy_grid_flattened.device)
        zs = self.emb(all_inds)
        zs = self.normalize_z(zs)

        rand_inds = torch.randint(0, N, size=(batch_size * 2, 1)).long().squeeze(1)

        slt_zs = zs[rand_inds].reshape(batch_size, 2, -1)
        alphas = torch.rand_like(slt_zs[:, 0:1, 0:1])
        z = self.inter_fn(val=alphas, low=slt_zs[:, 0], high=slt_zs[:, 1]).squeeze(1)
        x = xy_grid_flattened.repeat(batch_size, 1, 1)

        if chunked:
            rgb = torch.zeros((x.shape[0], x.shape[1], 3), device=x.device)
            _p = 8192 * 1
            for ib in range(0, len(rgb), 1):
                for ip in range(0, rgb.shape[1], _p):
                    rgb[ib:ib+1, ip:ip+_p] = self.forward_with_z(x[ib:ib+1, ip:ip+_p], z)
        else:
            rgb = self.forward_with_z(x, z)

        rand_inds = rand_inds.reshape(batch_size, 2)

        return rgb, rand_inds[:, 0], rand_inds[:, 1], alphas, z

class tiny_cuda(nn.Module):
    def __init__(self, n_emb, config, norm_p = None, z_dim = 64, in_feat=2, out_feat=3, debug=False):
        super().__init__()

        try:
            import tinycudann as tcnn
        except ImportError:
            print("This sample requires the tiny-cuda-nn extension for PyTorch.")
            print("You can install it by running:")
            print("============================================================")
            print("tiny-cuda-nn$ cd bindings/torch")
            print("tiny-cuda-nn/bindings/torch$ python setup.py install")
            print("============================================================")
            sys.exit()

        self.debug = debug

        # embedding
        if not self.debug:
            self.norm_p = norm_p
            if self.norm_p is None or self.norm_p == -1:
                self.emb = nn.Embedding(num_embeddings = n_emb, embedding_dim = z_dim)
            else:
                self.emb = nn.Embedding(num_embeddings = n_emb, embedding_dim = z_dim, max_norm=1.0, norm_type=norm_p)

        # tiny-mlp
        self.net = tcnn.Network(n_input_dims=z_dim+in_feat, n_output_dims=out_feat, network_config=config["network"])

    def normalize_z(self, z):
        if self.norm_p == -1:
            z = z / torch.max(z, dim = -1)[0]
        elif self.norm_p is not None:
            z = F.normalize(z, p=self.norm_p, dim=-1)
        else:
            z = z
        return z


    def forward(self, x, ind):

        if not self.debug:
            z = self.emb(ind)
            z = self.normalize_z(z)

            xyz_ = torch.cat([x, z.unsqueeze(1).repeat(1, x.shape[1], 1)], dim = -1).squeeze(0)

            xyz_ = xyz_*0.5 + 0.5
            # print(torch.amin(xyz_), torch.amax(xyz_))

            rgb = self.net(xyz_)

        else:
            rgb = self.net(x)

        return rgb


### ------------------------------------------------------------------------------
### ---------------- Unet postprocessing ------------------------------------------------------

class UNet2dBlock(nn.Module):
    def __init__(self, in_n, out_n, k_size=4, stride=2, pad=(1,1), islast=False):
        super().__init__()
        self.pad = pad
        self.net = nn.Conv2d(in_n, out_n, k_size, stride)
        if islast:
            self.act = nn.Tanh()
        else:
            self.act = nn.LeakyReLU(inplace=True)

    def forward(self, input):
        # print("before pad: ", input.shape)
        input = F.pad(input, self.pad, mode='reflect')
        # print("after pad: ", input.shape)
        return self.act(self.net(input))

class Unet_Blend(nn.Module):
    def __init__(self, in_c, out_c, layer_n, res, n_c=32):
        super().__init__()
        self.layer_n = layer_n

        self.up = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) # bilinear

        padx,pady = self.upsampling_factor_padding(res[0],res[1])

        for i in range(layer_n):
            in_channel = in_c if i==0 else n_c*2**(i-1)
            out_channel = n_c*2**i
            recon_p = (1, 1 + padx[i], 1, 1 + pady[i])
            # print(f"{i}: in {in_channel}, out {out_channel}")
            setattr(self, f"enco_{i}" , UNet2dBlock(in_channel, out_channel, k_size=4, stride=2, pad=recon_p))

        for i in range(layer_n):
            # recon_p = (1, 1 + padx[i], 1, 1 + pady[i])

            in_channel = n_c*2**(layer_n-i-1) if i==0 else n_c*2**(layer_n-i) + n_c*2**(layer_n-i-1)
            out_channel = out_c if i==layer_n-1 else n_c*2**(layer_n-i-1)


            # print(f"{i}: in {in_channel}, out {out_channel}")
            # if i==layer_n-1:
            #     setattr(self, f"deco_{i}" , UNet2dBlock(in_channel, out_channel, k_size=3, stride=1, pad=(1,1,1,1), islast=True))
            # else:
            #     recon_p = (1, 1+padx[-i-2], 1, pady[-i-2])

            #     setattr(self, f"deco_{i}" , UNet2dBlock(in_channel, out_channel, k_size=3, stride=1, pad=recon_p, islast=False))
            #     print(i, recon_p)


            if i==0:
                recon_p = (1, 1+padx[-i-2], 1, pady[-i-2])
            else:
                recon_p = (1, 1, 1, 1)

            if i==layer_n-1:
                setattr(self, f"deco_{i}" , UNet2dBlock(in_channel, out_channel, k_size=3, stride=1, pad=recon_p, islast=True))
            else:
                setattr(self, f"deco_{i}" , UNet2dBlock(in_channel, out_channel, k_size=3, stride=1, pad=recon_p, islast=False))


    def upsampling_factor_padding(self, h_res,w_res):

        padx=[]
        pady=[]
        tmp_x = w_res
        tmp_y = h_res

        for i in range(self.layer_n):

            padx.append(tmp_x%2)
            pady.append(tmp_y%2)
            tmp_x = tmp_x//2
            tmp_y = tmp_y//2

        return padx, pady


    def forward(self, input):

        # encoding
        convs =[]
        for i in range(self.layer_n):
            block = getattr(self, f"enco_{i}" )
            conv = block(input if i==0 else conv)
            convs.append(conv)
            # print(f"{i} conv: {conv.shape}")


        # decoding
        deconvs=[]
        for i in range(self.layer_n):
            block = getattr(self, f"deco_{i}" )
            if i==0:
                deconv = block(self.up(convs[-1]))
            else:
                tmp = torch.cat([deconvs[-1], convs[-1-i]], dim=1)
                deconv = block(self.up(tmp))

            # print(f"{i} deconv: {deconv.shape}")

            deconvs.append(deconv)

        return deconvs[-1]*0.5+0.5

### ---------------- MLP postprocessing ------------------------------------------------------

class MLP_blend(nn.Module):
    def __init__(self, first_omega_0=30, hidden_omega_0=30, D=8,in_feat=2, out_feat=3, W=256, with_res=True, with_norm=True):
        super().__init__()

        self.in_feat = in_feat
        for i in range(D):
            if i == 0:
                layer = SineLayer(in_feat, W, is_first=True, is_res=False, omega_0=first_omega_0)
            else:
                layer = SineLayer(W, W, is_first=False, is_res=with_res, omega_0=hidden_omega_0)
            if with_norm:
                layer = nn.Sequential(layer, nn.LayerNorm(W))
            setattr(self, f"layer_{i+1}", layer)

        final_linear = nn.Linear(W, out_feat, bias=True)
        with torch.no_grad():
            final_linear.weight.uniform_(-np.sqrt(6 / W) / hidden_omega_0,  np.sqrt(6 / W) / hidden_omega_0)
        self.final_rgb = final_linear
        self.D = D
        self.tanh = nn.Tanh()

    def forward(self, x):
        h = x.shape[-2]
        x = x.permute(0,2,3,1).view(-1, self.in_feat).contiguous().unsqueeze(0)

        for i in range(self.D):
            # print("x: ", x.shape)
            x = getattr(self, f'layer_{i+1}')(x)
        rgb = self.tanh(self.final_rgb(x))
        rgb = rgb.reshape(1, h, -1, 3).permute(0,3,1,2)

        return rgb*0.5 + 0.5 # [-1,1] --> [0,1]



### --------------- Meta Model -----------------------------------------------


class SineLayer_meta(MetaModule):    
    def __init__(self, in_features, out_features, bias=True, is_first=False, is_res=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.is_res = is_res
        
        self.in_features = in_features
        self.linear = MetaLinear(in_features, out_features, bias=bias)
        self.init_weights(self.linear)
    
    def init_weights(self, layer):
        with torch.no_grad():
            if self.is_first:
                layer.weight.uniform_(-1 / self.in_features, 1 / self.in_features)      
            else:
                layer.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input, params=None):
        if self.is_res:
            return input + torch.sin(self.omega_0 * self.linear(input, params=params))
        else:
            return torch.sin(self.omega_0 * self.linear(input, params=params))


class CondSIREN_meta(MetaModule):
    def __init__(self, n_emb, norm_p = None, inter_fn = None, first_omega_0=30, hidden_omega_0=30, D=8, z_dim = 64, in_feat=2, out_feat=3, W=256, with_res=True, with_norm=True):
        super().__init__()
        self.norm_p = norm_p
        if self.norm_p is None or self.norm_p == -1:
            self.emb = MetaEmbedding(num_embeddings = n_emb, embedding_dim = z_dim)
        else:
            self.emb = MetaEmbedding(num_embeddings = n_emb, embedding_dim = z_dim, max_norm=1.0, norm_type=norm_p)

        for i in range(D):
            if i == 0:
                layer = SineLayer_meta(in_feat + z_dim, W, is_first=True, is_res=False, omega_0=first_omega_0)
                # layer = SineLayer_meta(in_feat, W, is_first=True, is_res=False, omega_0=first_omega_0)
            else:
                layer = SineLayer_meta(W, W, is_first=False, is_res=with_res, omega_0=hidden_omega_0)
            if with_norm:
                layer = MetaSequential(layer, MetaLayerNorm(W))
            setattr(self, f"layer_{i+1}", layer)

        final_linear = MetaLinear(W, out_feat, bias=True)
        with torch.no_grad():
            final_linear.weight.uniform_(-np.sqrt(6 / W) / hidden_omega_0,  np.sqrt(6 / W) / hidden_omega_0)
        self.final_rgb = MetaSequential(final_linear, nn.Identity())
        self.D = D
        self.inter_fn = inter_fn

    def normalize_z(self, z):
        if self.norm_p == -1:
            z = z / torch.max(z, dim = -1)[0]
        elif self.norm_p is not None:
            z = F.normalize(z, p=self.norm_p, dim=-1)
        else:
            z = z
        return z

    def forward_with_z(self, x, z, params=None):

        xyz_ = torch.cat([x, z.unsqueeze(1).repeat(1, x.shape[1], 1)], dim = -1)
        # print("input shape, ", xyz_.shape)
        for i in range(self.D):
            xyz_ = getattr(self, f'layer_{i+1}')(xyz_, params=self.get_subdict(params, f'layer_{i+1}.linear'))
        rgb = self.final_rgb(xyz_, params=self.get_subdict(params, 'final_rgb'))
        return rgb

    def ret_z(self, ind, params=None):
        z = self.emb(ind, params=self.get_subdict(params, 'emb'))
        z = self.normalize_z(z)
        return z

    def get_all_Z_mat(self):
        Z_mat = self.emb.weight
        return self.normalize_z(Z_mat)

    def forward(self, x, ind, ret_z=False, params=None):
        z = self.emb(ind, params=self.get_subdict(params, 'emb'))
        z = self.normalize_z(z)

        rgb = self.forward_with_z(x, z, params=params)

        if ret_z:
            return rgb, z

        return rgb

### --------------- lookahead model for debugging ---------------

import math

def exists(val):
    return val is not None

def cast_tuple(val, repeat = 1):
    return val if isinstance(val, tuple) else ((val,) * repeat)


# sin activation
class Sine(MetaModule):
    def __init__(self, w0 = 1.):
        super().__init__()
        self.w0 = w0
    def forward(self, x):
        return torch.sin(self.w0 * x)

# siren layer
class Siren(MetaModule):
    def __init__(self, dim_in, dim_out, w0 = 1., c = 6., is_first = False, use_bias = True, activation = None):
        super().__init__()
        self.dim_in = dim_in
        self.is_first = is_first

        weight = torch.zeros(dim_out, dim_in)
        bias = torch.zeros(dim_out) if use_bias else None
        self.init_(weight, bias, c = c, w0 = w0)

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias) if use_bias else None
        # self.linear = MetaLinear(dim_in, dim_out, bias=use_bias)
        self.activation = Sine(w0) if activation is None else activation

        self.flag = activation

    def init_(self, weight, bias, c, w0):
        dim = self.dim_in

        w_std = (1 / dim) if self.is_first else (math.sqrt(c / dim) / w0)
        weight.uniform_(-w_std, w_std)

        if exists(bias):
            bias.uniform_(-w_std, w_std)

    def forward(self, x, params=None):

        out =  F.linear(x, params['weight'], params['bias'])
        # if self.flag is not None:
        #   print('out:', out[0:10,0,...])
        out = self.activation(out)
        # if self.flag is not None:
        #   print('final:', out[0:10,0,...])
        return out

# siren network
class MySirenNet(MetaModule):
    def __init__(self, dim_in, dim_hidden, dim_out, num_layers, w0 = 1., w0_initial = 30., use_bias = True, final_activation = None):
        super().__init__()
        self.num_layers = num_layers
        self.dim_hidden = dim_hidden

        self.layers = nn.ModuleList([])
        for ind in range(num_layers):
            is_first = ind == 0
            layer_w0 = w0_initial if is_first else w0
            layer_dim_in = dim_in if is_first else dim_hidden

            self.layers.append(Siren(
                dim_in = layer_dim_in,
                dim_out = dim_hidden,
                w0 = layer_w0,
                use_bias = use_bias,
                is_first = is_first
            ))

        self.final_activation = nn.Identity() if not exists(final_activation) else final_activation
        self.last_layer = Siren(dim_in = dim_hidden, dim_out = dim_out, w0 = w0, use_bias = use_bias, activation = final_activation)

    def forward(self, x, mods = None, params = None):
        x = x.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input
        mods = cast_tuple(mods, self.num_layers)

        for i, (layer, mod) in enumerate(zip(self.layers, mods)):
            # print('subdict:', self.get_subdict(params, 'layers.{}'.format(i)))
            x = layer(x, params=self.get_subdict(params, 'layers.{}'.format(i)))

            if exists(mod):
                x *= rearrange(mod, 'd -> () d')

        x = self.last_layer(x, params=self.get_subdict(params, 'last_layer'))
        return x



# con-siren network
class ConSirenNet(MetaModule):
    def __init__(self, dim_in, dim_hidden, dim_out, num_layers, fea_c=32, w0 = 1., w0_initial = 30., use_bias = True, final_activation = None, cond_type='conv', N_in=1, n_layer_unet=3, test=False):
        super().__init__()
        self.test = test
        self.N_in = N_in
        self.num_layers = num_layers
        self.dim_hidden = dim_hidden
        self.cond_type = cond_type
        self.n_layer_unet = n_layer_unet
        if self.cond_type=='conv':
            self.con_layer = self._conv_layers(fea_c)
        elif self.cond_type=='unet':
            self.con_layer = MetaUNet(32, layer_n=self.n_layer_unet, N_in=self.N_in, test=test)
        elif self.cond_type=='dir':
            fea_c = 3

        self.layers = nn.ModuleList([])
        for ind in range(num_layers):
            is_first = ind == 0
            layer_w0 = w0_initial if is_first else w0
            if is_first:
                layer_dim_in = dim_in+fea_c
            else:
                layer_dim_in = dim_hidden

            self.layers.append(Siren(
                dim_in = layer_dim_in,
                dim_out = dim_hidden,
                w0 = layer_w0,
                use_bias = use_bias,
                is_first = is_first
            ))

        self.final_activation = nn.Identity() if not exists(final_activation) else final_activation

        self.last_layer = Siren(dim_in = dim_hidden, dim_out = dim_out, w0 = w0, use_bias = use_bias, activation = final_activation)

    def _conv_layers(self, feature_c):
        net = MetaSequential(
            MetaConv(self.N_in*3, feature_c, ksize=3, s=1, p=1),
            nn.LeakyReLU(),
            MetaConv(feature_c, feature_c, ksize=3, s=1, p=1),
            nn.LeakyReLU(),
            MetaConv(feature_c, feature_c, ksize=3, s=1, p=1),
            nn.LeakyReLU(),
            MetaConv(feature_c, feature_c, ksize=3, s=1, p=1),
            nn.Tanh(),
            )
        return net

    def forward(self, x, mods = None, params = None):

        if not self.test:
            coor = x[0].clone().detach().requires_grad_(True) # [H,W,2] allows to take derivative w.r.t. input
            con_img = x[1].clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input
        else:
            coor = x[0]
            con_img = x[1]

        if self.cond_type =='dir':
            imgFea = con_img.reshape(1,self.N_in*3,256,256) if self.N_in>1 else con_img
        else:
            imgFea = self.con_layer(con_img.reshape(1,self.N_in*3,256,256) if self.N_in>1 else con_img, self.get_subdict(params, 'con_layer'))

        if coor.dim()==3:
            imgFea = imgFea.squeeze(0).permute(1,2,0) #[H,W,fea_c+2]
        else:
            imgFea = imgFea.permute(0,2,3,1) #[H,W,fea_c+2]

        x = torch.cat([coor,imgFea],dim=-1)

        mods = cast_tuple(mods, self.num_layers)

        for i, (layer, mod) in enumerate(zip(self.layers, mods)):
            # print('subdict:', self.get_subdict(params, 'layers.{}'.format(i)))
            x = layer(x, params=self.get_subdict(params, 'layers.{}'.format(i)))

            if exists(mod):
                x *= rearrange(mod, 'd -> () d')


        x = self.last_layer(x, params=self.get_subdict(params, 'last_layer'))


        return x


class MetaConv(MetaModule):
    def __init__(self,in_c, out_c, ksize=4, s=2, p=1):
        super().__init__()
        self.metaconv = MetaConv2d(in_c, out_c,  kernel_size=ksize, stride=s, padding=p)

    def forward(self,x, params=None):
        x = self.metaconv(x, params=self.get_subdict(params, 'metaconv'))
        return x

class MetaDeConv(MetaModule):
    def __init__(self,in_c, out_c):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.metaconv1 = MetaConv2d(in_c, out_c, kernel_size=3, stride=1, padding=1)

    def forward(self, x, params=None):
        x = self.metaconv1(self.up(x), params=self.get_subdict(params, 'metaconv1'))
        return x

# siren network
class MetaUNet(MetaModule):
    def __init__(self, dim_hidden, layer_n=3, N_in=1, test=False):
        super().__init__()
        self.test=test
        self.layer_n = layer_n
        self.dim_hidden = dim_hidden
        self.convs = nn.ModuleList([])

        conv_channels={   0: 3*N_in,
                          1: self.dim_hidden,
                          2: self.dim_hidden*2,
                          3: self.dim_hidden*4,
                          4: self.dim_hidden*4
                    }

        for i in range(self.layer_n):
            in_c = conv_channels[i]
            out_c = conv_channels[i+1]

            self.convs.append(MetaConv(in_c, out_c, ksize=4, s=2, p=1))

        self.deconvs = nn.ModuleList([])
        for i in range(self.layer_n):
            in_c = conv_channels[self.layer_n-i] if i==0 else conv_channels[self.layer_n-i]*2 
            out_c = conv_channels[self.layer_n-i-1] if i!=self.layer_n-1 else self.dim_hidden
            self.deconvs.append(MetaDeConv(in_c, out_c))

        # self.lastlayer = nn.Conv2d(out_c, final_c, kernel_size=3, stride=1, padding=1)
        self.Tanh = nn.Tanh()
        self.LeakyReLU = nn.LeakyReLU()

    def forward(self, x, params):
        if not self.test:
            x = x.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input

        # vis image
        # import matplotlib
        # matplotlib.use('Agg')
        # import matplotlib.pyplot as plt
        # from meta_utils import save_image

        # save_image((1+x[0,0:3,:,:].permute(1,2,0))*0.5, '0.png')
        # save_image((1+x[0,3:6,:,:].permute(1,2,0))*0.5, '1.png')


        convs=[]
        for i in range(self.layer_n):
            x = self.LeakyReLU(self.convs[i](x, params=self.get_subdict(params, f'convs.{i}')))
            if i!=self.layer_n-1:
                convs.append(x)

        for i in range(self.layer_n):
            if i==0:
                x = self.LeakyReLU(self.deconvs[i](x, params=self.get_subdict(params, f'deconvs.{i}')))
            elif i==self.layer_n-1:
                temp = torch.cat([x, convs[self.layer_n-1-i]], dim=1)
                x = self.Tanh(self.deconvs[i](temp, params=self.get_subdict(params, f'deconvs.{i}')))
            else:
                temp = torch.cat([x, convs[self.layer_n-1-i]], dim=1)
                x = self.LeakyReLU(self.deconvs[i](temp, params=self.get_subdict(params, f'deconvs.{i}')))

        return x



if __name__=="__main__":

    h=270
    w=480
    base_shift = 35

    L_tag = torch.tensor([0]).cuda()

    net = CondSIREN_meta(n_emb = 2, norm_p = 1, inter_fn=linterp, D=5, z_dim = 128, in_feat=2, out_feat=3*3, W=256, with_res=False, with_norm=True).cuda()
    # net = MySirenNet(2, 256, 3, 5, w0_initial=200., w0=200., final_activation=lambda x: x + .5).cuda()
    # net = ConSirenNet(2, 256, 3, 5, w0_initial=200., w0=200.,final_activation=lambda x: x + .5, cond_type='unet', test=True, N_in=1, n_layer_unet=4).cuda()

    print("net: ", net)

    nparams_decoder = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print('Number of learnable parameters (decoder): %d' %(nparams_decoder))

    coords_h = np.linspace(-1, 1, h, endpoint=False)
    coords_w = np.linspace(-1, 1,  w + base_shift * 2, endpoint=False)
    xy_grid = np.stack(np.meshgrid(coords_w, coords_h), -1)
    xy_grid = torch.FloatTensor(xy_grid).cuda()
    grid_inp = xy_grid.view(-1, 2).contiguous().unsqueeze(0)

    my_params = OrderedDict(net.meta_named_parameters())

    # my_params2 = gradient_update_parameters(net,
    #                                     inner_loss,
    #                                     step_size=args.step_size,
    #                                     first_order=args.first_order)

    # print(my_params)

    # out = net(grid_inp, params=OrderedDict(net.meta_named_parameters()))

    # print(out.shape)

    # define net
    # net2 = Unet_Blend(32, 3, 4, (h,w))
    # net = MLP_blend(D=4, in_feat=32, out_feat=3, W=512, with_res=False, with_norm=True)
    # total_params = sum(p.numel() for p in net.parameters())

    # # print(len(tmp))
    # optimizer = torch.optim.Adam(list(net.parameters()) + list(net2.parameters()), lr=1e-5)

    # for j in 

    # print("net: ", total_params)
    # input = torch.randn(1,32,h,w)

    # out = net(input)

    # print("out:", out.shape)