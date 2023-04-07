from tkinter import NE
import torch
from xfields_bilinear_sampler import bilinear_sampler_2d
epsilon = 0.00001

def Blending_train_stereo(inputs,
                          Neighbors,
                          jacobians,
                          h_res,w_res):
                
        coord_in       = inputs
        coord_neighbor = torch.flip(inputs, [0])
     
        delta = torch.sign(coord_in - coord_neighbor).repeat(1,1,h_res,w_res)
        
        offset_forward = delta*jacobians
        
        warped_shading = bilinear_sampler_2d(Neighbors,offset_forward)
    
        return warped_shading, offset_forward

def Blending_train_stereo_video_only(inputs,
                                     Neighbor,
                                     jacobian):
                
        coord_in       = inputs[:1]
        coord_neighbor = inputs[1:]
     
        delta = torch.sum(coord_in - coord_neighbor)#.repeat(1,1,h_res,w_res)
        
        offset_forward = delta*jacobian
        
        warped_shading = bilinear_sampler_2d(Neighbor,offset_forward)
    
        return warped_shading, offset_forward, delta

# def Blending_train_stereo(inputs,
#                           Neighbor,
#                           jacobians,
#                           delta,
#                           h_res,w_res):
        
#         coord_in       = inputs
#         coord_neighbor = torch.flip(inputs, [0])
     
#         delta = -delta*torch.sign(coord_in - coord_neighbor)

#         offset_forward = delta*jacobians

#         warped_shading = bilinear_sampler_2d(Neighbor,offset_forward)

#         return warped_shading, offset_forward

# def Blending_train_delta(inputs,
#                    Neighbors,
#                    jacobians,
#                    delta,
#                    h_res,w_res,
#                    args):
                
#         coord_in  = inputs[:1]
#         coord_nbr = inputs[1:]
#         delta = delta * (coord_in - coord_nbr)

#         seq = int(torch.min(inputs, dim=0).values.item())%2
#         jcbn_lft     = jacobians[:1,:2]  if seq else jacobians[:1,2:]
#         jcbn_rgt     = jacobians[:1,2:]  if seq else jacobians[:1,:2]
#         jcbn_nbr_lft = jacobians[1:2,:2] if seq else jacobians[1:2,2:]
#         jcbn_nbr_rgt = jacobians[2:3,2:] if seq else jacobians[2:3,:2]

#         jacobians_center = torch.cat([jcbn_lft, jcbn_rgt])
#         offset_forward   = delta * jacobians_center
#         warped_images    = bilinear_sampler_2d(Neighbors, offset_forward)

#         jacobians_neighbor = torch.cat([jcbn_nbr_lft, jcbn_nbr_rgt])
#         warped_jcbns_nbr   = bilinear_sampler_2d(jacobians_neighbor, offset_forward)
#         offset_backward    = delta * warped_jcbns_nbr

#         dist              = torch.sum(torch.abs(offset_backward-offset_forward),1,keepdim=True)
#         weight            = torch.exp(-args.sigma*w_res*dist)
#         weight_normalized = weight/(torch.sum(weight,0,keepdim=True)+ epsilon)
#         interpolated      = torch.sum(torch.mul(warped_images,weight_normalized),0,keepdim=True)

#         return interpolated, offset_forward

def Blending_train_C2(inputs,
                      Neighbors,
                      jacobians,
                      h_res,w_res,
                      args):
                
        coord_in  = inputs[:1]
        coord_nbr = inputs[1:]
        delta = (coord_in - coord_nbr)

        seq = int(torch.min(inputs, dim=0).values.item())%2
        jcbn_lft     = jacobians[:1,:2]  if seq else jacobians[:1,2:]
        jcbn_rgt     = jacobians[:1,2:]  if seq else jacobians[:1,:2]
        jcbn_nbr_lft = jacobians[1:2,:2] if seq else jacobians[1:2,2:]
        jcbn_nbr_rgt = jacobians[2:3,2:] if seq else jacobians[2:3,:2]

        jacobians_center = torch.cat([jcbn_lft, jcbn_rgt])
        offset_forward   = delta * jacobians_center
        warped_images    = bilinear_sampler_2d(Neighbors, offset_forward)

        jacobians_neighbor = torch.cat([jcbn_nbr_lft, jcbn_nbr_rgt])
        warped_jcbns_nbr   = bilinear_sampler_2d(jacobians_neighbor, offset_forward)
        offset_backward    = delta * warped_jcbns_nbr

        dist              = torch.sum(torch.abs(offset_backward-offset_forward),1,keepdim=True)
        weight            = torch.exp(-args.sigma*w_res*dist)
        weight_normalized = weight/(torch.sum(weight,0,keepdim=True)+ epsilon)
        interpolated      = torch.sum(torch.mul(warped_images,weight_normalized),0,keepdim=True)

        return interpolated, offset_forward

def Blending_train_C2(inputs,
                      Neighbors,
                      jacobians,
                      h_res,w_res,
                      args, delta):
                
        coord_in  = inputs[:1]
        coord_nbr = inputs[1:]
        delta = delta * (coord_in - coord_nbr)

        seq = int(torch.min(inputs, dim=0).values.item())%2
        jcbn_lft     = jacobians[:1,:2]  if seq else jacobians[:1,2:]
        jcbn_rgt     = jacobians[:1,2:]  if seq else jacobians[:1,:2]
        jcbn_nbr_lft = jacobians[1:2,:2] if seq else jacobians[1:2,2:]
        jcbn_nbr_rgt = jacobians[2:3,2:] if seq else jacobians[2:3,:2]
        # print(coord_in - coord_nbr, delta, inputs)

        jacobians_center = torch.cat([jcbn_lft, jcbn_rgt])
        offset_forward   = delta * jacobians_center
        warped_images    = bilinear_sampler_2d(Neighbors, offset_forward)

        jacobians_neighbor = torch.cat([jcbn_nbr_lft, jcbn_nbr_rgt])
        warped_jcbns_nbr   = bilinear_sampler_2d(jacobians_neighbor, offset_forward)
        offset_backward    = delta * warped_jcbns_nbr

        dist              = torch.sum(torch.abs(offset_backward-offset_forward),1,keepdim=True)
        weight            = torch.exp(-args.sigma*w_res*dist)
        weight_normalized = weight/(torch.sum(weight,0,keepdim=True)+ epsilon)
        interpolated      = torch.sum(torch.mul(warped_images,weight_normalized),0,keepdim=True)

        return interpolated, offset_forward

def Blending_train_stereo_video(inputs,
                                Neighbors,
                                jacobians,
                                w_res,
                                args):
                
        coord_in  = inputs[:1]
        coord_nbr = inputs[1:]
        delta = (coord_in - coord_nbr) * torch.Tensor([0.5, 0.5, 1]).cuda()[:, None, None, None]

        # Time interpolation
        delta_time     = delta[:2]
        Neighbors_time = Neighbors[:2]

        jcbn_lft_time     = jacobians[0:1, :2]
        jcbn_rgt_time     = jacobians[2:3, :2]
        jcbn_nbr_lft_time = jacobians[1:2, :2]
        jcbn_nbr_rgt_time = jacobians[3:4, :2]

        jacobians_center_time = torch.cat([jcbn_lft_time, jcbn_rgt_time])
        offset_forward_time   = delta_time * jacobians_center_time
        warped_images_time    = bilinear_sampler_2d(Neighbors_time, offset_forward_time)

        jacobians_neighbor_time = torch.cat([jcbn_nbr_lft_time, jcbn_nbr_rgt_time])
        warped_jcbns_nbr_time   = bilinear_sampler_2d(jacobians_neighbor_time, offset_forward_time)
        offset_backward_time    = delta_time * warped_jcbns_nbr_time

        dist              = torch.sum(torch.abs(offset_backward_time-offset_forward_time),1,keepdim=True)
        weight            = torch.exp(-args.sigma*w_res*dist)
        weight_normalized = weight/(torch.sum(weight,0,keepdim=True)+ epsilon)
        interpolated      = torch.sum(torch.mul(warped_images_time,weight_normalized),0,keepdim=True)

        # View warping
        delta_view     = delta[-1:]
        Neighbors_view = torch.cat((Neighbors[-1:], Neighbors[-1:]))

        jcbn_lft_view = jacobians[0:1, 2:]
        jcbn_rgt_view = jacobians[2:3, 2:]

        jacobians_center_view = torch.cat([jcbn_lft_view, jcbn_rgt_view])
        offset_forward_view   = delta_view * jacobians_center_view
        
        warped = bilinear_sampler_2d(Neighbors_view,offset_forward_view)

        return interpolated, offset_forward_time, warped, offset_forward_view

def Blending_train_stereo_video_time_single(inputs,
                                            Neighbors,
                                            jacobians,
                                            w_res,
                                            args):
                
        coord_in  = inputs[:1]
        coord_nbr = inputs[1:]
        delta = (coord_in - coord_nbr) * torch.Tensor([0.5, 0.5]).cuda()[:, None, None, None]

        # Time interpolation
        delta_time     = delta
        Neighbors_time = Neighbors
        
        jcbn_lft_time     = jacobians[0:1,0:2]
        jcbn_rgt_time     = jacobians[0:1,0:2]
        jcbn_nbr_lft_time = jacobians[1:2,0:2]
        jcbn_nbr_rgt_time = jacobians[2:3,0:2]

        jacobians_center_time = torch.cat([jcbn_lft_time, jcbn_rgt_time])
        offset_forward_time   = delta_time * jacobians_center_time
        warped_images_time    = bilinear_sampler_2d(Neighbors_time, offset_forward_time)

        jacobians_neighbor_time = torch.cat([jcbn_nbr_lft_time, jcbn_nbr_rgt_time])
        warped_jcbns_nbr_time   = bilinear_sampler_2d(jacobians_neighbor_time, offset_forward_time)
        offset_backward_time    = delta_time * warped_jcbns_nbr_time

        dist              = torch.sum(torch.abs(offset_backward_time-offset_forward_time),1,keepdim=True)
        weight            = torch.exp(-args.sigma*w_res*dist)
        weight_normalized = weight/(torch.sum(weight,0,keepdim=True)+ epsilon)
        interpolated      = torch.sum(torch.mul(warped_images_time,weight_normalized),0,keepdim=True)

        return interpolated, offset_forward_time

def Blending_train_stereo_video_time_linear(inputs,
                                            Neighbors,
                                            jacobians,
                                            w_res,
                                            args):
                
        coord_in  = inputs[:1]
        coord_nbr = inputs[1:]
        delta = (coord_in - coord_nbr) * torch.Tensor([0.5, 0.5]).cuda()[:, None, None, None]

        # Time interpolation
        delta_time     = delta
        Neighbors_time = Neighbors
        
        seq = int((coord_nbr[:1]//2).item())%2
        jcbn_lft_time     = jacobians[0:1,0:2] if seq else jacobians[0:1,2:4]
        jcbn_rgt_time     = jacobians[0:1,2:4] if seq else jacobians[0:1,0:2]
        jcbn_nbr_lft_time = jacobians[1:2,0:2] if seq else jacobians[1:2,2:4]
        jcbn_nbr_rgt_time = jacobians[2:3,2:4] if seq else jacobians[2:3,0:2]

        jacobians_center_time = torch.cat([jcbn_lft_time, jcbn_rgt_time])
        offset_forward_time   = delta_time * jacobians_center_time
        warped_images_time    = bilinear_sampler_2d(Neighbors_time, offset_forward_time)

        jacobians_neighbor_time = torch.cat([jcbn_nbr_lft_time, jcbn_nbr_rgt_time])
        warped_jcbns_nbr_time   = bilinear_sampler_2d(jacobians_neighbor_time, offset_forward_time)
        offset_backward_time    = delta_time * warped_jcbns_nbr_time

        dist              = torch.sum(torch.abs(offset_backward_time-offset_forward_time),1,keepdim=True)
        weight            = torch.exp(-args.sigma*w_res*dist)
        weight_normalized = weight/(torch.sum(weight,0,keepdim=True)+ epsilon)
        interpolated      = torch.sum(torch.mul(warped_images_time,weight_normalized),0,keepdim=True)

        return interpolated, offset_forward_time


def Blending_train_stereo_video_time(inputs,
                                     Neighbors,
                                     jacobians,
                                     w_res,
                                     args):
                
        coord_in  = inputs[:1]
        coord_nbr = inputs[1:]
        delta = (coord_in - coord_nbr) * torch.Tensor([0.5, 0.5]).cuda()[:, None, None, None]

        # Time interpolation
        delta_time     = delta
        Neighbors_time = Neighbors

        jcbn_lft_time     = jacobians[0:1, :2]
        jcbn_rgt_time     = jacobians[2:3, :2]
        jcbn_nbr_lft_time = jacobians[1:2, :2]
        jcbn_nbr_rgt_time = jacobians[3:4, :2]

        jacobians_center_time = torch.cat([jcbn_lft_time, jcbn_rgt_time])
        offset_forward_time   = delta_time * jacobians_center_time
        warped_images_time    = bilinear_sampler_2d(Neighbors_time, offset_forward_time)

        jacobians_neighbor_time = torch.cat([jcbn_nbr_lft_time, jcbn_nbr_rgt_time])
        warped_jcbns_nbr_time   = bilinear_sampler_2d(jacobians_neighbor_time, offset_forward_time)
        offset_backward_time    = delta_time * warped_jcbns_nbr_time

        dist              = torch.sum(torch.abs(offset_backward_time-offset_forward_time),1,keepdim=True)
        weight            = torch.exp(-args.sigma*w_res*dist)
        weight_normalized = weight/(torch.sum(weight,0,keepdim=True)+ epsilon)
        interpolated      = torch.sum(torch.mul(warped_images_time,weight_normalized),0,keepdim=True)

        return interpolated, offset_forward_time


def Blending_train_stereo_video_linear(inputs,
                                       Neighbors,
                                       jacobians,
                                       w_res,
                                       args):
                
        coord_in  = inputs[:1]
        coord_nbr = inputs[1:]
        delta = (coord_in - coord_nbr) * torch.Tensor([0.5, 0.5, 1]).cuda()[:, None, None, None]

        # Time interpolation
        delta_time     = delta[:2]
        Neighbors_time = Neighbors[:2]

        seq = int((coord_nbr[:1]//2).item())%2
        jcbn_lft_time     = jacobians[0:1,0:2] if seq else jacobians[0:1,2:4]
        jcbn_rgt_time     = jacobians[0:1,2:4] if seq else jacobians[0:1,0:2]
        jcbn_nbr_lft_time = jacobians[1:2,0:2] if seq else jacobians[1:2,2:4]
        jcbn_nbr_rgt_time = jacobians[2:3,2:4] if seq else jacobians[2:3,0:2]

        jacobians_center_time = torch.cat([jcbn_lft_time, jcbn_rgt_time])
        offset_forward_time   = delta_time * jacobians_center_time
        warped_images_time    = bilinear_sampler_2d(Neighbors_time, offset_forward_time)

        jacobians_neighbor_time = torch.cat([jcbn_nbr_lft_time, jcbn_nbr_rgt_time])
        warped_jcbns_nbr_time   = bilinear_sampler_2d(jacobians_neighbor_time, offset_forward_time)
        offset_backward_time    = delta_time * warped_jcbns_nbr_time

        dist              = torch.sum(torch.abs(offset_backward_time-offset_forward_time),1,keepdim=True)
        weight            = torch.exp(-args.sigma*w_res*dist)
        weight_normalized = weight/(torch.sum(weight,0,keepdim=True)+ epsilon)
        interpolated      = torch.sum(torch.mul(warped_images_time,weight_normalized),0,keepdim=True)

        # View warping
        delta_view     = delta[-1:]
        Neighbors_view = Neighbors[-1:]

        jacobian_view = jacobians[0:1, 4:]

        offset_forward_view   = delta_view * jacobian_view
        
        warped = bilinear_sampler_2d(Neighbors_view,offset_forward_view)

        return interpolated, offset_forward_time, warped, offset_forward_view


def Blending_train(inputs,
                   Neighbors,
                   jacobians,
                   h_res,w_res,
                   args):
                
        coord_in  = inputs[:1]
        coord_nbr = inputs[1:]
        delta = (coord_in - coord_nbr)

        seq = int(torch.min(inputs, dim=0).values.item())%2
        jcbn_lft     = jacobians[0:1] 
        jcbn_rgt     = jacobians[2:3] 
        jcbn_nbr_lft = jacobians[1:2]
        jcbn_nbr_rgt = jacobians[3:4]

        jacobians_center = torch.cat([jcbn_lft, jcbn_rgt])
        offset_forward   = delta * jacobians_center
        warped_images    = bilinear_sampler_2d(Neighbors, offset_forward)

        jacobians_neighbor = torch.cat([jcbn_nbr_lft, jcbn_nbr_rgt])
        warped_jcbns_nbr   = bilinear_sampler_2d(jacobians_neighbor, offset_forward)
        offset_backward    = delta * warped_jcbns_nbr

        dist              = torch.sum(torch.abs(offset_backward-offset_forward),1,keepdim=True)
        weight            = torch.exp(-args.sigma*w_res*dist)
        weight_normalized = weight/(torch.sum(weight,0,keepdim=True)+ epsilon)
        interpolated      = torch.sum(torch.mul(warped_images,weight_normalized),0,keepdim=True)

        return interpolated, offset_forward

def Blending_train_mask(mask_model,
                        inputs,
                        Neighbors,
                        jacobians):
                
        coord_in  = inputs[:1]
        coord_nbr = inputs[1:]
        delta = (coord_in - coord_nbr)

        jcbn_lft     = jacobians[0:1] 
        jcbn_rgt     = jacobians[2:3]

        jacobians_center = torch.cat([jcbn_lft, jcbn_rgt])
        offset_forward   = delta * jacobians_center
        warped_images    = bilinear_sampler_2d(Neighbors, offset_forward)

        mask_left        = mask_model(torch.cat((Neighbors[:1], Neighbors[1:], 
                                                 warped_images[:1], warped_images[1:]), 1))
        mask_right       = 1 - mask_left

        interpolated      = mask_left * warped_images[:1] + mask_right * warped_images[1:]

        return interpolated, offset_forward

def Blending_test_stereo(coord_in,
                         coord_neighbor,
                         Neighbors_im,
                         Neighbors_flow,
                         flows,
                         h_res,w_res,
                         args):
            
        flow           = flows[:1,::]
        delta = (coord_in - coord_neighbor).repeat(1,1,h_res,w_res)
        # delta = (coord_in - coord_neighbor)*aa

        offset_forward = delta*flow
        shading        = Neighbors_im

        warped_shading = bilinear_sampler_2d(shading,offset_forward)
        warped_flow    = bilinear_sampler_2d(Neighbors_flow,offset_forward)

        warped_image    = warped_shading
        offset_backward =  delta*warped_flow

        dist              = torch.sum(torch.abs(offset_backward-offset_forward),1,keepdim=True)
        weight            = torch.exp(-args.sigma*w_res*dist)
        weight_normalized = weight/(torch.sum(weight,0,keepdim=True)+ epsilon)
        interpolated      = torch.sum(torch.mul(warped_image,weight_normalized),0,keepdim=True)

        return interpolated

# def Blending_test_stereo(coord_in,
#                          Neighbors_im,
#                          flows,
#                          h_res,w_res,
#                          args):
            
#         flow           = flows[:1]
#         Neighbors_flow = flows[1:]
#         delta = -(coord_in[:1] - coord_in[1:]).repeat(1,1,h_res,w_res)
#         # delta = (coord_in - coord_neighbor)*aa

#         offset_forward = delta*flow
#         shading        = Neighbors_im

#         warped_shading = bilinear_sampler_2d(shading,offset_forward)
#         warped_flow    = bilinear_sampler_2d(Neighbors_flow,offset_forward)

#         warped_image    = warped_shading
#         offset_backward =  delta*warped_flow

#         dist              = torch.sum(torch.abs(offset_backward-offset_forward),1,keepdim=True)
#         weight            = torch.exp(-args.sigma*w_res*dist)
#         weight_normalized = weight/(torch.sum(weight,0,keepdim=True)+ epsilon)
#         interpolated      = torch.sum(torch.mul(warped_image,weight_normalized),0,keepdim=True)

#         return interpolated

def Blending_test_stereo_video(coord_in,
                               Neighbors_im,
                               flows,
                               w_res,
                               args, backwarp=None):
            
        flow           = flows[:1]
        # Neighbors_flow = flows[1:]
        delta = (coord_in[:1] - coord_in[1:])[:, 1:]

        offset_forward = delta*flow
        shading        = Neighbors_im
        
        if backwarp:
                warped_shading = backwarp(shading, offset_forward*w_res)
        else:
                warped_shading = bilinear_sampler_2d(shading,offset_forward)
        # warped_flow    = bilinear_sampler_2d(Neighbors_flow,offset_forward)

        warped_image    = warped_shading
        # offset_backward =  delta*warped_flow

        # dist              = torch.sum(torch.abs(offset_backward-offset_forward),1,keepdim=True)
        coeff             = (1-torch.abs(delta))
        # weight            = torch.exp(-args.sigma*w_res*dist)
        # weight_normalized = weight/(torch.sum(weight,0,keepdim=True)+ epsilon)
        # interpolated      = torch.sum(torch.mul(warped_image,weight_normalized),0,keepdim=True)

        # return interpolated, offset_forward, warped_image, weight_normalized, coeff
        return None, offset_forward, warped_image, None, coeff

def Blending_test_C2_Tr(coord_in,
                        coord_neighbor,
                        Neighbors_im,
                        Neighbors_flow,
                        jacobian,
                        h_res,w_res,
                        args, delta):
                  
        seq = int(torch.min(coord_neighbor, dim=0).values.item())%2
        jacobian = jacobian[:,:2] if seq else jacobian[:,2:]
        Neighbors_flow = Neighbors_flow[:,:2] if seq else Neighbors_flow[:,2:]

        delta = delta * (coord_in - coord_neighbor)#.repeat(1,1,h_res,w_res)

        offset_forward = delta*jacobian
        shading        = Neighbors_im

        warped_shading = bilinear_sampler_2d(shading,offset_forward)
        warped_flow    = bilinear_sampler_2d(Neighbors_flow,offset_forward)

        warped_image    = warped_shading
        offset_backward =  delta*warped_flow

        dist              = torch.sum(torch.abs(offset_backward-offset_forward),1,keepdim=True)
        coeff             = (1-torch.abs(coord_in - coord_neighbor))
        weight            = coeff * torch.exp(-args.sigma*w_res*dist)
        weight_normalized = weight/(torch.sum(weight,0,keepdim=True)+ epsilon)
        interpolated      = torch.sum(torch.mul(warped_image,weight_normalized),0,keepdim=True)

        return interpolated, offset_forward

def Blending_test_C2(coord_in,
                     coord_neighbor,
                     Neighbors_im,
                     Neighbors_flow,
                     jacobian,
                     w_res,
                     args):
                  
        seq = int(torch.min(coord_neighbor, dim=0).values.item())%2
        jacobian = jacobian[:,:2] if seq else jacobian[:,2:]
        Neighbors_flow = Neighbors_flow[:,:2] if seq else Neighbors_flow[:,2:]

        delta = (coord_in - coord_neighbor)#.repeat(1,1,h_res,w_res)

        offset_forward = delta*jacobian
        shading        = Neighbors_im

        warped_shading = bilinear_sampler_2d(shading,offset_forward)
        warped_flow    = bilinear_sampler_2d(Neighbors_flow,offset_forward)

        warped_image    = warped_shading
        offset_backward =  delta*warped_flow

        dist              = torch.sum(torch.abs(offset_backward-offset_forward),1,keepdim=True)
        coeff             = (1-torch.abs(coord_in - coord_neighbor))
        weight            = torch.exp(-args.sigma*w_res*dist)
        weight_normalized = weight/(torch.sum(weight,0,keepdim=True)+ epsilon)
        interpolated      = torch.sum(torch.mul(warped_image,weight_normalized),0,keepdim=True)

        return interpolated, offset_forward, warped_image, weight_normalized, coeff

def time_test(coord_in,
                  coord_neighbor,
                  Neighbors_im,
                  Neighbors_flow,
                  jacobian,
                  w_res,
                  args):
                  
        # jacobian = jacobian[:,:2]
        # Neighbors_flow = Neighbors_flow[:,:2]

        delta = (coord_in - coord_neighbor)#.repeat(1,1,h_res,w_res)

        offset_forward = delta*jacobian
        shading        = Neighbors_im

        warped_shading = bilinear_sampler_2d(shading,offset_forward)
        # warped_flow    = bilinear_sampler_2d(Neighbors_flow,offset_forward)

        warped_image    = warped_shading
        # offset_backward =  delta*warped_flow

        # dist              = torch.sum(torch.abs(offset_backward-offset_forward),1,keepdim=True)
        coeff             = (1-torch.abs(coord_in - coord_neighbor))
        #------>> weight            = coeff * torch.exp(-args.sigma*w_res*dist) ########<---------
        # weight            = torch.exp(-args.sigma*w_res*dist)
        # weight_normalized = weight/(torch.sum(weight,0,keepdim=True)+ epsilon)
        
        # weightT            = torch.exp(-args.sigma*w_res*dist)
        # weight_normalizedT = weightT/(torch.sum(weightT,0,keepdim=True)+ epsilon)
        # interpolated       = torch.sum(torch.mul(warped_image,weight_normalized),0,keepdim=True)
        
        # return interpolated, offset_forward, warped_image, weight_normalized
        return offset_forward, warped_image, coeff

def Blending_test(coord_in,
                  coord_neighbor,
                  Neighbors_im,
                  Neighbors_flow,
                  jacobian,
                  w_res,
                  args):
                  
        # jacobian = jacobian[:,:2]
        # Neighbors_flow = Neighbors_flow[:,:2]

        delta = (coord_in - coord_neighbor)#.repeat(1,1,h_res,w_res)

        offset_forward = delta*jacobian
        shading        = Neighbors_im

        warped_shading = bilinear_sampler_2d(shading,offset_forward)
        warped_flow    = bilinear_sampler_2d(Neighbors_flow,offset_forward)

        warped_image    = warped_shading
        offset_backward =  delta*warped_flow

        dist              = torch.sum(torch.abs(offset_backward-offset_forward),1,keepdim=True)
        coeff             = (1-torch.abs(coord_in - coord_neighbor))
        #------>> weight            = coeff * torch.exp(-args.sigma*w_res*dist) ########<---------
        weight            = torch.exp(-args.sigma*w_res*dist)
        weight_normalized = weight/(torch.sum(weight,0,keepdim=True)+ epsilon)
        
        # weightT            = torch.exp(-args.sigma*w_res*dist)
        # weight_normalizedT = weightT/(torch.sum(weightT,0,keepdim=True)+ epsilon)
        interpolated       = torch.sum(torch.mul(warped_image,weight_normalized),0,keepdim=True)
        
        # return interpolated, offset_forward, warped_image, weight_normalized
        return interpolated, offset_forward, warped_image, weight_normalized, coeff

def Blending_testM1(coord_in,
                  coord_neighbor,
                  Neighbors_im,
                  jacobian,
                  w_res,
                  args, backwarp=None):
                  
        # jacobian = jacobian[:,:2]
        # Neighbors_flow = Neighbors_flow[:,:2]

        delta = (coord_in - coord_neighbor)#.repeat(1,1,h_res,w_res)

        offset_forward = delta*jacobian
        shading        = Neighbors_im

        # warped_shading = bilinear_sampler_2d(shading,offset_forward)
        if backwarp:
                warped_shading = backwarp(shading, offset_forward*w_res)
        else:
                warped_shading = bilinear_sampler_2d(shading,offset_forward)

        warped_image    = warped_shading

        coeff             = (1-torch.abs(coord_in - coord_neighbor))
        
        # return interpolated, offset_forward, warped_image, weight_normalized
        return offset_forward, warped_image, coeff

def Blending_test4In(coord_in,
                     coord_neighbor,
                     Neighbors_im,
                     jacobianT,
                     jacobianV,
                     w_res,
                     args):

        delta = (coord_in - coord_neighbor)#.repeat(1,1,h_res,w_res)
        # print("+++")
        # print(delta)
        # print(jacobianT.shape, jacobianV.shape)

        offset_forward = delta[:, :1]*jacobianT + delta[:, 1:]*jacobianV
        # print(offset_forward.shape)
        # print(Neighbors_im.shape)

        warped_image = bilinear_sampler_2d(Neighbors_im,offset_forward)
        # print(warped_image.shape)
        # exit()

        coeff             = (1-torch.abs(coord_in - coord_neighbor))
        
        return offset_forward, warped_image, coeff

def Blending_test_sv(coord_in,
                     coord_neighbor,
                     Neighbors_im,
                     Neighbors_jcb,
                     jacobian,
                     w_res,
                     args):
                  
        jacobian_time = jacobian[:,:2]
        jacobian_view = jacobian[:,2:]
        
        Neighbors_jcb_time = Neighbors_jcb[:,:2]
        Neighbors_jcb_view = Neighbors_jcb[:,2:]

        delta = (coord_in - coord_neighbor)#.repeat(1,1,h_res,w_res)
        delta_time = delta[:, :1]
        delta_view = delta[:, 1:]
        
        offset_forward  = delta_view*jacobian_view
        offset_forward += delta_time*jacobian_time

        warped_shading    = bilinear_sampler_2d(Neighbors_im       ,offset_forward)
        warped_view_flow  = bilinear_sampler_2d(Neighbors_jcb_view ,offset_forward)
        warped_time_flow  = bilinear_sampler_2d(Neighbors_jcb_time ,offset_forward)

        warped_image     = warped_shading
        offset_backward  = delta_view*warped_view_flow 
        offset_backward += delta_time*warped_time_flow

        dist              = torch.sum(torch.abs(offset_backward-offset_forward),1,keepdim=True)
        
        coeff             = torch.prod((1-torch.abs(coord_in - coord_neighbor)), 1, keepdim=True)
        # exit()
        #------>> weight            = coeff * torch.exp(-args.sigma*w_res*dist) ########<---------
        weight            = torch.exp(-args.sigma*w_res*dist)
        weight_normalized = weight/(torch.sum(weight,0,keepdim=True)+ epsilon)
        interpolated      = torch.sum(torch.mul(warped_image,weight_normalized),0,keepdim=True)
        
        # return interpolated, offset_forward, warped_image, weight_normalized
        return interpolated, offset_forward, warped_image, weight_normalized, coeff

def Blending_test_mask(mask_model,
                       coord_in,
                       coord_neighbor,
                       Neighbors_im,
                       jacobian):

        delta = (coord_in - coord_neighbor)#.repeat(1,1,h_res,w_res)

        offset_forward = delta*jacobian

        warped_images = bilinear_sampler_2d(Neighbors_im, offset_forward)

        mask_left        = mask_model(torch.cat((Neighbors_im[:1], Neighbors_im[1:], 
                                                 warped_images[:1], warped_images[1:]), 1))
        
#     mask21 = maskModel(torch.cat((flow_out*0.5*640, -flow_out*0.5*640, image1, image3, wimage1, wimage3, weights_0, weights_1), 1))+weights_0
        mask_right       = 1 - mask_left

        coeff             = (1-torch.abs(coord_in - coord_neighbor))
        interpolated      = coeff[:1] * mask_left * warped_images[:1] +\
                            coeff[1:] * mask_right * warped_images[1:]
        interpolated      = interpolated / (coeff[:1] * mask_left + coeff[1:] * mask_right)

        return interpolated, offset_forward, warped_images, mask_left

# def Blending_test_mask(mask_model,
#                        coord_in,
#                        coord_neighbor,
#                        Neighbors_im,
#                        jacobian):

#         delta = (coord_in - coord_neighbor)#.repeat(1,1,h_res,w_res)

#         offset_forward = delta*jacobian

#         warped_images = bilinear_sampler_2d(Neighbors_im, offset_forward)

#         # mask_left        = mask_model(torch.cat((Neighbors_im[:1], Neighbors_im[1:], 
#         #                                          warped_images[:1], warped_images[1:]), 1))
        
#     mask21 = maskModel(torch.cat((flow_out*0.5*640, -flow_out*0.5*640, image1, image3, wimage1, wimage3, weights_0, weights_1), 1))+weights_0
#         mask_right       = 1 - mask_left

#         coeff             = (1-torch.abs(coord_in - coord_neighbor))
#         interpolated      = coeff[:1] * mask_left * warped_images[:1] +\
#                             coeff[1:] * mask_right * warped_images[1:]
#         interpolated      = interpolated / (coeff[:1] * mask_left + coeff[1:] * mask_right)

#         return interpolated, offset_forward, warped_images, mask_left