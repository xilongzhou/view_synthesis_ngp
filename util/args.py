import argparse

class TrainingArguments():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        
        self.parser.add_argument('--dataset',  type=str, 
                            help='path to dataset',        default = 'dataset/time/v')
        self.parser.add_argument('--type',     type=str, nargs= "*",
                            help='xfield type',            default = ['time'])
        self.parser.add_argument('--dim',      type=int, nargs= "*",
                            help='dimension of input xfields',   default = [2])
        self.parser.add_argument('--factor',   type=int,
                            help='downsampling factor',      default = 1)
        self.parser.add_argument('--nfg',      type=int,
                            help='capacity multiplier',    default = 12)
        self.parser.add_argument('--num_n',    type=int,
                            help='number of neighbors',    default = 1)
        self.parser.add_argument('--num_planes',    type=int,
                            help='number of planes',    default = 6)
        self.parser.add_argument('--num_freqs_pe',    type=int,
                            help='#frequencies for positional encoding',    default = 10)
        self.parser.add_argument('--jacob_scale',    type=int,
                            help='scaling for jacobian',    default = 1)
        self.parser.add_argument('--num_iters',    type=int,
                            help='num of iterations',    default = 50000)
        self.parser.add_argument('--progress_iter',    type=int,
                            help='update frequency',    default = 5000)
        self.parser.add_argument('--lr',       type=float,
                            help='learning rate',          default = 0.0001)
        self.parser.add_argument('--dist',    type=float,
                            help='coordinate pair distance',    default = 0.1)
        self.parser.add_argument('--distV',    type=float,
                            help='coordinate pair distance view',    default = 0.03)
        self.parser.add_argument('--sigma',    type=float,
                            help='bandwidth parameter',    default = 0.5)
        self.parser.add_argument('--br',      type=float,
                            help='baseline ratio',         default = 1)
        self.parser.add_argument('--load_pretrained', type=bool,
                            help='loading pretrained model',default = False)
        self.parser.add_argument('--savepath', type=str,
                            help='saving path',             default = 'resultsTest1')

        # my change
        self.parser.add_argument('--debug', help='use debug mode', action='store_true')


        self.parser.add_argument('--out_rgb', help='output rgb layers', action='store_true')
        self.parser.add_argument('--out_disp', help='for debugging', action='store_true')
        self.parser.add_argument('--out_rgbdisp', help='for debugging', action='store_true')
        self.parser.add_argument('--nn_blend', help='use nn for blending', action='store_true')
        self.parser.add_argument('--blendnn_type', help='blend network type: unet | mlp', type=str, default="unet")
        self.parser.add_argument('--net', help='using mlp or conv net: mlp | conv', type=str, default="mlp")
        self.parser.add_argument('--w_vgg', help='weight of loss',type=float, default = 0.0)
        self.parser.add_argument('--w_l1', help='weight of l1 loss',type=float, default = 1.0)
        self.parser.add_argument('--mlp_d', help='nnblend MLP layer number',type=int, default = 2)
        self.parser.add_argument('--mlp_w', help='channel of nnblend MLP',type=int, default = 128)
        self.parser.add_argument('--n_layer', help='layer number of meta MLP',type=int, default = 5)
        self.parser.add_argument('--max_disp', help='max_disp for shifring',type=int, default = 10)
        self.parser.add_argument('--resolution', help='resolution [h,w]',nargs='+', type=int, default = [270, 480])
        self.parser.add_argument('--use_mynet', help='use my network for debugging', action='store_true')
        self.parser.add_argument('--use_norm', help='use my network for debugging', action='store_true')
        self.parser.add_argument('--n_c', help='number of channel in netMet',type=int, default = 256)

        # for meta learning
        self.parser.add_argument('--meta_learn', help='meta learning', action='store_true')
        self.parser.add_argument('--in_out_same', help='make inner and outer loss same', action='store_true')
        self.parser.add_argument('--inner_lr', help='the inner loss',type=float, default = 0.001)
        self.parser.add_argument('--first_order', help='first order', action='store_true')
        self.parser.add_argument('--datapath', help='the path of training dataset', type=str, default="../Dataset/SynData_s1_all")
        self.parser.add_argument('--batch_size', help='batch size of dataloader',type=int, default = 2)
        # self.parser.add_argument('--max_iters', help='max iterations of training',type=int, default = 100000)
        self.parser.add_argument('--inner_steps', help='# of inner step of meta learning',type=int, default = 1)

