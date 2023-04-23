import sys
import os
import glob

import argparse


parser = argparse.ArgumentParser()

parser.add_argument('--datapath', type=str, default="")        
parser.add_argument('--savepath', type=str, default="")        
parser.add_argument('--num_iters', type=int, default=500000)        


opt = parser.parse_args()



for idx in os.listdir(opt.datapath):

    save_path = os.path.join(opt.savepath,idx)
    datapath = os.path.join(opt.datapath,idx)

    cmd = "python train2_noloader.py"\
        + " --savepath " + save_path\
        + " --datapath " + datapath\
        + " --num_iters " + str(opt.num_iters)\
        
    print(cmd)
    os.system(cmd)
