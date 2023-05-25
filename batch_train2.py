import sys
import os
import glob

import argparse

from typing import List


parser = argparse.ArgumentParser()

parser.add_argument('--datapath', type=str, default="")        
parser.add_argument('--savepath', type=str, default="")        
parser.add_argument('--num_iters', type=int, default=500000)        
parser.add_argument('--range', type=str, default="0-20")        
parser.add_argument('--use_viinter',help='use_viinter',action='store_true')
parser.add_argument('--add_mask',help='add mask',action='store_true')
parser.add_argument('--mask_only',help='mask_only',action='store_true')
parser.add_argument('--hypernet',help='hypernet',action='store_true')
parser.add_argument('--debug',help='debug',action='store_true')


opt = parser.parse_args()

bot = int(opt.range.split("-")[0])
up = int(opt.range.split("-")[1])

# print(bot)
# print(up)

addmask = " --add_mask " if opt.add_mask else ""
maskonly = " --mask_only " if opt.mask_only else ""
hypernet = " --hypernet " if opt.hypernet else ""
debug = " --debug " if opt.debug else ""
use_viinter = " --use_viinter " if opt.use_viinter else ""

for idx,scene_id in enumerate(os.listdir(opt.datapath)):

    if idx<bot or idx>up:
        # print(f"skip idx: {idx}, scene_id: {scene_id}")
        continue
    print(f"process idx: {idx}, scene_id: {scene_id}")

    save_path = os.path.join(opt.savepath,scene_id)
    datapath = os.path.join(opt.datapath,scene_id)

    cmd = "python train2_noloader.py"\
        + " --savepath " + save_path\
        + " --datapath " + datapath\
        + " --num_iters " + str(opt.num_iters)\
        + hypernet + addmask + maskonly + debug + use_viinter
        
    print(cmd)
    os.system(cmd)
