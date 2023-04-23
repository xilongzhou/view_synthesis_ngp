import sys
import os
import glob

import argparse

from typing import List



def num_range(s: str) -> List[int]:
    '''Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints.'''

    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2))+1))
    vals = s.split(',')
    return [int(x) for x in vals]


parser = argparse.ArgumentParser()

parser.add_argument('--datapath', type=str, default="")        
parser.add_argument('--savepath', type=str, default="")        
parser.add_argument('--num_iters', type=int, default=500000)        
parser.add_argument('--range', type=str, default="0-20")        


opt = parser.parse_args()

bot = int(opt.range.split("-")[0])
up = int(opt.range.split("-")[1])

# print(bot)
# print(up)

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
        
    print(cmd)
    os.system(cmd)
