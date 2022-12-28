import os
import glob
from info import classes_dict

nic_dict = {}
root_dir = r'/data/klin/ClassC/nic/*/*/*.txt'
for log_name in glob.glob(root_dir):
    with open(log_name, 'r') as f:
        content = f.readlines()
        for line in content:
            if 'mean of bpp' in line:
                bpp = line.strip('\n').split(':')[-1]
            elif 'mean of psnr' in line:
                psnr = line.strip('\n').split(':')[-1]
            
        seq_name = log_name.split('/')[6]
        level = int(log_name.split('/')[5])
        nic_dict[(seq_name, level)] = (float(bpp), float(psnr))

for level in range(8):
    print(level, ': [', end = '')
    for seq_name in classes_dict['ClassC']['sequence_name']:
        print('%.8f' % nic_dict[(seq_name, level)][0], end = ', ')
    print(']')
