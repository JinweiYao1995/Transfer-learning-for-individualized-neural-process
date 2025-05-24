# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 20:27:14 2025

@author: yaojinwei
"""



#Loads of the prediction RMSE for the proposed method 
import torch 

imtp66 = []
for check_num in range(1,81):
    checkpoint = torch.load(f'experiments//runs_imtp//imtp//checkpoint{check_num}//full_0606.pth')
    imtp66.append(checkpoint['valid_errors']['target2'].item())
    
    
imtp88 = []
for check_num in range(1,81):
    checkpoint = torch.load(f'experiments//runs_imtp//imtp//checkpoint{check_num}//full_0808.pth')
    imtp88.append(checkpoint['valid_errors']['target2'].item())
    
    
imtp1010 = []
for check_num in range(1,81):
    checkpoint = torch.load(f'experiments//runs_imtp//imtp//checkpoint{check_num}//full_1010.pth')
    imtp1010.append(checkpoint['valid_errors']['target2'].item())
        

    
    
#Loads of the prediction RMSE for the proposed w/o sources
imtps66 = []
for check_num in range(1,81):
    checkpoint = torch.load(f'experiments//runs_imtp_single//imtps//checkpoint{check_num}//full_0606.pth')
    imtps66.append(checkpoint['valid_errors']['target2'].item())
    
imtps88 = []
for check_num in range(1,81):
    checkpoint = torch.load(f'experiments//runs_imtp_single//imtps//checkpoint{check_num}//full_0808.pth')
    imtps88.append(checkpoint['valid_errors']['target2'].item())
    
imtps1010 = []
for check_num in range(1,81):
    checkpoint = torch.load(f'experiments//runs_imtp_single//imtps//checkpoint{check_num}//full_1010.pth')
    imtps1010.append(checkpoint['valid_errors']['target2'].item())
    



#Load the RMSE for MGP-based transfer learning 


import pyreadr
MGP6 = pyreadr.read_r('down_original.RData')['RMSE6'].squeeze().tolist()
MGP8 = pyreadr.read_r('down_original.RData')['RMSE8'].squeeze().tolist()
MGP10 = pyreadr.read_r('down_original.RData')['RMSE10'].squeeze().tolist()



#Testing for single-task ANP 

import os
import argparse
import yaml
from easydict import EasyDict

import torch
from dataset import load_data
from dataset.utils import to_device
from model import get_model
from train.trainer import evaluate_test
import matplotlib.pyplot as plt


# ENVIRONMENTAL SETTINGS
# to prevent over-threading
torch.set_num_threads(1)

DATASETS = ['synthetic']
CHECKPOINTS = ['best_nll', 'best_error', 'last']

# arguments
parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='', choices=DATASETS)
parser.add_argument('--eval_root', type=str, default='experiments')
parser.add_argument('--eval_dir', type=str, default='')
parser.add_argument('--eval_name', type=str, default='')
parser.add_argument('--eval_ckpt', type=str, default='best_error', choices=CHECKPOINTS)
parser.add_argument('--device', type=str, default='0')
parser.add_argument('--reset', default=False, action='store_true')
parser.add_argument('--verbose', '-v', default=False, action='store_true')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--global_batch_size', type=int, default=4)

args = parser.parse_args()


# load test config

exp_name = 'stp'

with open(os.path.join('configs', args.data, 'config_single_test.yaml')) as f:  #just need to change this name for iteration
    config_test = EasyDict(yaml.safe_load(f))
config_test.seed = args.seed


args.eval_dir = config_test.eval_dir

if args.eval_dir != '':
    config_test.eval_dir = args.eval_dir

# set device and evaluation directory
os.environ['CUDA_VISIBLE_DEVICES'] = args.device
device = torch.device('cuda')
config_test.eval_dir = os.path.join(args.eval_root, config_test.eval_dir)


if args.eval_name == '':
    eval_list = os.listdir(config_test.eval_dir)
else:
    eval_list = [args.eval_name]
    
      
for context in [6,8,10]:
        # test different context points settings 
        config_test.context_size['target1'] = context
        checknum = 1
        eval_path = os.path.join(config_test.eval_dir, exp_name, f'checkpoint{checknum}', f'{args.eval_ckpt}.pth')
        # last_path = os.path.join(config_test.eval_dir, exp_name, 'checkpoints', 'last.pth')
        last_path = eval_path
        if not (os.path.exists(eval_path) and os.path.exists(last_path)):
                if args.verbose:
                        print(f'checkpoint of {exp_name} does not exist or still running - skip...')
                continue
        # load trained single NP model and testing configuration 
        ckpt = torch.load(eval_path, map_location=device)
        config = ckpt['config']
        params = ckpt['model']
    
        model = get_model(config, device)
        model.load_state_dict_(params)
    
        nlls, errors = evaluate_test(model, device, config_test) #this need to be rewrite.
        if exp_name == 'stp':
            if context == 6:
                cs6 = [value['target1'].item() for key, value in errors.items()]
            if context == 8:
                cs8 = [value['target1'].item() for key, value in errors.items()]
            if context == 10:
                cs10 = [value['target1'].item() for key, value in errors.items()]       



import os
import argparse
import yaml
from easydict import EasyDict

import torch
from dataset import load_data
from dataset.utils import to_device
from model import get_model
from train.trainer import evaluate_test
import matplotlib.pyplot as plt


# ENVIRONMENTAL SETTINGS
# to prevent over-threading
torch.set_num_threads(1)

DATASETS = ['synthetic']
CHECKPOINTS = ['best_nll', 'best_error', 'last']

# arguments
parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='', choices=DATASETS)
parser.add_argument('--eval_root', type=str, default='experiments')
parser.add_argument('--eval_dir', type=str, default='')
parser.add_argument('--eval_name', type=str, default='')
parser.add_argument('--eval_ckpt', type=str, default='best_error', choices=CHECKPOINTS)
parser.add_argument('--device', type=str, default='0')
parser.add_argument('--reset', default=False, action='store_true')
parser.add_argument('--verbose', '-v', default=False, action='store_true')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--global_batch_size', type=int, default=4)

args = parser.parse_args()


# load test config
exp_name = 'mtp'

with open(os.path.join('configs', args.data, 'config_mtp_test.yaml')) as f:  #just need to change this name for iteration
    config_test = EasyDict(yaml.safe_load(f))
config_test.seed = args.seed


args.eval_dir = config_test.eval_dir

if args.eval_dir != '':
    config_test.eval_dir = args.eval_dir

# set device and evaluation directory
os.environ['CUDA_VISIBLE_DEVICES'] = args.device
device = torch.device('cuda')
config_test.eval_dir = os.path.join(args.eval_root, config_test.eval_dir)


if args.eval_name == '':
    eval_list = os.listdir(config_test.eval_dir)
else:
    eval_list = [args.eval_name]
    
      
for context in [6,8,10]:
    # test models in eval_list   
        config_test.context_size['target1'] = context
        checknum = 1
        eval_path = os.path.join(config_test.eval_dir, exp_name, f'checkpoint{checknum}', f'{args.eval_ckpt}.pth')
        last_path = eval_path
        if not (os.path.exists(eval_path) and os.path.exists(last_path)):
                if args.verbose:
                        print(f'checkpoint of {exp_name} does not exist or still running - skip...')
                continue
        # load trained MTNP and testing configuration 
        ckpt = torch.load(eval_path, map_location=device)
        config = ckpt['config']
        params = ckpt['model']
    
        model = get_model(config, device)
        model.load_state_dict_(params)
    
        nlls, errors = evaluate_test(model, device, config_test) #this need to be rewrite.
        if exp_name == 'mtp':
            if context == 6:
                c6 = [value['target1'].item() for key, value in errors.items()]
            if context == 8:
                c8 = [value['target1'].item() for key, value in errors.items()]
            if context == 10:
                c10 = [value['target1'].item() for key, value in errors.items()]      







#Visualization of Figure 5.5 

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Group the data
data = [cs6, c6,imtp66,imtps66,MGP6, cs8, c8,imtp88,imtps88, MGP8, cs10, c10, imtp1010, imtps1010,MGP10]


# Define colors for each method category
colors = ['lightblue']  + ['lightgreen']  + ['lightcoral'] + ['lightsalmon'] + ['plum']+['lightblue']  + ['lightgreen']  + ['lightcoral'] + ['lightsalmon'] +['plum']+['lightblue']  + ['lightgreen']  + ['lightcoral'] + ['lightsalmon'] + ['plum']

positions = [0.9, 1.5, 2.1, 2.7, 3.3,  5.6, 6.2, 6.8,7.4,8.0,  10.3, 10.9,11.5,12.1, 12.7]

plt.figure(figsize=(10, 6))

# Set Times New Roman and font size for everything globally
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 18

# Create the boxplot with custom positions and no x-axis labels
box = plt.boxplot(data, positions=positions, patch_artist=True)

# Apply colors to each box
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)

# Set the median lines to black
for median in box['medians']:
    median.set(color='black', linewidth=2)

plt.ylim(0, 1.6)

# Remove x-axis ticks and labels, keep the axis line
plt.xticks([])
plt.tick_params(axis='x', which='both', bottom=False, top=False)
plt.tick_params(axis='both', which='major', labelsize=18)  # optional fine-tuning

# Remove all spines (frame around the plot)
for spine in plt.gca().spines.values():
    spine.set_visible(False)

# Legend with colored patches
legend_handles = [
    mpatches.Patch(color=colors[0], label='ANP'),
    mpatches.Patch(color=colors[3], label='Proposed w/o source'),
    mpatches.Patch(color=colors[1], label='MTNP'),
    mpatches.Patch(color=colors[4], label='MGP'),
    mpatches.Patch(color=colors[2], label='Proposed')  
]

plt.legend(
    handles=legend_handles,
    loc='lower center',
    bbox_to_anchor=(0.5, 1),
    ncol=3,
    frameon=False,
    fontsize=18
)

# Adjust spacing to make room for the legend
plt.subplots_adjust(top=0.85)

plt.show()













