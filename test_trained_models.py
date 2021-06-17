#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 22:57:08 2020
Last assessed on Sat Dec 26 19:23:57 2020

@author: tibrayev

"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as dset
import torch.optim as optim
import numpy as np
import random
import sys
import time
import argparse
import copy
import json

torch.set_printoptions(linewidth = 160)
np.set_printoptions(linewidth = 160)
np.set_printoptions(precision=4)
np.set_printoptions(suppress='True')

from custom_models_cifar_vgg import vgg11
from torchvision.models import resnet50, resnet18
from utilities import get_data_loaders
from custom_normalization_functions import custom_3channel_img_normalization_with_per_image_params, custom_3channel_img_normalization_with_dataset_params

import matplotlib.pyplot as plt
from torchvision.utils import make_grid as grid

SEED = 1
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser(description='Run inference on given model parameters', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset',        default='imagenet2012', type=str,   help='Dataset name')
parser.add_argument('--model',          default='resnet18',     type=str,   help='Model architecture to be trained')
parser.add_argument('--batch_size',     default=256,            type=int,   help='Batch size for data loading')
parser.add_argument('--parallel',       default=True,           type=bool,  help='Flag to whether parallelize model over multiple GPUs')
parser.add_argument('--pretrained',     default=False,          type=bool,  help='Flag to load pretrained model')
parser.add_argument('--checkpoint',     default=None,           type=str,   help='Path to checkpoint file')
parser.add_argument('--log',            default=None,           type=str,   help='Path to the log file')

#%% Parse script parameters.
global args
args = parser.parse_args()

DATASET         = args.dataset
MODEL           = args.model
BATCH_SIZE      = args.batch_size
PARALLEL        = args.parallel
PRETRAINED      = args.pretrained
CKPT_DIR        = args.checkpoint

if PRETRAINED and CKPT_DIR is not None:
    raise ValueError("Received both pretrained=True and checkpoint dir! Please specify only one!")
elif not PRETRAINED and CKPT_DIR is None:
    raise ValueError("Received pretrained=False but no checkpoint dir! Please specify only one!")

# Log
if args.log == 'sys':
    f = sys.stdout
elif args.log is not None:
    f = open(args.log, 'a', buffering=1)
else:
    raise ValueError("Should specify log file")


# Timestamp
f.write('\n*******************************************************************\n')
f.write('==>> Run on: '+time.strftime("%Y-%m-%d %H:%M:%S")+'\n')
f.write('==>> Seed was set to: {}\n'.format(SEED))

# Device instantiation
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#%% Load the dataset.
if DATASET == 'CIFAR10':
    root_dir = './datasets/{}'.format(DATASET)
    if not os.path.exists(root_dir): os.makedirs(root_dir)
    normalization_func = custom_3channel_img_normalization_with_per_image_params(img_dimensions = [3, 32, 32], device = device)
elif DATASET == 'imagenet2012':
    root_dir = '/local/a/imagenet/imagenet2012'
    normalization_func = custom_3channel_img_normalization_with_dataset_params(mean=[0.485, 0.456, 0.406], 
                                                                               std=[0.229, 0.224, 0.225],
                                                                               img_dimensions = [3, 224, 224], device = device)
else:
    raise ValueError("Script supports only two datasets: CIFAR10 and imagenet2012")

train_loader, valid_loader, test_loader = get_data_loaders(DATASET, 
                                                           root_dir, 
                                                           BATCH_SIZE, 
                                                           augment=True, 
                                                           random_seed=SEED, 
                                                           valid_size=0.0, 
                                                           shuffle=True, 
                                                           num_workers=16, 
                                                           pin_memory=True)

f.write('==>> Dataset used: {}\n'.format(DATASET))
f.write('==>> Batch size: {}\n'.format(BATCH_SIZE))
f.write('==>> Total training batches: {}\n'.format(len(train_loader)))
f.write('==>> Total validation batches: {}\n'.format(len(valid_loader)))
f.write('==>> Total testing batches: {}\n'.format(len(test_loader)))


#%% #FIXME: Load the model.
if MODEL == 'vgg11':
    model = vgg11(num_classes=len(test_loader.dataset.classes))
    if PRETRAINED:
        model_init_sd = torch.load('./results/checkpoint_clean_model.pth', map_location='cpu')['model']
        model.load_state_dict(model_init_sd)
elif MODEL == 'resnet50':
    model = resnet50(pretrained=PRETRAINED, num_classes=len(test_loader.dataset.classes))
elif MODEL == 'resnet18':
    model = resnet18(pretrained=PRETRAINED, num_classes=len(test_loader.dataset.classes))
else:
    raise ValueError("Received unsupported model!")

model.to(device)

if PARALLEL:
    model = nn.DataParallel(model)

f.write("{}\n".format(model))

if PRETRAINED: 
    f.write("Pretrained model was loaded!\n")
elif CKPT_DIR is not None:
    ckpt = torch.load(CKPT_DIR, map_location=device)
    model.load_state_dict(ckpt['model'])
    f.write("Pretrained model was loaded from checkpoint: {}\n".format(CKPT_DIR))    


for param in model.parameters():
    param.requires_grad = False


#%% Post-prune Pre-Fine-Tuning model evaluation.
model.eval()
correct     = 0.0
ave_loss    = 0.0
total       = 0
with torch.no_grad():
    for batch_idx, (x_val, y_val) in enumerate(test_loader):
        x_val, y_val = x_val.to(device), y_val.to(device)
        x_norm = normalization_func(x_val)
        output = model(x_norm)
        loss   = F.cross_entropy(output, y_val)
        
        _, predictions   = torch.max(output.data, 1)
        total           += y_val.size(0)
        correct         += (predictions == y_val).sum().item()
        ave_loss        += loss.item()
        
f.write('==>>> MODEL EVAL ON TEST SET | val loss: {:.6f}, val acc: {:.4f}\n'.format(
                ave_loss*1.0/(batch_idx + 1), correct*1.0/total))
