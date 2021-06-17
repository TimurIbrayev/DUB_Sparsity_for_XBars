#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 22:48:30 2020

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
import copy
import argparse

torch.set_printoptions(linewidth = 160)
np.set_printoptions(linewidth = 160)
np.set_printoptions(precision=4)
np.set_printoptions(suppress='True')

from custom_models_cifar_vgg import vgg11
from torchvision.models import resnet18
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


#%% Parse script parameters.

DATASET         = 'CIFAR10'
BATCH_SIZE      = 128
PARALLEL        = False
PRETRAINED      = False
CKPT_DIR        = './results/checkpoint_clean_model.pth'

LOG = 'lbl_sensitivity'


PRUNE_RATIOS = [i*1.0 for i in range(0, 100+1, 5)]
PRUNE_LAYERS = [0, 1, 2, 3, 4, 5, 6, 7]

f = open('./results/{}/{}/log_layer_sensitivities.txt'.format(DATASET, LOG), 'a', buffering=1)
# f = sys.stdout
# Timestamp
f.write('\n*******************************************************************\n')
f.write('==>> Run on: '+time.strftime("%Y-%m-%d %H:%M:%S")+'\n')
f.write('==>> Seed was set to: {}\n'.format(SEED))

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
if DATASET == 'CIFAR10':
    model = vgg11(num_classes=len(test_loader.dataset.classes))
elif DATASET == 'imagenet2012':
    model = resnet18(pretrained=PRETRAINED, num_classes=len(test_loader.dataset.classes))
else:
    raise ValueError("Received unsupported model!")

model.to(device)

if PARALLEL:
    model = nn.DataParallel(model)

f.write("{}\n".format(model))
for param in model.parameters():
    param.requires_grad = False
    

#%% Updating model, optimizer, lr_scheduler, tracking variables, etc. if RESUME flag is specified...
if CKPT_DIR is not None:
    ckpt = torch.load(CKPT_DIR, map_location=device)
    MSD  = copy.deepcopy(ckpt['model'])
    model.load_state_dict(MSD)
    f.write("==>> Loaded model from checkpoint: {}\n".format(CKPT_DIR))
else:
    MSD  = copy.deepcopy(model.state_dict())

l_id = 0
prunable_params_cnt = 0.0
layerwise_prunable_params_cnt = []
for m in model.modules():
    if isinstance(m, nn.Conv2d):
        if l_id in PRUNE_LAYERS:
            prunable_params_cnt += m.weight.numel()
            layerwise_prunable_params_cnt.append(m.weight.numel())
        else:
            raise ValueError("Layer id is not found in prune layers!")
        l_id += 1

f.write("Total {:.0f} prunable parameters!\n".format(prunable_params_cnt))
f.write("Prunable layer-wise: {}\n".format(layerwise_prunable_params_cnt))

#%% Layer Sensitivity Analysis
f.write("\nStarting layer sensitivity analysis...\n")
pruning_sensitivity = []

for layer_id in PRUNE_LAYERS: # for every layer...
    f.write("\nPruning prunable layer with index {}:\n".format(layer_id))
    layer_sensitivity = []
    for prune_ratio in PRUNE_RATIOS: # for every prune ratio...
        # pre-load clean model weights
        model.load_state_dict(MSD)
        for param in model.parameters():
            param.requires_grad_(False)
        
        # search for prune layer
        l_id = 0
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                if l_id == layer_id:
                    # get all weights
                    weights = m.weight.abs().view(-1).clone().detach()
                    # find threshold based on target prune ratio
                    threshold = np.percentile(weights.cpu(), prune_ratio)
                    # prune based on this threshold
                    mask = (m.weight.abs() <= threshold)
                    m.weight.masked_fill_(mask, 0.0)
                    count_zeros = (m.weight == 0.0).sum()
                    # break the loop as this script does one layer at a time
                    break
                else:
                    l_id += 1
        
        # check what is the accuracy degradation with the target pruning ratio
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
        
        if (batch_idx+1) == len(test_loader):
            f.write('==>>> Prunable layer {} | pruned [{:.2f}]  | val loss: {:.6f}, val acc: {:.4f}\n'.format(
                    l_id, count_zeros*100.0/layerwise_prunable_params_cnt[l_id], 
                    ave_loss*1.0/(batch_idx + 1), correct*1.0/total))        
        
        layer_sensitivity.append(correct*100.0/total)
    pruning_sensitivity.append(layer_sensitivity)

torch.save({"pruning_sensitivity": pruning_sensitivity}, './results/{}/{}/log_layer_sensitivities.pth'.format(DATASET, LOG))
