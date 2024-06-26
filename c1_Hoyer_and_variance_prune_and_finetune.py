#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 22:57:08 2020
Last assessed on Sun Nov 21 22:31:12 2021

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
import math

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

parser = argparse.ArgumentParser(description='Run layer-by-layer pruning and perform fine-tuning', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset',        default='CIFAR10',      type=str,   help='Dataset name')
parser.add_argument('--model',          default='vgg11',        type=str,   help='Model architecture to be trained')
parser.add_argument('--pretrained',     default=False,          type=bool,  help='Flag to whether load pretrained model or not')
parser.add_argument('--checkpoint',     default=None,           type=str,   help='Path to checkpoint file')
parser.add_argument('--batch_size',     default=128,            type=int,   help='Batch size for data loading')
parser.add_argument('--parallel',       default=False,          type=bool,  help='Flag to whether parallelize model over multiple GPUs')
parser.add_argument('--valid_split',    default=0.000,          type=float, help='Fraction of training set dedicated for validation')
parser.add_argument('--log',            default=None,           type=str,   help='Path to the log file')
parser.add_argument('--w_tol',          default=1.0e-3,         type=float, help='Weight tolerance to consider prunable weights')
parser.add_argument('--tile_size',      default=64,             type=int,   help='Logical crossbar (tile) size')
parser.add_argument('--weight_quant',   default=1,              type=int,   help='The number of bits assumed to be used for weight quantization')
parser.add_argument('--prs', nargs='+', default=None,           type=float, help='Layer-by-layer pruning ratios to which network needs to be pruned')


#%% Parse script parameters.
global args
args = parser.parse_args()


from c1_Hoyer_and_variance_class import HoyerAndVariance

DATASET         = args.dataset
MODEL           = args.model
PRETRAINED      = args.pretrained
CKPT_DIR        = args.checkpoint
BATCH_SIZE      = args.batch_size
PARALLEL        = args.parallel
VALID_SPLIT     = args.valid_split

WEIGHT_TOL      = args.w_tol
TILE_SIZE       = args.tile_size
WEIGHT_QUANT    = args.weight_quant


if DATASET == 'CIFAR10':    
    FINE_TUNE = {'MAX_EPOCHS':      200,
                 'MOMENTUM':        0.0,
                 'WEIGHT_DECAY':    0.0,
                 'INIT_LR':         0.01,
                 'LR_SCHEDULE':     [50, 100, 150],
                 'LR_SCHEDULE_GAMMA': 0.1}
    
elif DATASET == 'imagenet2012':
    FINE_TUNE = {'MAX_EPOCHS':      120,
                 'MOMENTUM':        0.0,
                 'WEIGHT_DECAY':    0.0,
                 'INIT_LR':         0.01,
                 'LR_SCHEDULE':     [30, 60, 90],
                 'LR_SCHEDULE_GAMMA': 0.1}

if PRETRAINED:
    PRUNE_FINE_TUNE_DIR = ((args.log).split("log_")[0] + 'checkpoint_prune_lbl_' + (args.log).split("log_")[1]).split(".txt")[0] + ".pth"  
elif CKPT_DIR is not None:
    PRUNE_FINE_TUNE_DIR = CKPT_DIR.split("checkpoint_")[0] + 'checkpoint_prune_lbl_'+ CKPT_DIR.split("checkpoint_")[1]
else:
    raise ValueError("Expected either pretrained=True or ckpt_dir, but not BOTH!")

    
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
                                                           valid_size=VALID_SPLIT, 
                                                           shuffle=True, 
                                                           num_workers=16, 
                                                           pin_memory=True)

if VALID_SPLIT > 0.0:
    validation_loader = valid_loader
else:
    validation_loader = test_loader

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
    prunable_param_names = ['features.0.weight', 'features.3.weight', 'features.6.weight', 'features.8.weight', 
                            'features.11.weight', 'features.13.weight', 'features.16.weight', 'features.18.weight']
elif MODEL == 'resnet50':
    model = resnet50(pretrained=PRETRAINED, num_classes=len(test_loader.dataset.classes))
    prunable_param_names = [n for n, p in model.named_parameters() if ('conv' in n) or ('downsample.0' in n)]
elif MODEL == 'resnet18':
    model = resnet18(pretrained=PRETRAINED, num_classes=len(test_loader.dataset.classes))
    prunable_param_names = [n for n, p in model.named_parameters() if ('conv' in n) or ('downsample.0' in n)]    
else:
    raise ValueError("Received unsupported model!")

model.to(device)

if PARALLEL:
    model = nn.DataParallel(model)
    prune_params = [p for n, p in model.module.named_parameters() if p.requires_grad and n in prunable_param_names]
    rest_params =  [p for n, p in model.module.named_parameters() if p.requires_grad and n not in prunable_param_names]
else:
    prune_params = [p for n, p in model.named_parameters() if p.requires_grad and n in prunable_param_names]
    rest_params =  [p for n, p in model.named_parameters() if p.requires_grad and n not in prunable_param_names]


grad_requirement_dict = {name: param.requires_grad for name, param in model.named_parameters()}
f.write("{}\n".format(model))
f.write("Total prunable modules: {}\n".format(len(prune_params)))


if PRETRAINED: 
    f.write("Pretrained model was loaded!\n")
elif CKPT_DIR is not None:
    ckpt = torch.load(CKPT_DIR, map_location=device)
    model.load_state_dict(ckpt['model'])
    f.write("Pretrained model was loaded from checkpoint: {}\n".format(CKPT_DIR))    

#%% Setting up pruning and retraining parameters.
prune = HoyerAndVariance(model, device,
                         lambda_variance = 0.0,
                         lambda_mean = 0.0,
                         tol = WEIGHT_TOL,
                         tile_size = TILE_SIZE,
                         weight_quantization = WEIGHT_QUANT)

f.write("\n==>> {}\n\n".format(prune))


#%% Model evaluation on validation set.
model.eval()
correct     = 0.0
ave_loss    = 0.0
total       = 0
with torch.no_grad():
    for batch_idx, (x_val, y_val) in enumerate(validation_loader):
        x_val, y_val = x_val.to(device), y_val.to(device)
        x_norm = normalization_func(x_val)
        output = model(x_norm)
        loss   = F.cross_entropy(output, y_val)
        
        _, predictions   = torch.max(output.data, 1)
        total           += y_val.size(0)
        correct         += (predictions == y_val).sum().item()
        ave_loss        += loss.item()
        
f.write('==>>> MODEL EVAL ON VALIDATION SET | val loss: {:.6f}, val acc: {:.4f}\n'.format(
                ave_loss*1.0/(batch_idx + 1), correct*1.0/total))

#%% Model evaluation on test set.
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
                        
#%% Post-Training Pruning and Fine-Tuning with masks on.
"""
This part of the script performs assessment of tile sparsity, 
which is defined as the sparsity of the least sparse column
within the tile,
for each tile in the network trained with variable lambda L1 regularization 
AND
determines the masks for weights which are then used for fine-tuning.
This is performed in three stages.

***First, tile sparsity is estimated for each tile by considering almost zero
weights for pruning based on user defined weight tolerance (prune.tol),
but without actually performing pruning of almost zero weights.



***Second, for each tile based on its tile sparsity, 
i.e. based on the least sparse column in each tile, 
the tile will be either

(1) forced to have the same sparsity on all columns within the tile
which will be equal to sparsity of higher tile sparsity by 
pruning BOTH almost zero weights AND weights above weight tolerance (prune.tol) 
starting from smallest until it reaches next (higher) tile sparsity level.

or

(2) loosened to have the same sparsity on all columns within the tile 
which will be equal to sparsity of lower tile sparsity by 
pruning almost zero weights starting from smallest 
EXCEPT some portion of large (in magnitude) weights below weight tolerance (prune.tol)
until it reaches previous (lower) tile sparsity level.
Note: the latter case might require not pruning tile at all, 
if the lower tile sparsity is the lowest tile sparsity level (no pruning at all).

For example, if tile sparsity (i.e. the least sparse column) has sparsity of 
41/64 = 0.64, which makes it closer to the higher tile sparsity level (48/64=0.75),
the pruning will be performed on weights (within that tile) which are almost zeros
and additional weights which will make tile to have ALL columns to have sparsity of 0.75.

If tile sparsity has sparsity of 39/64 = 0.61, 
which makes it closer to the lower tile sparsity level (32/64=0.5), 
the pruning will be performed on weights (within that tile) which are almost zeros 
starting from lowest ones but until the tile has ALL columns with sparsity of 0.5.

By enforcing this rule, we allow max utilization of tile based on its tile sparsity.
If the least sparse column cannot be pruned at all due to importance of weights,
then there is no reason for other neighboring columns in the same tile to be pruned,
because they will not result in ADC requirement reduction.
Contrary, if the least sparse column can be pruned to reach higher tile sparsity,
then there is no reason for other neighboring columns in the same tile to be retained.



***Third, the fine-tuning is performed by retraining weights remaining after second stage
without any regularization or constraints on weight magnitude.
"""

count_almost_zeros, count_almost_zeros_layerwise = prune.count_almost_zeros(model)
f.write("==>> Total almost zero weights: {:.0f}/{} [{:.2f}]\n".format(count_almost_zeros, prune.total_weights, count_almost_zeros*100.0/prune.total_weights))
f.write("==>> Total almost zeroes layerewise:\n")
for l_id, (cnt_almost_zeroes, cnt_params) in enumerate(zip(count_almost_zeros_layerwise, prune.layerwise_weights)):
    f.write("Prunable layer {}:\t {:.0f}/{:.0f} [{:.2f}]\n".format(
        l_id, cnt_almost_zeroes, cnt_params, cnt_almost_zeroes*100.0/cnt_params))

tile_sparsity_hist = prune.hist_tile_sparsity_almost_zeros(model)
f.write("==>> For tile size of {} and ADC resolution of {} bits,\n"\
        "the following is the tile sparsity histogram,\n"\
        "based on ALMOST ZERO weights and BEFORE pruning:\n".format(
            prune.tile_size, prune.ADC_res_bits))
vals, bins = tile_sparsity_hist
for v, b in zip(vals, bins):
    f.write("{:.3f}:\t{}\n".format(b, v))





if args.prs is not None:
    assert len(args.prs) == len(prune_params), "Prune ratios are specified, but not for all layers!"
    f.write("==>> Pruning fixed prune ratios based on tile sparsity: {}\n".format(args.prs))
    count_zeros, masks = prune.prune_fixed_prune_ratios_based_on_tile_sparsity(model, 
                                                                               target_prune_ratios=args.prs)
else:
    count_zeros, masks = prune.prune_based_on_tile_sparsity(model)
f.write("==>> Total pruned weights: {:.0f}/{} [{:.2f}]\n".format(count_zeros, prune.total_weights, count_zeros*100.0/prune.total_weights))

count_zeros, count_zeros_layerwise = prune.count_zeros(model)
f.write("==>> Total zeroes layerwise:\n")
for l_id, (cnt_zeroes, cnt_params) in enumerate(zip(count_zeros_layerwise, prune.layerwise_weights)):
    f.write("Prunable layer {}:\t {:.0f}/{:.0f} [{:.2f}]\n".format(
        l_id, cnt_zeroes, cnt_params, cnt_zeroes*100.0/cnt_params))

pruned_tile_sparsity_hist = prune.hist_tile_sparsity(model)
f.write("==>> For tile size of {} and ADC resolution of {} bits,\n"\
        "the following is the tile sparsity histogram,\n"\
        "based on PRUNED weights (= 0.0) AFTER pruning:\n".format(
            prune.tile_size, prune.ADC_res_bits))
vals, bins = pruned_tile_sparsity_hist
for v, b in zip(vals, bins):
    f.write("{:.3f}:\t{}\n".format(b, v))


#%% Post-prune Pre-Fine-Tuning model evaluation.
model.eval()
correct     = 0.0
ave_loss    = 0.0
total       = 0
with torch.no_grad():
    for batch_idx, (x_val, y_val) in enumerate(validation_loader):
        x_val, y_val = x_val.to(device), y_val.to(device)
        x_norm = normalization_func(x_val)
        output = model(x_norm)
        loss   = F.cross_entropy(output, y_val)
        
        _, predictions   = torch.max(output.data, 1)
        total           += y_val.size(0)
        correct         += (predictions == y_val).sum().item()
        ave_loss        += loss.item()
        
f.write('==>>> POST-PRUNE PRE-FINE-TUNE MODEL EVAL ON VALIDATION SET | val loss: {:.6f}, val acc: {:.4f}\n'.format(
                ave_loss*1.0/(batch_idx + 1), correct*1.0/total))

#%% Fine-Tuning with no regularization.
criterion = nn.CrossEntropyLoss()
num_epochs = FINE_TUNE['MAX_EPOCHS']

if PARALLEL:
    if MODEL == 'vgg11':
        conv_params = [p for n, p in model.module.named_parameters() if p.requires_grad and 'features' in n]
    elif MODEL == 'resnet50' or MODEL == 'resnet18':
        conv_params = [p for n, p in model.module.named_parameters() if p.requires_grad and n in prunable_param_names]
else:
    if MODEL == 'vgg11':
        conv_params = [p for n, p in model.named_parameters() if p.requires_grad and 'features' in n]
    elif MODEL == 'resnet50' or MODEL == 'resnet18':
        conv_params = [p for n, p in model.named_parameters() if p.requires_grad and n in prunable_param_names]
    
optimizer = optim.SGD(conv_params, lr=FINE_TUNE['INIT_LR'], momentum=FINE_TUNE['MOMENTUM'], weight_decay=FINE_TUNE['WEIGHT_DECAY'])

if VALID_SPLIT > 0.0:
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=4, factor=FINE_TUNE['LR_SCHEDULE_GAMMA'], verbose=True)
else:
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                  milestones=FINE_TUNE['LR_SCHEDULE'],
                                                  gamma = FINE_TUNE['LR_SCHEDULE_GAMMA'])
    
f.write("==>> FINE_TUNE Optimizer settings: {}\n".format(optimizer))
f.write("==>> FINE_TUNE LR scheduler type: {}\n".format(lr_scheduler.__class__))
f.write("==>> FINE_TUNE LR scheduler state: {}\n".format(lr_scheduler.state_dict()))
f.write("==>> FINE_TUNE Number of training epochs: {}\n".format(num_epochs))

train_loss          = []
train_acc           = []
valid_loss          = []
valid_acc           = []
best_val_acc        = 0.0

for epoch in range(num_epochs):
    # Train for one epoch
    model.train()
    correct         = 0.0
    ave_loss        = 0.0
    total           = 0
    for batch_idx, (x_train, y_train) in enumerate(train_loader):
        x_train, y_train = x_train.to(device), y_train.to(device)
    
        optimizer.zero_grad()
        model.zero_grad()
        x_norm = normalization_func(x_train)
        output = model(x_norm)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()
        
        # zeroing-out pruned weights (This step is essential, if optimizer has momentum)
        # Momentum will have update factor regardless zero gradients
        prune.zero_out_weights(model, masks)
        
        count_zeros, _ = prune.count_zeros(model)
    
        _, predictions   = torch.max(output.data, 1)
        total           += y_train.size(0)
        correct         += (predictions == y_train).sum().item()
        ave_loss        += loss.item()
    
        if (batch_idx+1)%1000 == 0 or (batch_idx+1) == len(train_loader):
                f.write('==>>> FINE-TUNE | fine-tune epoch: {}, loss: {:.6f}, acc: {:.4f}, zeros: {}/{}\n'.format(
                        epoch, ave_loss*1.0/(batch_idx + 1), correct*1.0/total, count_zeros, prune.total_weights))
    train_loss.append(ave_loss*1.0/(batch_idx + 1))
    train_acc.append(correct*100.0/total)
    
    # Evaluate on the clean val set
    model.eval()
    correct     = 0.0
    ave_loss    = 0.0
    total       = 0
    with torch.no_grad():
        for batch_idx, (x_val, y_val) in enumerate(validation_loader):
            x_val, y_val = x_val.to(device), y_val.to(device)
            x_norm = normalization_func(x_val)
            output = model(x_norm)
            loss   = criterion(output, y_val)
            
            _, predictions   = torch.max(output.data, 1)
            total           += y_val.size(0)
            correct         += (predictions == y_val).sum().item()
            ave_loss        += loss.item()
            
            if (batch_idx+1) == len(validation_loader):
                f.write('==>>> CLEAN VALIDATE | epoch: {}, batch index: {}, val loss: {:.6f}, val acc: {:.4f}\n'.format(
                        epoch, batch_idx+1, ave_loss*1.0/(batch_idx + 1), correct*1.0/total))
        valid_loss.append(ave_loss*1.0/(batch_idx+1))
        valid_acc.append(correct*100.0/total)

    if VALID_SPLIT > 0.0:
        lr_scheduler.step(ave_loss*1.0/(batch_idx+1))
    else:
        lr_scheduler.step()
        
    if (correct*100.0/total) >= best_val_acc:
        best_val_acc = correct*100.0/total
        best_state_dict = copy.deepcopy(model.state_dict())
        
        torch.save({'SEED': SEED,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'num_epochs': num_epochs,
                    'train_loss': train_loss,
                    'valid_loss': valid_loss,
                    'train_acc': train_acc,
                    'valid_acc': valid_acc,
                    'tile_sparsity_hist': tile_sparsity_hist,
                    'pruned_tile_sparsity_hist': pruned_tile_sparsity_hist,
                    }, PRUNE_FINE_TUNE_DIR)
        
f.write("Best val accuracy during fine-tuning: {:.2f}\n".format(best_val_acc))


#%% Model evaluation on test set.
model.load_state_dict(best_state_dict)
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
