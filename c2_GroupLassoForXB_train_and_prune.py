#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 19:28:50 2021

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

parser = argparse.ArgumentParser(description='Run GroupLasso for Crossbar (XB) training and perform pruning', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset',        default='CIFAR10',      type=str,   help='Dataset name')
parser.add_argument('--model',          default='vgg11',        type=str,   help='Model architecture to be trained')
parser.add_argument('--pretrained',     default=False,          type=bool,  help='Flag to whether load pretrained model or not')
parser.add_argument('--checkpoint',     default=None,           type=str,   help='Path to checkpoint file')
parser.add_argument('--batch_size',     default=128,            type=int,   help='Batch size for data loading')
parser.add_argument('--parallel',       default=False,          type=bool,  help='Flag to whether parallelize model over multiple GPUs')
parser.add_argument('--valid_split',    default=0.000,          type=float, help='Fraction of training set dedicated for validation')
#parser.add_argument('--cfg_dir',        default=None,           type=str,   help='Path to configuration file')
parser.add_argument('--w_tol',          default=1.0e-3,         type=float, help='Weight tolerance to consider prunable weights')
parser.add_argument('--lambda_g',       default=0.0005,         type=float, help='GroupLasso lambda factor')
parser.add_argument('--lambda_l2',      default=0.0005,         type=float, help='L2 Regularization lambda factor')
parser.add_argument('--save_dir',       default='xb_tile',      type=str,   help='Name of the subfolder to store logs and checkpoints')
parser.add_argument('--tile_size',      default=64,             type=int,   help='Logical crossbar (tile) size')
parser.add_argument('--weight_quant',   default=1,              type=int,   help='The number of bits assumed to be used for weight quantization')
#training arguments
parser.add_argument('--lr',             default=0.1,            type=float, help='Initial learning rate during training')
#pruning arguments
parser.add_argument('--pr_structure',   defeault='xb_tile',     type=str,   help='Type of structure to assume as one group', choices=['xb_tile', 'xb_row', 'xb_column'])
parser.add_argument('--prs', nargs='+', default=None,           type=float, help='Layer-by-layer pruning ratios to which network needs to be pruned')


#%% Parse script parameters.
global args
args = parser.parse_args()


from c2_GroupLassoForXB_class import GroupLassoForXB

DATASET         = args.dataset
MODEL           = args.model
PRETRAINED      = args.pretrained
CKPT_DIR        = args.checkpoint
BATCH_SIZE      = args.batch_size
PARALLEL        = args.parallel
VALID_SPLIT     = args.valid_split

WEIGHT_TOL      = args.w_tol
LAMBDA_G        = args.lambda_g
LAMBDA_L2REG    = args.lambda_l2
PR_STRUCTURE    = args.pr_structure
LOG             = 'GroupLassoFor_{}/'.format(PR_STRUCTURE) + args.save_dir

VERSION         = 'GroupLassoFor_{}_lambdas_l2reg{}_lambda_g{}_tol{}'.format(PR_STRUCTURE, LAMBDA_L2REG, LAMBDA_G, WEIGHT_TOL)


if DATASET == 'CIFAR10':
    TRAIN = {'MAX_EPOCHS':      160,
             'MOMENTUM':        0.9,
             'WEIGHT_DECAY_PRUNED_PARAMS':      LAMBDA_L2REG,
             'WEIGHT_DECAY_NOTPRUNED_PARAMS':   LAMBDA_L2REG,
             'INIT_LR':         args.lr,
             'LR_SCHEDULE':     [50, 100, 150],
             'LR_SCHEDULE_GAMMA': 0.1}
    
    FINE_TUNE = {'MAX_EPOCHS':      200,
                 'MOMENTUM':        0.0,
                 'WEIGHT_DECAY':    0.0,
                 'INIT_LR':         0.01,
                 'LR_SCHEDULE':     [50, 100, 150],
                 'LR_SCHEDULE_GAMMA': 0.1}
    
elif DATASET == 'imagenet2012':
    TRAIN = {'MAX_EPOCHS':      120,
             'MOMENTUM':        0.9,
             'WEIGHT_DECAY_PRUNED_PARAMS':      LAMBDA_L2REG,
             'WEIGHT_DECAY_NOTPRUNED_PARAMS':   LAMBDA_L2REG,
             'INIT_LR':         args.lr,
             'LR_SCHEDULE':     [30, 60, 90],
             'LR_SCHEDULE_GAMMA': 0.1}
    
    FINE_TUNE = {'MAX_EPOCHS':      120,
                 'MOMENTUM':        0.0,
                 'WEIGHT_DECAY':    0.0,
                 'INIT_LR':         0.01,
                 'LR_SCHEDULE':     [30, 60, 90],
                 'LR_SCHEDULE_GAMMA': 0.1}


if not os.path.exists('./results/{}/{}'.format(DATASET, LOG)): os.makedirs('./results/{}/{}'.format(DATASET, LOG))
SAVE_DIR        = './results/{}/{}/checkpoint_model_{}.pth'.format(DATASET, LOG, VERSION)

PRUNE_FINE_TUNE_DIR = './results/{}/{}/checkpoint_model_{}_pruned_finetuned.pth'.format(DATASET, LOG, VERSION)

f = open('./results/{}/{}/log_model_{}.txt'.format(DATASET, LOG, VERSION), 'a', buffering=1)
# f = sys.stdout

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
if PRETRAINED: f.write("Pretrained model was loaded!\n")
f.write("Total prunable modules: {}\n".format(len(prune_params)))


criterion = nn.CrossEntropyLoss()
num_epochs = TRAIN['MAX_EPOCHS']

optimizer = optim.SGD([
    {'params': prune_params, 'lr': TRAIN['INIT_LR'], 'momentum': TRAIN['MOMENTUM'], 'weight_decay': TRAIN['WEIGHT_DECAY_PRUNED_PARAMS']},
    {'params': rest_params,  'lr': TRAIN['INIT_LR'], 'momentum': TRAIN['MOMENTUM'], 'weight_decay': TRAIN['WEIGHT_DECAY_NOTPRUNED_PARAMS']},
                        ])

# Learning rate scheduler
if VALID_SPLIT > 0.0:
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=TRAIN['LR_SCHEDULE_GAMMA'], verbose=True)
else:
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=TRAIN['LR_SCHEDULE'], gamma = TRAIN['LR_SCHEDULE_GAMMA'])


#%% Setting up pruning and retraining parameters.
prune = GroupLassoForXB(model, device,
                        pruning_structure = PR_STRUCTURE,
                        lambda_g = LAMBDA_G, 
                        tol = WEIGHT_TOL,
                        tile_size = args.tile_size,
                        weight_quantization = args.weight_quant)

f.write("\n==>> {}\n\n".format(prune))

#%% Updating model, optimizer, lr_scheduler, tracking variables, etc. if RESUME flag is specified...
if CKPT_DIR is not None:
    ckpt = torch.load(CKPT_DIR, map_location=device)
    f.write("==>> Resuming training from loaded checkpoint from: {}\n".format(CKPT_DIR))    
    model.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optimizer'])
    lr_scheduler.load_state_dict(ckpt['lr_scheduler'])
    start_epoch             = ckpt['epoch'] + 1
    best_val_acc            = ckpt['best_val_acc']
    best_val_loss           = ckpt['best_val_loss']
    best_epoch              = ckpt['best_epoch']
    best_msdict             = ckpt['best_msdict']
    train_loss              = ckpt['train_loss']
    train_loss_cls          = ckpt['train_loss_cls']
    train_loss_groups       = ckpt['train_loss_groups']
    train_acc               = ckpt['train_acc']
    train_almost_zeros      = ckpt['train_almost_zeros']
    valid_loss              = ckpt['valid_loss']
    valid_acc               = ckpt['valid_acc']

else:
    f.write("==>> Starting training from scratch!\n")
    start_epoch             = 0
    best_val_acc            = 0.0
    best_val_loss           = float('inf')
    train_loss              = []
    train_loss_cls          = []
    train_loss_groups       = []
    train_acc               = []
    train_almost_zeros      = []
    valid_loss              = []
    valid_acc               = []


f.write("==>> Optimizer settings: {}\n".format(optimizer))
f.write("==>> LR scheduler type: {}\n".format(lr_scheduler.__class__))
f.write("==>> LR scheduler state: {}\n".format(lr_scheduler.state_dict()))
f.write("==>> Number of training epochs: {}\n".format(num_epochs))

#%% TRAIN-PRUNE.
for epoch in range(start_epoch, num_epochs):
    # Train for one epoch
    model.train()
    correct             = 0.0
    ave_loss            = 0.0
    ave_loss_cls        = 0.0
    ave_loss_groups     = 0.0
    total               = 0
    for batch_idx, (x_train, y_train) in enumerate(train_loader):
        x_train, y_train = x_train.to(device), y_train.to(device)
    
        optimizer.zero_grad()
        x_norm = normalization_func(x_train)
        output = model(x_norm)
        loss_cls = criterion(output, y_train)

        loss_groups = prune.compute_loss(model)

        loss = loss_cls + loss_groups
        loss.backward()
        optimizer.step()
        
        count_almost_zeros, _ = prune.count_almost_zeros(model)
    
        _, predictions      = torch.max(output.data, 1)
        total               += y_train.size(0)
        correct             += (predictions == y_train).sum().item()
        ave_loss            += loss.item()
        ave_loss_cls        += loss_cls.item()
        ave_loss_groups     += loss_groups.item()
    
        if (batch_idx+1)%1000 == 0 or (batch_idx+1) == len(train_loader):
                f.write('==>>> TRAIN-PRUNE | train epoch: {}, loss: {:.6f}, acc: {:.4f}, almost zeros: {}/{}\n'.format(
                        epoch, ave_loss*1.0/(batch_idx + 1), correct*1.0/total, count_almost_zeros, prune.total_weights))
                f.write('==>>> cls loss: {:.6f}\t grouplasso loss: {:.6f}\t\n'.format(
                        ave_loss_cls*1.0/(batch_idx + 1), ave_loss_groups*1.0/(batch_idx + 1)))
    train_loss.append(ave_loss*1.0/(batch_idx + 1))
    train_loss_cls.append(ave_loss_cls*1.0/(batch_idx + 1))
    train_loss_groups.append(ave_loss_groups*1.0/(batch_idx + 1))
    train_acc.append(correct*100.0/total)
    train_almost_zeros.append(count_almost_zeros)




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

    # Adjust learning rate
    if VALID_SPLIT > 0.0:
        lr_scheduler.step(ave_loss*1.0/(batch_idx+1))
    else:
        lr_scheduler.step()

    if (correct*100.0/total) >= best_val_acc:
        best_val_acc = correct*100.0/total
        best_epoch   = copy.deepcopy(epoch)
        best_msdict  = copy.deepcopy(model.state_dict())
        best_val_loss = ave_loss*1.0/(batch_idx+1)
        
    torch.save({'SEED': SEED,
                'model': model.state_dict(),
                'best_msdict': best_msdict,
                'best_epoch': best_epoch,
                'best_val_acc': best_val_acc,
                'best_val_loss': best_val_loss,
                'grad_requirement_dict': grad_requirement_dict,
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'num_epochs': num_epochs,
                'train_loss': train_loss,
                'train_loss_cls': train_loss_cls,
                'train_loss_groups': train_loss_groups,
                'train_acc': train_acc,
                'train_almost_zeros': train_almost_zeros,
                'valid_loss': valid_loss,
                'valid_acc': valid_acc}, SAVE_DIR)

f.write("Best val accuracy during training: {:.2f}\n".format(best_val_acc))
                        
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
if SAVE_DIR:
    save = torch.load(SAVE_DIR, map_location=device)
    model.load_state_dict(save['best_msdict'])

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