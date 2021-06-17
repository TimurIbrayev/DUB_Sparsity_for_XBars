#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 22:57:08 2020

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
from torchvision.models import resnet50
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

parser = argparse.ArgumentParser(description='Run training on models for CIFAR10 and ImageNet', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset',        default='imagenet2012', type=str,   help='Dataset name')
parser.add_argument('--model',          default='resnet50',     type=str,   help='Model architecture to be trained')
parser.add_argument('--batch_size',     default=256,            type=int,   help='Batch size for data loading')
parser.add_argument('--parallel',       default=False,          type=bool,  help='Flag to whether parallelize model over multiple GPUs')
parser.add_argument('--valid_split',    default=0.025,          type=float, help='Fraction of training set dedicated for validation')
parser.add_argument('--resume',         default=False,          type=bool,  help='Flag whether to resume training from checkpoint')
parser.add_argument('--checkpoint',     default=None,           type=str,   help='Path to checkpoint file')

#%% Parse script parameters.
global args
args = parser.parse_args()

DATASET         = args.dataset
MODEL           = args.model
BATCH_SIZE      = args.batch_size
PARALLEL        = args.parallel
RESUME          = args.resume
CKPT_DIR        = args.checkpoint
VALID_SPLIT     = args.valid_split

LOG             = 'clean_train'
VERSION         = 'clean'
TRAIN           = {
                		"MAX_EPOCHS":      120,
                    "MOMENTUM":        0.9,
                    "WEIGHT_DECAY":    0.0001,
                    "INIT_LR":         0.1,
                    "LR_SCHEDULE":     [50, 100, 150],
                    "LR_SCHEDULE_GAMMA": 0.1
                    }

if not os.path.exists('./results/{}/{}'.format(DATASET, LOG)): os.makedirs('./results/{}/{}'.format(DATASET, LOG))
SAVE_DIR        = './results/{}/{}/checkpoint_model_{}.pth'.format(
                            DATASET, LOG, VERSION)

f = open('./results/{}/{}/log_model_{}.txt'.format(
        DATASET, LOG, VERSION),
          'a', buffering=1)
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
                                                           num_workers=1, 
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
elif MODEL == 'resnet50':
    model = resnet50(pretrained=False, num_classes=len(test_loader.dataset.classes))
else:
    raise ValueError("Received unsupported model!")

model.to(device)
if PARALLEL:
    model = nn.DataParallel(model)

grad_requirement_dict = {name: param.requires_grad for name, param in model.named_parameters()}
f.write("{}\n".format(model))

criterion = nn.CrossEntropyLoss()
num_epochs = TRAIN['MAX_EPOCHS']

optimizer = optim.SGD(model.parameters(), lr=TRAIN['INIT_LR'], momentum=TRAIN['MOMENTUM'], weight_decay=TRAIN['WEIGHT_DECAY'])

if VALID_SPLIT > 0.0:
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=TRAIN['LR_SCHEDULE_GAMMA'], verbose=True)
else:
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                  milestones=TRAIN['LR_SCHEDULE'],
                                                  gamma = TRAIN['LR_SCHEDULE_GAMMA'])

#%% Updating model, optimizer, lr_scheduler, tracking variables, etc. if RESUME flag is specified...
if RESUME:
    if CKPT_DIR is None:
        raise ValueError("No checkpoint specified to resume training!")
    else:
        f.write("==>> Resuming training from loaded checkpoint from: {}\n".format(CKPT_DIR))
        ckpt = torch.load(CKPT_DIR, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        lr_scheduler.load_state_dict(ckpt['lr_scheduler'])
        start_epoch             = ckpt['epoch']
        best_val_acc            = ckpt['best_val_acc']
        best_val_loss           = ckpt['best_val_loss']
        train_loss              = ckpt['train_loss']
        train_acc               = ckpt['train_acc']
        valid_loss              = ckpt['valid_loss']
        valid_acc               = ckpt['valid_acc']
else:
    f.write("==>> Starting training from scratch!\n")
    start_epoch             = 0
    best_val_acc            = 0.0
    best_val_loss           = float('inf')
    train_loss              = []
    train_acc               = []
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
    total               = 0
    for batch_idx, (x_train, y_train) in enumerate(train_loader):
        x_train, y_train = x_train.to(device), y_train.to(device)
    
        optimizer.zero_grad()
        x_norm = normalization_func(x_train)
        output = model(x_norm)
        loss = criterion(output, y_train)
        
        loss.backward()
        optimizer.step()
            
        _, predictions      = torch.max(output.data, 1)
        total               += y_train.size(0)
        correct             += (predictions == y_train).sum().item()
        ave_loss            += loss.item()
    
        if (batch_idx+1) == len(train_loader):
                f.write('==>>> TRAIN-PRUNE | train epoch: {}, loss: {:.6f}, acc: {:.4f}\n'.format(
                        epoch, ave_loss*1.0/(batch_idx + 1), correct*1.0/total))
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
                'valid_loss': valid_loss,
                'train_acc': train_acc,
                'valid_acc': valid_acc}, SAVE_DIR)

f.write("Best val accuracy during training: {:.2f}\n".format(best_val_acc))
                          
#%% Test set model evaluation.
model.load_state_dict(best_msdict)

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
