#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 22:54:34 2020

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

torch.set_printoptions(linewidth = 160)
np.set_printoptions(linewidth = 160)
np.set_printoptions(precision=4)
np.set_printoptions(suppress='True')

from custom_models_cifar_vgg import vgg11
from custom_normalization_functions import custom_3channel_img_normalization_with_per_image_params

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
LOG = 'clean_model'
#f = open('./results/log_{}.txt'.format(LOG), 'a', buffering=1)
f = sys.stdout
# Timestamp
f.write('\n*******************************************************************\n')
f.write('==>> Run on: '+time.strftime("%Y-%m-%d %H:%M:%S")+'\n')
f.write('==>> Seed was set to: {}\n'.format(SEED))

DATASET             = 'CIFAR10'
BATCH_SIZE          = 128
MAX_EPOCHS          = 200
MOMENTUM            = 0.9
WEIGHT_DECAY        = 0.0005
INIT_LR             = 0.1
LR_SCHEDULE         = [50, 100, 150]
LR_SCHEDULE_GAMMA   = 0.1


#%% Load the dataset.
root_dir = './datasets/{}'.format(DATASET)
if not os.path.exists(root_dir): os.mkdir(root_dir)
train_transform = transforms.Compose([transforms.RandomCrop(size=32, padding=4),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor()])
train_set = dset.CIFAR10(root = root_dir, train = True, transform = train_transform, download=True)

test_transform = transforms.Compose([transforms.ToTensor()])
test_set  = dset.CIFAR10(root = root_dir, train = False, transform = test_transform, download=True)

train_loader = torch.utils.data.DataLoader(dataset = train_set, batch_size = BATCH_SIZE, shuffle=True)
test_loader  = torch.utils.data.DataLoader(dataset = test_set,  batch_size = BATCH_SIZE, shuffle=False)

f.write('==>> Batch size: {}\n'.format(BATCH_SIZE))
f.write('==>> Total training batch number: {}\n'.format(len(train_loader)))
f.write('==>> Total testing batch number: {}\n'.format(len(test_loader)))

#%% Data manipulators.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
normalization_func = custom_3channel_img_normalization_with_per_image_params(img_dimensions = [3, 32, 32],
                                                                             device = device)

#%% #FIXME: Load the model.
model = vgg11(num_classes=len(test_set.classes))
model.to(device)

grad_requirement_dict = {name: param.requires_grad for name, param in model.named_parameters()}
params = [p for p in model.parameters() if p.requires_grad]
f.write("{}\n".format(model))

#%% Training params.
criterion = nn.CrossEntropyLoss()
num_epochs = MAX_EPOCHS

optimizer = optim.SGD(params, lr=INIT_LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                              milestones=LR_SCHEDULE,
                                              gamma = LR_SCHEDULE_GAMMA)
f.write("==>> Optimizer init settings: {}\n".format(optimizer))
f.write("==>> LR Schedule: {} with Gamma: {}\n".format(LR_SCHEDULE, LR_SCHEDULE_GAMMA))
f.write("==>> Number of training epochs: {}\n".format(num_epochs))

#%% Training cycle.
train_loss = []
valid_loss = []
train_acc = []
valid_acc = []
for epoch in range(num_epochs):
    # Train for one epoch
    model.train()
    correct     = 0.0
    ave_loss    = 0.0
    total       = 0
    for batch_idx, (x_train, y_train) in enumerate(train_loader):
        x_train, y_train = x_train.to(device), y_train.to(device)
        
        optimizer.zero_grad()
        x_norm = normalization_func(x_train)
        output = model(x_norm)
        loss   = criterion(output, y_train)
        loss.backward()
        optimizer.step()

        _, predictions   = torch.max(output.data, 1)
        total           += y_train.size(0)
        correct         += (predictions == y_train).sum().item()
        ave_loss        += loss.item()
        
        if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == len(train_loader):
            f.write('==>>> CLEAN TRAIN | epoch: {}, batch index: {}, train loss: {:.6f}, train acc: {:.4f}\n'.format(
                        epoch, batch_idx+1, ave_loss*1.0/(batch_idx + 1), correct*1.0/total))
    train_loss.append(ave_loss*1.0/(batch_idx+1))
    train_acc.append(correct*100.0/total)
    
    # Adjust learning rate
    lr_scheduler.step()
    
    # Evaluate on the clean val set
    model.eval()
    correct     = 0.0
    ave_loss    = 0.0
    total       = 0
    with torch.no_grad():
        for batch_idx, (x_val, y_val) in enumerate(test_loader):
            x_val, y_val = x_val.to(device), y_val.to(device)
            x_norm = normalization_func(x_val)
            output = model(x_norm)
            loss   = criterion(output, y_val)
            
            _, predictions   = torch.max(output.data, 1)
            total           += y_val.size(0)
            correct         += (predictions == y_val).sum().item()
            ave_loss        += loss.item()
            
            if (batch_idx + 1) % 100 == 0 or (batch_idx+1) == len(test_loader):
                f.write('==>>> CLEAN VALIDATE | epoch: {}, batch index: {}, val loss: {:.6f}, val acc: {:.4f}\n'.format(
                        epoch, batch_idx+1, ave_loss*1.0/(batch_idx + 1), correct*1.0/total))
        valid_loss.append(ave_loss*1.0/(batch_idx+1))
        valid_acc.append(correct*100.0/total)
        
    torch.save({'SEED': SEED,
                'model': model.state_dict(),
                'grad_requirement_dict': grad_requirement_dict,
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'num_epochs': num_epochs,
                'train_loss': train_loss,
                'valid_loss': valid_loss,
                'train_acc': train_acc,
                'valid_acc': valid_acc}, './results/checkpoint_{}.pth'.format(LOG))

























