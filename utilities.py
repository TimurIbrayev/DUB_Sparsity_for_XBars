#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 00:09:35 2020

"""
import torch
import torchvision.transforms as transforms
import torchvision.datasets as dset
from torch.utils.data.sampler import SequentialSampler, SubsetRandomSampler
import numpy as np

def get_data_loaders(dataset,
                     data_dir, 
                     batch_size, 
                     augment,
                     random_seed,
                     valid_size=0.0,
                     shuffle=True,
                     num_workers=1,
                     pin_memory=True):
    error_msg = "[!] valid_size should be in the range [0, 1]."
    assert ((valid_size >= 0) and (valid_size <= 1)), error_msg
    
    # if dataset == 'CIFAR10' or dataset == 'cifar10':
    #     normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    # elif dataset == 'imagenet2012':
    #     normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    if dataset == 'CIFAR10':
        test_transform      = transforms.Compose([transforms.ToTensor()])
        valid_transform     = transforms.Compose([transforms.ToTensor()])        
        if augment: 
            train_transform = transforms.Compose([transforms.RandomCrop(size=32, padding=4),
                                                  transforms.RandomHorizontalFlip(),
                                                  transforms.ToTensor()])
        else:
            train_transform = transforms.Compose([transforms.ToTensor()])
            
        train_set = dset.CIFAR10(root=data_dir, train=True,  transform=train_transform, download=True)
        valid_set = dset.CIFAR10(root=data_dir, train=True,  transform=valid_transform, download=True)
        test_set  = dset.CIFAR10(root=data_dir, train=False, transform=test_transform,  download=True)
            
    elif dataset == 'imagenet2012':
        test_transform      = transforms.Compose([transforms.Resize(256),
                                                  transforms.CenterCrop(224),
                                                  transforms.ToTensor()])
        valid_transform     = transforms.Compose([transforms.Resize(256),
                                                  transforms.CenterCrop(224),
                                                  transforms.ToTensor()])
        if augment:
            train_transform = transforms.Compose([transforms.Resize(256),
                                                  transforms.RandomResizedCrop(224),
                                                  transforms.RandomHorizontalFlip(),
                                                  transforms.ToTensor()])
        else:
            train_transform = transforms.Compose([transforms.Resize(256),
                                                  transforms.CenterCrop(224),
                                                  transforms.ToTensor()])
        train_set = dset.ImageFolder(root=data_dir + '/train', transform=train_transform)
        valid_set = dset.ImageFolder(root=data_dir + '/train', transform=valid_transform)
        test_set  = dset.ImageFolder(root=data_dir + '/val',   transform=test_transform)
    
    print('\nForming the samplers for train and validation splits with split fraction={}'.format(valid_size))
    num_train = len(train_set)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))
    
    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
        
    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    
    print('Preparing dataloaders...\n')
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers, pin_memory=pin_memory)
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size, sampler=valid_sampler, num_workers=num_workers, pin_memory=pin_memory)
    test_loader  = torch.utils.data.DataLoader(test_set,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    
    return train_loader, valid_loader, test_loader
    
















    