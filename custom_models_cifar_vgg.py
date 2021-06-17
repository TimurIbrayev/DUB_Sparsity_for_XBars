#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 22:55:07 2020

@modified by: tibrayev
"""

import torch
import torch.nn as nn

class customizable_VGG(nn.Module):
    def __init__(self, features, num_classes=100, fc1=512, fc2=512, init_weights=True):
        super(customizable_VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512, fc1),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(fc1, fc2),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(fc2, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x, with_latent = False):
        x = self.features(x)
        features = torch.flatten(x, 1)
        outputs  = self.classifier(features)
        if with_latent:
            return outputs, features
        else:
            return outputs

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def vgg11(**kwargs):
    model = customizable_VGG(make_layers(cfgs['A'], batch_norm=False), **kwargs)
    print("Requested model is modified to suit CIFAR10/100 image resolutions of 3x32x32!")
    return model

def vgg11_bn(**kwargs):
    model = customizable_VGG(make_layers(cfgs['A'], batch_norm=True), **kwargs)
    print("Requested model is modified to suit CIFAR10/100 image resolutions of 3x32x32!")
    return model

def vgg13(**kwargs):
    model = customizable_VGG(make_layers(cfgs['B'], batch_norm=False), **kwargs)
    print("Requested model is modified to suit CIFAR10/100 image resolutions of 3x32x32!")
    return model

def vgg13_bn(**kwargs):
    model = customizable_VGG(make_layers(cfgs['B'], batch_norm=True), **kwargs)
    print("Requested model is modified to suit CIFAR10/100 image resolutions of 3x32x32!")
    return model

def vgg16(**kwargs):
    model = customizable_VGG(make_layers(cfgs['D'], batch_norm=False), **kwargs)
    print("Requested model is modified to suit CIFAR10/100 image resolutions of 3x32x32!")
    return model

def vgg16_bn(**kwargs):
    model = customizable_VGG(make_layers(cfgs['D'], batch_norm=True), **kwargs)
    print("Requested model is modified to suit CIFAR10/100 image resolutions of 3x32x32!")
    return model

def vgg19(**kwargs):
    model = customizable_VGG(make_layers(cfgs['E'], batch_norm=False), **kwargs)
    print("Requested model is modified to suit CIFAR10/100 image resolutions of 3x32x32!")
    return model

def vgg19_bn(**kwargs):
    model = customizable_VGG(make_layers(cfgs['E'], batch_norm=True), **kwargs)
    print("Requested model is modified to suit CIFAR10/100 image resolutions of 3x32x32!")
    return model