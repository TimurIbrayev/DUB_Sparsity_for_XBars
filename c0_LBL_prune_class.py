#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 16 21:34:33 2020

@author: tibrayev
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from torch._six import container_abcs
from itertools import repeat


def _ntuple(n):
    def parse(x):
        if isinstance(x, container_abcs.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse

_single = _ntuple(1)
_pair = _ntuple(2)
_triple = _ntuple(3)
_quadruple = _ntuple(4)

class LBL_prune_class():
    def __init__(self, model, tile_size=64, ADC_res_bits=None):
        super(LBL_prune_class, self).__init__()
        (self.total_weights, 
         self.layerwise_weights)    = self.count_total_weights(model)
        self.tile_size              = _pair(tile_size)
        self.ADC_res_bits           = (int(math.ceil(math.log2(self.tile_size[1])))+2) if ADC_res_bits is None else ADC_res_bits

    def __repr__(self):
        status_msg = 'prune_v0_lbl with the following parameters: \n'\
                     '  total_weights={}\n'.format(self.total_weights)
        return status_msg
    
    def count_total_weights(self, model):
        count_total = 0
        count_layerwise = []
        with torch.no_grad():
            for m in model.modules():
                if isinstance(m, nn.Conv2d):
                    count_total += m.weight.numel()
                    count_layerwise.append(m.weight.numel())
        return count_total, count_layerwise
    
    def prune_layer(self, model, layer_id, target_prune_ratio):
        l_id = 0
        with torch.no_grad():
            for m in model.modules():
                if isinstance(m, nn.Conv2d):
                    if l_id == layer_id:
                        # get all weights
                        weights = m.weight.abs().view(-1).clone().detach()
                        # find threshold based on target prune ratio
                        threshold = np.percentile(weights.cpu(), target_prune_ratio)
                        # prune based on this threshold
                        mask = (m.weight.abs() <= threshold)
                        m.weight.masked_fill_(mask, 0.0)
                        count_zeros = (m.weight == 0.0).sum()
                        # break the loop as this script does one layer at a time
                        break
                    else:
                        l_id += 1
        return mask, count_zeros
    
    def switch_grad_req_on_layer(self, model, layer_id, requires_grad=False):
        l_id = 0
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                if l_id == layer_id:
                    for param in m.parameters():
                        param.requires_grad_(requires_grad)
                    break
                else:
                    l_id += 1
    
    def mask_layer(self, model, layer_id, mask):
        l_id = 0
        with torch.no_grad():
            for m in model.modules():
                if isinstance(m, nn.Conv2d):
                    if l_id == layer_id:
                        m.weight.masked_fill_(mask, 0.0)
                        break
                    else:
                        l_id += 1
    
    def mask_all_layers(self, model, masks):
        l_id = 0
        with torch.no_grad():
            for m in model.modules():
                if isinstance(m, nn.Conv2d):
                    m.weight.masked_fill_(masks[l_id], 0.0)
                    l_id += 1

    def count_zeros(self, model):
        count_zeros = 0
        count_zeros_layerwise = []
        with torch.no_grad():
            for m in model.modules():
                if isinstance(m, nn.Conv2d):
                    count_zeros += (m.weight == 0.0).sum()
                    count_zeros_layerwise.append((m.weight == 0.0).sum())
        return count_zeros, count_zeros_layerwise

               
    "Additional functions to assess network sparsity from tile perspective!"
    def assess_tile_sparsity(self, model):
        sparsity = []
        with torch.no_grad():
            for m in model.modules():
                if isinstance(m, nn.Conv2d):
                    w, h = m.weight.flatten(1).size()
                    weight = m.weight.view(w, h)
                    for i in range(0, w, self.tile_size[0]):
                        for j in range(0, h, self.tile_size[1]):
                            tile = weight[i:(i+self.tile_size[0]), j:(j+self.tile_size[1])]
                            zeros_column = (tile == 0.0).sum(dim=1).float()
                            zeros_column_min = torch.min(zeros_column)
                            zeros_column_per = zeros_column_min/tile.size(1)
                            sparsity.append(zeros_column_per.item())
        return sparsity
    
    def assess_tile_sparsity_on_given_layers(self, model, layer_ids=[]):
        sparsity = []
        idx = 0
        with torch.no_grad():
            for m in model.modules():
                if isinstance(m, nn.Conv2d):
                    if idx in layer_ids:
                        w, h = m.weight.flatten(1).size()
                        weight = m.weight.view(w, h)
                        for i in range(0, w, self.tile_size[0]):
                            for j in range(0, h, self.tile_size[1]):
                                tile = weight[i:(i+self.tile_size[0]), j:(j+self.tile_size[1])]
                                zeros_column = (tile == 0.0).sum(dim=1).float()
                                zeros_column_min = torch.min(zeros_column)
                                zeros_column_per = zeros_column_min/tile.size(1)
                                sparsity.append(zeros_column_per.item())
                    idx += 1
        return sparsity
    
    def hist_tile_sparsity(self, model=None, sparsity=None):
        if sparsity is None and model is None:
            raise ValueError("One of the input arguments is expected, but got both None!")
        elif model is not None:
            print("Received model! Using model to estimate sparsity and compute histogram!")
            sparsity = self.assess_tile_sparsity(model)
        
        bins = [1-(2**(-k)) for k in range(0, self.ADC_res_bits)]
        bins.append(1.0)
        hist = np.histogram(sparsity, bins)
        return hist
       