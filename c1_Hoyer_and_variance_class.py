#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 17:23:55 2020
Last assessed on Fri Dec 25 18:56:23 2020

@author: tibrayev
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch._six import container_abcs
from itertools import repeat

import math

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


class gradient_gate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return input
    
    @staticmethod
    def backward(ctx, grad_output):
        grad_input  = grad_output.clone()
        grad_input.masked_fill_(grad_input <= 0.0, 0.0)
        return grad_input


class HoyerAndVariance():
    def __init__(self, 
                 # required arguments
                 model, device, 
                 # algorithmic strength arguments
                 lambda_variance = 1.0e-4,
                 lambda_mean     = 0.0,
                 tol=1.0e-3,
                 # hardware specific arguments
                 tile_size=64, 
                 ADC_res_bits=None,
                 weight_quantization=1,
                 ):
        super(HoyerAndVariance, self).__init__()
        self.device             = device
        self.lambda_variance    = lambda_variance
        self.lambda_mean        = lambda_mean
        self.tol                = tol
        
        if isinstance(tile_size, container_abcs.Iterable):
            self.tile_size = tile_size
        else:
            assert (weight_quantization == 1) or (weight_quantization % 2 == 0), "Weight quantization should be either 1 or a power of 2! But got {}".format(weight_quantization)
            self.weight_quantization    = weight_quantization
            self.weights_in_tile_row    = int(tile_size/self.weight_quantization)
            self.tile_size              = (self.weights_in_tile_row, tile_size)
        self.ADC_res_bits       = (int(math.ceil(math.log2(self.tile_size[1])))+2) if ADC_res_bits is None else ADC_res_bits
        
        self.total_prune_layers = 0
        self.total_weights      = 0
        self.total_tiles        = 0
        
        self.layer_params       = []
        self.layerwise_weights  = []
        self.layerwise_tiles    = []
        self.init_model_assessment(model)
        
        
    def __repr__(self):
        status_msg = 'HoyerAndVariance with the following parameters: \n'\
                     '  tile_size={}\n'\
                     '  total_prune_layers={}\n'\
                     '  total_weights={}\n'\
                     '  total_tiles={}\n'\
                     '  tol={}\n'\
                     '  mean_over_hoyer_lambda={}\n'\
                     '  variance_over_hoyer_lambda={}\n'.format(    
                       self.tile_size, 
                       self.total_prune_layers, self.total_weights, self.total_tiles,
                       self.tol, self.lambda_mean, self.lambda_variance)
        return status_msg
    
# =============================================================================
#    INITIAL ASSESSMENT METHODS
# =============================================================================
    def init_model_assessment(self, model):
        l_id = 0
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                w, h = m.weight.flatten(1).size()
                cnt_tiles = 0
                for i in range(0, w, self.tile_size[0]):
                    for j in range(0, h, self.tile_size[1]):
                        cnt_tiles += 1
                
                # if uniformity loss is not required to be computed, it is faster to implement sparsity loss computation as convolution operation.
                # for that, it is easier to precompute 0 padding required for smaller or irregularly shaped layers (with respect to given tile sizes)
                # which is then applied to flattened layer to make it possible to convolve with kernel having tile sizes.
                # Note: the padding value is constant 0s and are applied only from one side for each dimension requiring padding:
                # that it right side of the tensor for h and bottom side of the tensor for w.  
                pad_h = 0 if (h % self.tile_size[1]) == 0.0 else self.tile_size[1] - (h % self.tile_size[1])
                pad_w = 0 if (w % self.tile_size[0]) == 0.0 else self.tile_size[0] - (w % self.tile_size[0])
                padding = (0, pad_h, 0, pad_w)
                
                layer = {}
                # add model and xbar dependent configs
                layer['wh']             = w, h
                layer['cnt_tiles']      = cnt_tiles
                layer['padding']        = padding
                layer['kernel_column']  = torch.ones((1, 1, 1, self.tile_size[1]), device=self.device, requires_grad=False)
                layer['kernel_row']     = (self.tile_size[0], 1)
                self.layer_params.append(layer)
                
                self.total_prune_layers += 1
                self.total_weights      += w*h
                self.total_tiles        += cnt_tiles
                self.layerwise_weights.append(w*h)
                self.layerwise_tiles.append(cnt_tiles)
                l_id += 1

# =============================================================================
#   TRAINING METHODS     
# =============================================================================
    # def IntraTileHS(self, model):
    #     loss_IntraTileHS = None
    #     l_idx = 0
    #     for m in model.modules():
    #         if isinstance(m, nn.Conv2d):
    #             w, h                    = self.layer_params[l_idx]['wh']
    #             kernel                  = self.layer_params[l_idx]['kernel']
    #             padding                 = self.layer_params[l_idx]['padding']
    #             weight                  = m.weight.view(1, 1, w, h)
    #             padded_2D_weights       = F.pad(weight, padding, 'constant', value=0.0)
                
    #             padded_2D_weights_squared   = torch.pow(padded_2D_weights, 2)
                
                
    #             IntraTileHS_numerators      = torch.pow(F.conv2d(torch.abs(padded_2D_weights), kernel, stride=self.tile_size), 2) # numel should be equal to the number of tiles in the layer
    #             IntraTileHS_denominators    = F.conv2d(padded_2D_weights_squared, kernel, stride=self.tile_size) # numel should be equal to the number of tiles in the layer
    #             IntraTileHS_tiles           = IntraTileHS_numerators / IntraTileHS_denominators
    #             #IntraTileHS_tile_ths        = IntraTileHS_denominators / torch.sqrt(IntraTileHS_numerators)
                
    #             if loss_IntraTileHS is None:
    #                 loss_IntraTileHS  = self.intrahs_lambda * torch.sum(IntraTileHS_tiles)
    #             else:
    #                 loss_IntraTileHS += self.intrahs_lambda * torch.sum(IntraTileHS_tiles)
    #             l_idx += 1
    #     return loss_IntraTileHS
    
    
    def compute_loss(self, model):
        loss_tile   = None
        l_idx       = 0
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                w, h                    = self.layer_params[l_idx]['wh']
                kernel_column           = self.layer_params[l_idx]['kernel_column']
                kernel_row              = self.layer_params[l_idx]['kernel_row']
                padding                 = self.layer_params[l_idx]['padding']
                weight                  = m.weight.view(1, 1, w, h)
                padded_2D_weights       = F.pad(weight, padding, 'constant', value=0.0)
                
                padded_2D_weights_squared   = torch.pow(padded_2D_weights, 2)
                
                
                column_numerators       = torch.pow(F.conv2d(torch.abs(padded_2D_weights), kernel_column, stride=(1, self.tile_size[1])), 2)
                column_denominators     = F.conv2d(padded_2D_weights_squared, kernel_column, stride=(1, self.tile_size[1])) + 0.000001
                column_hoyers           = column_numerators / column_denominators
                
                # No gate here!
                
                variance_of_hoyers      = F.unfold(column_hoyers, kernel_row, stride=kernel_row).var(dim=1).squeeze(0)
                
                if loss_tile is None:
                    loss_tile  = self.lambda_variance * torch.sum(variance_of_hoyers)
                else:
                    loss_tile += self.lambda_variance * torch.sum(variance_of_hoyers)
                l_idx += 1
        return loss_tile

    def compute_loss_with_gate(self, model):
        loss_tile   = None
        l_idx       = 0
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                w, h                    = self.layer_params[l_idx]['wh']
                kernel_column           = self.layer_params[l_idx]['kernel_column']
                kernel_row              = self.layer_params[l_idx]['kernel_row']
                padding                 = self.layer_params[l_idx]['padding']
                weight                  = m.weight.view(1, 1, w, h)
                padded_2D_weights       = F.pad(weight, padding, 'constant', value=0.0)
                
                padded_2D_weights_squared   = torch.pow(padded_2D_weights, 2)
                
                
                column_numerators       = torch.pow(F.conv2d(torch.abs(padded_2D_weights), kernel_column, stride=(1, self.tile_size[1])), 2)
                column_denominators     = F.conv2d(padded_2D_weights_squared, kernel_column, stride=(1, self.tile_size[1])) + 0.000001
                column_hoyers           = column_numerators / column_denominators
                
                # Gate here!
                gated_hoyer             = gradient_gate.apply(column_hoyers)
                
                variance_of_hoyers      = F.unfold(gated_hoyer, kernel_row, stride=kernel_row).var(dim=1).squeeze(0)
                
                if loss_tile is None:
                    loss_tile  = self.lambda_variance * torch.sum(variance_of_hoyers)
                else:
                    loss_tile += self.lambda_variance * torch.sum(variance_of_hoyers)
                l_idx += 1
        return loss_tile
    
    def compute_loss_nogate_mean_and_variance(self, model):
        loss_tile   = None
        l_idx       = 0
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                w, h                    = self.layer_params[l_idx]['wh']
                kernel_column           = self.layer_params[l_idx]['kernel_column']
                kernel_row              = self.layer_params[l_idx]['kernel_row']
                padding                 = self.layer_params[l_idx]['padding']
                weight                  = m.weight.view(1, 1, w, h)
                padded_2D_weights       = F.pad(weight, padding, 'constant', value=0.0)
                
                padded_2D_weights_squared   = torch.pow(padded_2D_weights, 2)
                
                
                column_numerators       = torch.pow(F.conv2d(torch.abs(padded_2D_weights), kernel_column, stride=(1, self.tile_size[1])), 2)
                column_denominators     = F.conv2d(padded_2D_weights_squared, kernel_column, stride=(1, self.tile_size[1])) + 0.000001
                column_hoyers           = column_numerators / column_denominators
                
                unfolded_tensor         = F.unfold(column_hoyers, kernel_row, stride=kernel_row)
                
                # No gate here!
                
                variance_of_hoyers      = unfolded_tensor.var(dim=1).squeeze(0)
                mean_of_hoyers          = (unfolded_tensor.mean(dim=1).squeeze(0) - 1.0)
                
                if loss_tile is None:
                    loss_tile  = self.lambda_variance * torch.sum(variance_of_hoyers) + self.lambda_mean * torch.sum(mean_of_hoyers)
                else:
                    loss_tile += self.lambda_variance * torch.sum(variance_of_hoyers) + self.lambda_mean * torch.sum(mean_of_hoyers)
                l_idx += 1
        return loss_tile
    
    def compute_loss_mean_and_gatedvariance(self, model):
        loss_tile   = None
        l_idx       = 0
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                w, h                    = self.layer_params[l_idx]['wh']
                kernel_column           = self.layer_params[l_idx]['kernel_column']
                kernel_row              = self.layer_params[l_idx]['kernel_row']
                padding                 = self.layer_params[l_idx]['padding']
                weight                  = m.weight.view(1, 1, w, h)
                padded_2D_weights       = F.pad(weight, padding, 'constant', value=0.0)
                
                padded_2D_weights_squared   = torch.pow(padded_2D_weights, 2)
                
                
                column_numerators       = torch.pow(F.conv2d(torch.abs(padded_2D_weights), kernel_column, stride=(1, self.tile_size[1])), 2)
                column_denominators     = F.conv2d(padded_2D_weights_squared, kernel_column, stride=(1, self.tile_size[1])) + 0.000001
                column_hoyers           = column_numerators / column_denominators
                
                unfolded_tensor         = F.unfold(column_hoyers, kernel_row, stride=kernel_row)
                # Gate here!
                gated_hoyer             = gradient_gate.apply(unfolded_tensor)
                                
                variance_of_hoyers      = gated_hoyer.var(dim=1).squeeze(0)
                mean_of_hoyers          = (unfolded_tensor.mean(dim=1).squeeze(0) - 1.0)
                
                if loss_tile is None:
                    loss_tile  = self.lambda_variance * torch.sum(variance_of_hoyers) + self.lambda_mean * torch.sum(mean_of_hoyers)
                else:
                    loss_tile += self.lambda_variance * torch.sum(variance_of_hoyers) + self.lambda_mean * torch.sum(mean_of_hoyers)
                l_idx += 1
        return loss_tile    
        

    
    def zero_out_gradients(self, model, masks):
        l_idx = 0
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.grad[masks[l_idx]] = 0.0
                l_idx += 1   
    
    def zero_out_weights(self, model, masks):
        l_idx = 0
        with torch.no_grad():
            for m in model.modules():
                if isinstance(m, nn.Conv2d):
                    m.weight.masked_fill_(masks[l_idx], 0.0)
                    l_idx += 1

# =============================================================================
#   TILE SPARSITY ASSESSMENT METHODS        
# =============================================================================
    def assess_tile_sparsity(self, model):
        sparsity = []
        l_idx = 0
        with torch.no_grad():
            for m in model.modules():
                if isinstance(m, nn.Conv2d):
                    w, h = self.layer_params[l_idx]['wh']
                    weight = m.weight.view(w, h)
                    for i in range(0, w, self.tile_size[0]):
                        for j in range(0, h, self.tile_size[1]):
                            tile = weight[i:(i+self.tile_size[0]), j:(j+self.tile_size[1])]
                            zeros_column = (tile == 0.0).sum(dim=1).float()
                            zeros_column_min = torch.min(zeros_column)
                            zeros_column_per = zeros_column_min/tile.size(1)
                            sparsity.append(zeros_column_per.item())
                    l_idx += 1
        return sparsity
    
    def assess_tile_sparsity_on_given_layers(self, model, layer_ids=[]):
        sparsity = []
        l_idx = 0
        with torch.no_grad():
            for m in model.modules():
                if isinstance(m, nn.Conv2d):
                    if l_idx in layer_ids:
                        w, h = self.layer_params[l_idx]['wh']
                        weight = m.weight.view(w, h)
                        for i in range(0, w, self.tile_size[0]):
                            for j in range(0, h, self.tile_size[1]):
                                tile = weight[i:(i+self.tile_size[0]), j:(j+self.tile_size[1])]
                                zeros_column = (tile == 0.0).sum(dim=1).float()
                                zeros_column_min = torch.min(zeros_column)
                                zeros_column_per = zeros_column_min/tile.size(1)
                                sparsity.append(zeros_column_per.item())
                    l_idx += 1
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


    def assess_tile_sparsity_almost_zeros(self, model):
        sparsity = []
        l_idx = 0
        with torch.no_grad():
            for m in model.modules():
                if isinstance(m, nn.Conv2d):
                    w, h = self.layer_params[l_idx]['wh']
                    weight = m.weight.view(w, h)
                    for i in range(0, w, self.tile_size[0]):
                        for j in range(0, h, self.tile_size[1]):
                            tile = weight[i:(i+self.tile_size[0]), j:(j+self.tile_size[1])]
                            almost_zeros_column = (tile.abs() <= self.tol).sum(dim=1).float()
                            almost_zeros_column_min = torch.min(almost_zeros_column)
                            almost_zeros_column_per = almost_zeros_column_min/tile.size(1)
                            sparsity.append(almost_zeros_column_per.item())
                    l_idx += 1
        return sparsity
    
    def assess_tile_sparsity_on_given_layers_almost_zeros(self, model, layer_ids=[]):
        sparsity = []
        l_idx = 0
        with torch.no_grad():
            for m in model.modules():
                if isinstance(m, nn.Conv2d):
                    if l_idx in layer_ids:
                        w, h = self.layer_params[l_idx]['wh']
                        weight = m.weight.view(w, h)
                        for i in range(0, w, self.tile_size[0]):
                            for j in range(0, h, self.tile_size[1]):
                                tile = weight[i:(i+self.tile_size[0]), j:(j+self.tile_size[1])]
                                almost_zeros_column = (tile.abs() <= self.tol).sum(dim=1).float()
                                almost_zeros_column_min = torch.min(almost_zeros_column)
                                almost_zeros_column_per = almost_zeros_column_min/tile.size(1)
                                sparsity.append(almost_zeros_column_per.item())
                    l_idx += 1
        return sparsity

    def hist_tile_sparsity_almost_zeros(self, model=None, sparsity=None):
        if sparsity is None and model is None:
            raise ValueError("One of the input arguments is expected, but got both None!")
        elif model is not None:
            print("Received model! Using model to estimate sparsity and compute histogram!")
            sparsity = self.assess_tile_sparsity_almost_zeros(model)
        
        bins = [1-(2**(-k)) for k in range(0, self.ADC_res_bits)]
        bins.append(1.0)
        hist = np.histogram(sparsity, bins)
        return hist

# =============================================================================
#   MONITORING METHODS                    
# =============================================================================
    def count_almost_zeros(self, model):
        count_almost_zeros = 0
        count_almost_zeros_layerwise = []
        l_idx = 0
        with torch.no_grad():
            for m in model.modules():
                if isinstance(m, nn.Conv2d):
                    count_almost_zeros += (m.weight.abs() <= self.tol).sum()
                    count_almost_zeros_layerwise.append((m.weight.abs() <= self.tol).sum())
                    l_idx += 1
        return count_almost_zeros, count_almost_zeros_layerwise
    
    def count_zeros(self, model):
        count_zeros = 0
        count_zeros_layerwise = []
        with torch.no_grad():
            for m in model.modules():
                if isinstance(m, nn.Conv2d):
                    count_zeros += (m.weight == 0.0).sum()
                    count_zeros_layerwise.append((m.weight == 0.0).sum())
        return count_zeros, count_zeros_layerwise
    
# =============================================================================
#   PRUNING METHODS
# =============================================================================
    def prune_based_on_tile_sparsity(self, model):
        bins = [1-(2**(-k)) for k in range(0, self.ADC_res_bits)]
        bins.append(1.0)
        
        l_idx = 0
        with torch.no_grad():
            for m in model.modules():
                if isinstance(m, nn.Conv2d):
                    w, h = self.layer_params[l_idx]['wh']
                    weight = m.weight.view(w, h)
                    for i in range(0, w, self.tile_size[0]):
                        for j in range(0, h, self.tile_size[1]):
                            tile = weight[i:(i+self.tile_size[0]), j:(j+self.tile_size[1])].abs()
                            almost_zeros_column = (tile <= self.tol).sum(dim=1).float()
                            almost_zeros_column_min = torch.min(almost_zeros_column)
                            almost_zeros_column_per = almost_zeros_column_min/tile.size(1)
                            bin_idx = np.histogram(almost_zeros_column_per.item(), bins)[0].argmax()
                            if bin_idx == self.ADC_res_bits-1: # case when tile sparsity is (tile.size(1)/tile.size(1))
                                tile.masked_fill_((tile <= self.tol), 0.0)
                            elif bin_idx == self.ADC_res_bits-2: # case when tile sparsity is ((tile.size(1)-1)/tile.size(1))
                                tile.masked_fill_((tile >= 0.0), 0.0)
                            else:
                                bin_mid = (bins[bin_idx] + bins[bin_idx+1])/2
                                if almost_zeros_column_per.item() >= bin_mid: # forced tile
                                    target_percentile_on_each_column = bins[bin_idx+1]*100.0
                                else: # loosened tile
                                    target_percentile_on_each_column = bins[bin_idx]*100.0
                                for k in range(tile.size(0)):
                                    column_th = np.percentile(tile[k].cpu(), target_percentile_on_each_column)
                                    tile[k].masked_fill_((tile[k] <= column_th), 0.0)
                            
                            weight.data[i:(i+self.tile_size[0]), j:(j+self.tile_size[1])].masked_fill_((tile == 0.0), 0.0)
                    l_idx += 1
        
        return self.prep_masks_and_count_zeros(model)

    def prep_masks_and_count_zeros(self, model):
        masks = []
        count_zeros = 0.0
        with torch.no_grad():
            for m in model.modules():
                if isinstance(m, nn.Conv2d):
                    mask = (m.weight == 0.0)
                    count_zeros += mask.sum()
                    masks.append(mask.clone().detach())
        return count_zeros, masks


    
    def prune_fixed_prune_ratios_based_on_tile_sparsity(self, model, target_prune_ratios):
        bins = [1-(2**(-k)) for k in range(0, self.ADC_res_bits)]
        bins.append(1.0)
        
        l_idx = 0
        with torch.no_grad():
            for m in model.modules():
                if isinstance(m, nn.Conv2d):
                    w, h = self.layer_params[l_idx]['wh']
                    weight = m.weight.view(w, h)
                    # per layer threshold
                    threshold = np.percentile(weight.view(-1).abs().clone().detach().cpu(), target_prune_ratios[l_idx])
                    for i in range(0, w, self.tile_size[0]):
                        for j in range(0, h, self.tile_size[1]):
                            tile = weight[i:(i+self.tile_size[0]), j:(j+self.tile_size[1])].abs()
                            almost_zeros_column = (tile <= threshold).sum(dim=1).float()
                            almost_zeros_column_min = torch.min(almost_zeros_column)
                            almost_zeros_column_per = almost_zeros_column_min/tile.size(1)
                            bin_idx = np.histogram(almost_zeros_column_per.item(), bins)[0].argmax()
                            if bin_idx == self.ADC_res_bits-1: # case when tile sparsity is (tile.size(1)/tile.size(1))
                                tile.masked_fill_((tile <= threshold), 0.0)
                            elif bin_idx == self.ADC_res_bits-2: # case when tile sparsity is ((tile.size(1)-1)/tile.size(1))
                                #tile.masked_fill_((tile >= 0.0), 0.0)
                                target_percentile_on_each_column = bins[bin_idx]*100.0
                                for k in range(tile.size(0)):
                                    column_th = np.percentile(tile[k].cpu(), target_percentile_on_each_column)
                                    tile[k].masked_fill_((tile[k] <= column_th), 0.0)
                            else:
                                bin_mid = (bins[bin_idx] + bins[bin_idx+1])/2
                                if almost_zeros_column_per.item() >= bin_mid: # forced tile
                                    target_percentile_on_each_column = bins[bin_idx+1]*100.0
                                else: # loosened tile
                                    target_percentile_on_each_column = bins[bin_idx]*100.0
                                for k in range(tile.size(0)):
                                    column_th = np.percentile(tile[k].cpu(), target_percentile_on_each_column)
                                    tile[k].masked_fill_((tile[k] <= column_th), 0.0)
                            
                            weight.data[i:(i+self.tile_size[0]), j:(j+self.tile_size[1])].masked_fill_((tile == 0.0), 0.0)
                    l_idx += 1
        
        return self.prep_masks_and_count_zeros(model)
