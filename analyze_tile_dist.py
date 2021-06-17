#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 13:05:57 2021

@author: tibrayev
"""
#%run a_inference.py --checkpoint='./results/imagenet2012/lbl_sensitivity/checkpoint_lbl_prs_[60.0, 60.0, 70.0, 55.0, 60.0, 60.0, 60.0, 50.0, 65.0, 65.0, 70.0, 70.0, 85.0, 75.0, 75.0, 65.0, 75.0, 70.0, 75.0, 75.0].pth' --log='sys'

#######################################################
######### UNSTRUCTURED ##########
######################################################
from a_Hoyer_and_variance_class import HoyerAndVariance
prune = HoyerAndVariance(model, device)
ADC_res_bits = prune.ADC_res_bits
tile_size = prune.tile_size

bins = [1-(2**(-k)) for k in range(0, ADC_res_bits)]
distribution_tile_sparsitywise = {}
for b_i, b_v in enumerate(bins):
    distribution_tile_sparsitywise[b_i] = {'tile_sparsity': b_v if b_v != 63.5/64.0 else 1.0,
                                           'number_of_columns': 0, 
                                           'column_nonzero_dist_wrt_max': []}
bins.append(1.0)


with torch.no_grad():
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            weight = m.weight.flatten(1)
            w, h = weight.size()
            for i in range(0, w, tile_size[0]):
                for j in range(0, h, tile_size[1]):
                    tile = weight[i:(i+tile_size[0]), j:(j+tile_size[1])].abs()
                    zeros_column = (tile == 0.0).sum(dim=1).float()
                    zeros_column_min = torch.min(zeros_column)
                    zeros_column_per = zeros_column_min/tile.size(1)
                    hist = np.histogram(zeros_column_per.item(), bins)
                    bin_idx = hist[0].argmax()
                    
                    distribution_tile_sparsitywise[bin_idx]['number_of_columns'] += zeros_column.numel()
                    
                    
                    nonzeros_column = tile_size[1] - zeros_column
                    nonzeros_max = nonzeros_column.max()
                    column_nonzero_dist_wrt_max = (nonzeros_max - nonzeros_column).cpu()
                    distribution_tile_sparsitywise[bin_idx]['column_nonzero_dist_wrt_max'].append(column_nonzero_dist_wrt_max)
                    
                         
tile_sparsity = 1
bins_column_nonzero_distances = [k for k in range(0, tile_size[1]+1)]
max_distance_for_given_tile_sparsity = (1.0-distribution_tile_sparsitywise[tile_sparsity]['tile_sparsity'])*tile_size[1]
bins_column_nonzero_distances = [k for k in range(0, int(max_distance_for_given_tile_sparsity)+1)]
column_nonzero_dist_wrt_max_ALL_unstruc = torch.cat(distribution_tile_sparsitywise[tile_sparsity]['column_nonzero_dist_wrt_max'])
counts_unstruc, bins_unstruc, _ = plt.hist(column_nonzero_dist_wrt_max_ALL_unstruc, bins_column_nonzero_distances, density=True)



#######################################################
######### Our method (post-training pre-pruning) ##########
######################################################
CKPT_DIR_NEW = '/home/min/a/tibrayev/RESEARCH/active_pruning_for_xbars/results/imagenet2012/GatedVarianceOverHoyer/pretrained/checkpoint_model_GatedVarianceOverHoyer_lambdas_l2reg0.0001_var_h0.0005_tol0.001.pth'
ckpt = torch.load(CKPT_DIR_NEW, map_location=device)
model.load_state_dict(ckpt['model'])
f.write("Pretrained model was loaded from checkpoint: {}\n".format(CKPT_DIR_NEW))  
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


from a_Hoyer_and_variance_class import HoyerAndVariance
prune = HoyerAndVariance(model, device)
ADC_res_bits = prune.ADC_res_bits
tile_size = prune.tile_size

bins = [1-(2**(-k)) for k in range(0, ADC_res_bits)]
distribution_tile_sparsitywise = {}
for b_i, b_v in enumerate(bins):
    distribution_tile_sparsitywise[b_i] = {'tile_sparsity': b_v if b_v != 63.5/64.0 else 1.0,
                                           'number_of_columns': 0, 
                                           'column_nonzero_dist_wrt_max': []}
bins.append(1.0)


layerwise_ratios = [60.0, 60.0, 70.0, 55.0, 60.0, 60.0, 60.0, 50.0, 65.0, 65.0, 70.0, 70.0, 85.0, 75.0, 75.0, 65.0, 75.0, 70.0, 75.0, 75.0]
layerwise_ths = []
l_id = 0
with torch.no_grad():
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            weight = m.weight.flatten(1)
            w, h = weight.size()
            th = np.percentile(weight.view(-1).abs().clone().detach().cpu(), layerwise_ratios[l_id])
            layerwise_ths.append(th)
            for i in range(0, w, tile_size[0]):
                for j in range(0, h, tile_size[1]):
                    tile = weight[i:(i+tile_size[0]), j:(j+tile_size[1])].abs()
                    zeros_column = (tile <= th).sum(dim=1).float()
                    zeros_column_min = torch.min(zeros_column)
                    zeros_column_per = zeros_column_min/tile.size(1)
                    hist = np.histogram(zeros_column_per.item(), bins)
                    bin_idx = hist[0].argmax()
                    
                    distribution_tile_sparsitywise[bin_idx]['number_of_columns'] += zeros_column.numel()
                    
                    
                    nonzeros_column = tile_size[1] - zeros_column
                    nonzeros_max = nonzeros_column.max()
                    column_nonzero_dist_wrt_max = (nonzeros_max - nonzeros_column).cpu()
                    distribution_tile_sparsitywise[bin_idx]['column_nonzero_dist_wrt_max'].append(column_nonzero_dist_wrt_max)
                    
            l_id += 1
            
            
tile_sparsity = 1
bins_column_nonzero_distances = [k for k in range(0, tile_size[1]+1)]
max_distance_for_given_tile_sparsity = (1.0-distribution_tile_sparsitywise[tile_sparsity]['tile_sparsity'])*tile_size[1]
bins_column_nonzero_distances = [k for k in range(0, int(max_distance_for_given_tile_sparsity)+1)]
column_nonzero_dist_wrt_max_ALL_OUR_postTrain = torch.cat(distribution_tile_sparsitywise[tile_sparsity]['column_nonzero_dist_wrt_max'])
counts_OUR_postTrain, bins_OUR_postTrain, _ = plt.hist(column_nonzero_dist_wrt_max_ALL_OUR_postTrain, bins_column_nonzero_distances, density=True)



####### PLOTTING ######
# SMALL_SIZE = 10
# MEDIUM_SIZE = 34
# BIGGER_SIZE = 38
# plt.rc('font', size=BIGGER_SIZE, weight='bold', family='TimesNewRoman')
# plt.rc('axes', titlesize=BIGGER_SIZE, labelsize=BIGGER_SIZE, labelweight='bold', linewidth=7)
# plt.rc('xtick', labelsize=MEDIUM_SIZE)
# plt.rc('ytick', labelsize=MEDIUM_SIZE)
# plt.rc('legend', fontsize=BIGGER_SIZE)
# plt.rc('figure', titlesize=BIGGER_SIZE)
# ylabel_str = "Fraction of total tile columns\nof tiles with the least sparsity of 50%"
# xlabel_str = "Difference of L0 norm of each column and\nL0 norm of the least sparse column of their corresponding tiles"

# %matplotlib qt
# plt.figure()
# plt.ylim(0, 0.12)
# plt.hist(bins_OUR_postTrain[:-1], bins_OUR_postTrain, weights=counts_OUR_postTrain, alpha=0.5, label='Our method (post-train model)')
# plt.vlines(column_nonzero_dist_wrt_max_ALL_OUR_postTrain.mean().item(), 0.0, counts_OUR_postTrain.max(), color='blue', linewidth=8)

# plt.hist(bins_unstruc[:-1], bins_unstruc, weights=counts_unstruc, alpha=0.5, label='Unstructured pruning')
# plt.vlines(column_nonzero_dist_wrt_max_ALL_unstruc.mean().item(), 0.0, counts_unstruc.max(), color='red', linewidth=8)
# plt.ylabel(ylabel_str)
# plt.xlabel(xlabel_str)
# plt.legend()
# plt.grid(True, 'major', 'y', linewidth=4, alpha=0.4, color='gray')
# #plt.title("Distribution of tile columns based on their L0 distance w.r.t. L0 norm of the least sparse column of their corresponding tiles.")
# plt.savefig('./results/imagenet2012/GatedVarianceOverHoyer/pretrained/unstructuredVSpostTrain_TEST0.png', bbox_inches='tight',pad_inches = 0, dpi=400)




#####
SMALL_SIZE = 60
MEDIUM_SIZE = 70
BIGGER_SIZE = 70
plt.rc('font', size=SMALL_SIZE, weight='bold', family='TimesNewRoman')
plt.rc('axes', titlesize=BIGGER_SIZE, labelpad=20, labelsize=BIGGER_SIZE, labelweight='bold', linewidth=7)
plt.rc('xtick', labelsize=MEDIUM_SIZE)
plt.rc('ytick', labelsize=MEDIUM_SIZE)
plt.rc('xtick.major', pad=15)
plt.rc('ytick.major', pad=15)
plt.rc('legend', fontsize=SMALL_SIZE)
plt.rc('figure', titlesize=BIGGER_SIZE)
ylabel_str = "Fraction of all columns of tiles\nwith the least sparsity of 50%"
xlabel_str = "Difference of L0 norm of each column and\nL0 norm of the LSC of their corresponding tiles"

%matplotlib qt
plt.figure()
plt.axes([0.0, 0.0, 1.1, 1.0])
plt.ylim(0, 0.12)
plt.hist(bins_OUR_postTrain[:-1], bins_OUR_postTrain, weights=counts_OUR_postTrain, alpha=0.5, label='Our method (post-train model)')
plt.vlines(column_nonzero_dist_wrt_max_ALL_OUR_postTrain.mean().item(), 0.0, counts_OUR_postTrain.max(), color='blue', linewidth=8)
plt.text(column_nonzero_dist_wrt_max_ALL_OUR_postTrain.mean().item()-0.1, counts_OUR_postTrain.max()+0.005,'mean: {:.1f}'.format(column_nonzero_dist_wrt_max_ALL_OUR_postTrain.mean().item()),rotation=0, color='blue')

plt.hist(bins_unstruc[:-1], bins_unstruc, weights=counts_unstruc, alpha=0.5, label='Unstructured pruning')
plt.vlines(column_nonzero_dist_wrt_max_ALL_unstruc.mean().item(), 0.0, counts_unstruc.max(), color='red', linewidth=8)
plt.text(column_nonzero_dist_wrt_max_ALL_unstruc.mean().item()-0.1, counts_unstruc.max()+0.005,'mean: {:.1f}'.format(column_nonzero_dist_wrt_max_ALL_unstruc.mean().item()),rotation=0, color='red')

plt.ylabel(ylabel_str)
plt.xlabel(xlabel_str)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.22), ncol=2)
plt.grid(True, 'major', 'y', linewidth=4, alpha=0.4, color='gray')
#plt.title("Distribution of tile columns based on their L0 distance w.r.t. L0 norm of the least sparse column of their corresponding tiles.")
plt.savefig('./results/imagenet2012/GatedVarianceOverHoyer/pretrained/unstructuredVSpostTrain.png', bbox_inches='tight',pad_inches = 0, dpi=400)








#######################################################
######### Our method (post-training post-pruning) ##########
######################################################
from a_Hoyer_and_variance_class import HoyerAndVariance
prune = HoyerAndVariance(model, device)
ADC_res_bits = prune.ADC_res_bits
tile_size = prune.tile_size

bins = [1-(2**(-k)) for k in range(0, ADC_res_bits)]
distribution_tile_sparsitywise = {}
for b_i, b_v in enumerate(bins):
    distribution_tile_sparsitywise[b_i] = {'tile_sparsity': b_v if b_v != 63.5/64.0 else 1.0,
                                           'number_of_columns': 0, 
                                           'column_nonzero_dist_wrt_max': []}
bins.append(1.0)


layerwise_ratios = [60.0, 60.0, 70.0, 55.0, 60.0, 60.0, 60.0, 50.0, 65.0, 65.0, 70.0, 70.0, 85.0, 75.0, 75.0, 65.0, 75.0, 70.0, 75.0, 75.0]
layerwise_ths = []
tilewise_ths = []
l_id = 0
t_id = 0
with torch.no_grad():
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            weight = m.weight.flatten(1)
            w, h = weight.size()
            th = np.percentile(weight.view(-1).abs().clone().detach().cpu(), layerwise_ratios[l_id])
            layerwise_ths.append(th)
            for i in range(0, w, tile_size[0]):
                for j in range(0, h, tile_size[1]):
                    tile = weight[i:(i+tile_size[0]), j:(j+tile_size[1])].abs()
                    zeros_column = (tile <= th).sum(dim=1).float()
                    zeros_column_min = torch.min(zeros_column)
                    zeros_column_per = zeros_column_min/tile.size(1)
                    hist = np.histogram(zeros_column_per.item(), bins)
                    bin_idx = hist[0].argmax()
                    
                    bin_mid = (bins[bin_idx] + bins[bin_idx+1])/2
                    if zeros_column_per.item() >= bin_mid: # forced tile
                        target_percentile_on_each_column = bins[bin_idx+1]*100.0
                        bin_idx = bin_idx + 1
                        if l_id == 1: print("forced")
                    else: # loosened tile
                        target_percentile_on_each_column = bins[bin_idx]*100.0
                        if l_id == 1: print("loosened")
                    highest_th = 0.0
                    for k in range(tile.size(0)):
                        column_th = np.percentile(tile[k].cpu(), target_percentile_on_each_column)
                        if column_th >= highest_th:
                            highest_th = copy.deepcopy(column_th)
                    tilewise_ths.append(highest_th)
                    zeros_column = (tile <= highest_th).sum(dim=1).float()
                    
                    distribution_tile_sparsitywise[bin_idx]['number_of_columns'] += zeros_column.numel()
                    
                    
                    nonzeros_column = tile_size[1] - zeros_column
                    nonzeros_max = nonzeros_column.max()
                    column_nonzero_dist_wrt_max = (nonzeros_max - nonzeros_column).cpu()
                    distribution_tile_sparsitywise[bin_idx]['column_nonzero_dist_wrt_max'].append(column_nonzero_dist_wrt_max)
                    t_id += 1
            l_id += 1
            
            
tile_sparsity = 1
bins_column_nonzero_distances = [k for k in range(0, tile_size[1]+1)]
max_distance_for_given_tile_sparsity = (1.0-distribution_tile_sparsitywise[tile_sparsity]['tile_sparsity'])*tile_size[1]
bins_column_nonzero_distances = [k for k in range(0, int(max_distance_for_given_tile_sparsity)+1)]
column_nonzero_dist_wrt_max_ALL_OUR_postPrune= torch.cat(distribution_tile_sparsitywise[tile_sparsity]['column_nonzero_dist_wrt_max'])
counts_OUR_postPrune, bins_OUR_postPrune, _ = plt.hist(column_nonzero_dist_wrt_max_ALL_OUR_postPrune, bins_column_nonzero_distances, density=True)

# Plotting
plt.figure()
plt.axes([0.0, 0.0, 1.1, 1.0])
plt.ylim(0, 0.12)
plt.hist(bins_OUR_postPrune[:-1], bins_OUR_postPrune, weights=counts_OUR_postPrune, alpha=0.5, label='Our method (post-prune model)')
plt.vlines(column_nonzero_dist_wrt_max_ALL_OUR_postPrune.mean().item(), 0.0, counts_OUR_postPrune.max(), color='blue', linewidth=8)
plt.text(column_nonzero_dist_wrt_max_ALL_OUR_postPrune.mean().item()-0.1, counts_OUR_postPrune.max()+0.005,'mean: {:.1f}'.format(column_nonzero_dist_wrt_max_ALL_OUR_postPrune.mean().item()),rotation=0, color='blue')

plt.hist(bins_unstruc[:-1], bins_unstruc, weights=counts_unstruc, alpha=0.5, label='Unstructured pruning')
plt.vlines(column_nonzero_dist_wrt_max_ALL_unstruc.mean().item(), 0.0, counts_unstruc.max(), color='red', linewidth=8)
plt.text(column_nonzero_dist_wrt_max_ALL_unstruc.mean().item()-0.1, counts_unstruc.max()+0.005,'mean: {:.1f}'.format(column_nonzero_dist_wrt_max_ALL_unstruc.mean().item()),rotation=0, color='red')

plt.ylabel(ylabel_str)
plt.xlabel(xlabel_str)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.22), ncol=2)
plt.grid(True, 'major', 'y', linewidth=4, alpha=0.4, color='gray')
#plt.title("Distribution of tile columns based on their L0 distance w.r.t. L0 norm of the least sparse column of their corresponding tiles.")
plt.savefig('./results/imagenet2012/GatedVarianceOverHoyer/pretrained/unstructuredVSpostPrune.png', bbox_inches='tight',pad_inches = 0, dpi=400)






#######################################################
######### Our method (post-training post-pruning post-finetuning) ##########
######################################################
CKPT_DIR_NEW = '/home/min/a/tibrayev/RESEARCH/active_pruning_for_xbars/results/imagenet2012/GatedVarianceOverHoyer/pretrained/checkpoint_prune_lbl_model_GatedVarianceOverHoyer_lambdas_l2reg0.0001_var_h0.0005_tol0.001.pth'
ckpt = torch.load(CKPT_DIR_NEW, map_location=device)
model.load_state_dict(ckpt['model'])
f.write("Pretrained model was loaded from checkpoint: {}\n".format(CKPT_DIR_NEW))  
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


from a_Hoyer_and_variance_class import HoyerAndVariance
prune = HoyerAndVariance(model, device)
ADC_res_bits = prune.ADC_res_bits
tile_size = prune.tile_size

bins = [1-(2**(-k)) for k in range(0, ADC_res_bits)]
distribution_tile_sparsitywise = {}
for b_i, b_v in enumerate(bins):
    distribution_tile_sparsitywise[b_i] = {'tile_sparsity': b_v if b_v != 63.5/64.0 else 1.0,
                                           'number_of_columns': 0, 
                                           'column_nonzero_dist_wrt_max': []}
bins.append(1.0)



l_id = 0
t_id = 0
with torch.no_grad():
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            weight = m.weight.flatten(1)
            w, h = weight.size()
            for i in range(0, w, tile_size[0]):
                for j in range(0, h, tile_size[1]):
                    tile = weight[i:(i+tile_size[0]), j:(j+tile_size[1])].abs()
                    zeros_column = (tile <= tilewise_ths[t_id]).sum(dim=1).float()
                    zeros_column_min = torch.min(zeros_column)
                    zeros_column_per = zeros_column_min/tile.size(1)
                    hist = np.histogram(zeros_column_per.item(), bins)
                    bin_idx = hist[0].argmax()
                    
                    
                    distribution_tile_sparsitywise[bin_idx]['number_of_columns'] += zeros_column.numel()
                    
                    
                    nonzeros_column = tile_size[1] - zeros_column
                    nonzeros_max = nonzeros_column.max()
                    column_nonzero_dist_wrt_max = (nonzeros_max - nonzeros_column).cpu()
                    distribution_tile_sparsitywise[bin_idx]['column_nonzero_dist_wrt_max'].append(column_nonzero_dist_wrt_max)
                    t_id += 1
            l_id += 1
            
tile_sparsity = 1
bins_column_nonzero_distances = [k for k in range(0, tile_size[1]+1)]
max_distance_for_given_tile_sparsity = (1.0-distribution_tile_sparsitywise[tile_sparsity]['tile_sparsity'])*tile_size[1]
bins_column_nonzero_distances = [k for k in range(0, int(max_distance_for_given_tile_sparsity)+1)]
column_nonzero_dist_wrt_max_ALL_OUR_postTune= torch.cat(distribution_tile_sparsitywise[tile_sparsity]['column_nonzero_dist_wrt_max'])
counts_OUR_postTune, bins_OUR_postTune, _ = plt.hist(column_nonzero_dist_wrt_max_ALL_OUR_postTune, bins_column_nonzero_distances, density=True)
            
            
            





#######################################################
######### Our method (post-training post-pruning post-finetuning ALT) ##########
######################################################
CKPT_DIR_NEW = '/home/min/a/tibrayev/RESEARCH/active_pruning_for_xbars/results/imagenet2012/GatedVarianceOverHoyer/pretrained/checkpoint_prune_lbl_model_GatedVarianceOverHoyer_lambdas_l2reg0.0001_var_h0.0005_tol0.001.pth'
ckpt = torch.load(CKPT_DIR_NEW, map_location=device)
model.load_state_dict(ckpt['model'])
f.write("Pretrained model was loaded from checkpoint: {}\n".format(CKPT_DIR_NEW))  
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


from a_Hoyer_and_variance_class import HoyerAndVariance
prune = HoyerAndVariance(model, device)
ADC_res_bits = prune.ADC_res_bits
tile_size = prune.tile_size

bins = [1-(2**(-k)) for k in range(0, ADC_res_bits)]
distribution_tile_sparsitywise = {}
for b_i, b_v in enumerate(bins):
    distribution_tile_sparsitywise[b_i] = {'tile_sparsity': b_v if b_v != 63.5/64.0 else 1.0,
                                           'number_of_columns': 0, 
                                           'column_nonzero_dist_wrt_max': []}
bins.append(1.0)



l_id = 0
t_id = 0
with torch.no_grad():
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            weight = m.weight.flatten(1)
            w, h = weight.size()
            for i in range(0, w, tile_size[0]):
                for j in range(0, h, tile_size[1]):
                    tile = weight[i:(i+tile_size[0]), j:(j+tile_size[1])].abs()
                    highest_smallest_weight = 0.0
                    for k in range(tile.size(0)):
                        column_min_weight = tile[k][tile[k].nonzero(as_tuple=True)].min()
                        if column_min_weight >= highest_smallest_weight:
                            highest_smallest_weight = copy.deepcopy(column_min_weight)             
                    
                    zeros_column = (tile <= highest_smallest_weight).sum(dim=1).float()
                    zeros_column_min = torch.min(zeros_column)
                    zeros_column_per = zeros_column_min/tile.size(1)
                    hist = np.histogram(zeros_column_per.item(), bins)
                    bin_idx = hist[0].argmax()
                    
                    
                    distribution_tile_sparsitywise[bin_idx]['number_of_columns'] += zeros_column.numel()
                    
                    
                    nonzeros_column = tile_size[1] - zeros_column
                    nonzeros_max = nonzeros_column.max()
                    column_nonzero_dist_wrt_max = (nonzeros_max - nonzeros_column).cpu()
                    distribution_tile_sparsitywise[bin_idx]['column_nonzero_dist_wrt_max'].append(column_nonzero_dist_wrt_max)
                    t_id += 1
            l_id += 1
            
tile_sparsity = 1
bins_column_nonzero_distances = [k for k in range(0, tile_size[1]+1)]
max_distance_for_given_tile_sparsity = (1.0-distribution_tile_sparsitywise[tile_sparsity]['tile_sparsity'])*tile_size[1]
bins_column_nonzero_distances = [k for k in range(0, int(max_distance_for_given_tile_sparsity)+1)]
column_nonzero_dist_wrt_max_ALL_OUR_postTune_NEW = torch.cat(distribution_tile_sparsitywise[tile_sparsity]['column_nonzero_dist_wrt_max'])
counts_OUR_postTune_NEW, bins_OUR_postTune_NEW, _ = plt.hist(column_nonzero_dist_wrt_max_ALL_OUR_postTune_NEW, bins_column_nonzero_distances, density=True)







plt.figure()
plt.axes([0.0, 0.0, 1.1, 1.0])
plt.ylim(0, 0.12)
plt.hist(bins_OUR_postTune_NEW[:-1], bins_OUR_postTune_NEW, weights=counts_OUR_postTune_NEW, alpha=0.5, label='Our method', edgecolor='black', linewidth=8.0)
plt.vlines(column_nonzero_dist_wrt_max_ALL_OUR_postTune_NEW.mean().item(), 0.0, counts_OUR_postTune_NEW.max(), color='blue', linewidth=8)
plt.text(column_nonzero_dist_wrt_max_ALL_OUR_postTune_NEW.mean().item()-0.1, counts_OUR_postTune_NEW.max()+0.001,'mean: {:.1f}'.format(column_nonzero_dist_wrt_max_ALL_OUR_postTune_NEW.mean().item()),rotation=0, color='blue')

plt.hist(bins_unstruc[:-1], bins_unstruc, weights=counts_unstruc, alpha=0.5, label='Unstructured pruning', edgecolor='black', linewidth=8.0)
plt.vlines(column_nonzero_dist_wrt_max_ALL_unstruc.mean().item(), 0.0, counts_unstruc.max(), color='red', linewidth=8)
plt.text(column_nonzero_dist_wrt_max_ALL_unstruc.mean().item()-0.1, counts_unstruc.max()+0.001,'mean: {:.1f}'.format(column_nonzero_dist_wrt_max_ALL_unstruc.mean().item()),rotation=0, color='red')

plt.ylabel(ylabel_str)
plt.xlabel(xlabel_str)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.18), ncol=2, frameon=False)
plt.grid(True, 'major', 'y', linewidth=4, alpha=0.4, color='gray')
#plt.title("Distribution of tile columns based on their L0 distance w.r.t. L0 norm of the least sparse column of their corresponding tiles.")
plt.savefig('./results/imagenet2012/GatedVarianceOverHoyer/pretrained/unstructuredVSpostTune_NEW_alt.png', bbox_inches='tight',pad_inches = 0, dpi=400)
