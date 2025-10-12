# -*- coding: utf-8 -*-
"""
Created on Mon Jun 30 15:44:13 2025

@author: wayan
"""

#%% Imports

import pandas as pd 
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import Functions as f
from scipy.stats import zscore
from scipy.io import loadmat
from scipy.signal import convolve



matplotlib.rcParams['axes.spines.right'] = False
matplotlib.rcParams['axes.spines.top'] = False

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


#%% Load data

#############################################################################
# To reproduce figures: 
# GO THROUGH THE TODO AND FOLLOW THE INSTRUCTIONS THERE
#############################################################################

#TODO CHANGE THIS TO THE DESIRED FOLDER containing the audio correlated data
path = 'D:/Wayan/LightBead/method paper/clustering/zscored/supervoxels 2000/' 
data_LB = pd.read_pickle(path + 'dffs_audio_LB_corr_top05_all'+'.pkl') 
data_2p = pd.read_pickle(path + 'dffs_audio_2p_corr_top05_all'+'.pkl') 

dffs_LB = data_LB['audio_correlated']
dffs_2p = data_2p['audio_correlated']

Hz_LB= 28.2893
fr_LB = 1/Hz_LB
time_activity_LB = np.arange(fr_LB,(dffs_LB.shape[1]+fr_LB)*fr_LB,fr_LB) 

Hz_2p = 2.20337115787
fr_2p = 1/Hz_2p
time_activity_2p  = np.arange(fr_2p,(dffs_2p.shape[1]+fr_2p)*fr_2p,fr_2p) 

#TODO CHANGE THIS TO THE DESIRED FOLDER containng the aligned data for LB
path_dico_LB = 'D:/Wayan/LightBead/method paper/dico data/supervoxels_500/'
list_dic = ['GCaMP6f_04032024_a2_r1.pkl']
fname = ['04032024_GCamp6f_a2_r1_w3_n1000_labels.h5']
data = pd.read_pickle(path_dico_LB + list_dic[0])
time_audio_LB = data['time_audio_aligned']
pulse_song = data['pulse_song']
sine_song = data['sine_song']
       
#TODO CHANGE THIS TO THE DESIRED FOLDER containing for 2p
path_dico = 'D:/Wayan/LightBead/method paper/dico data/zscored/RigE/supervoxels_1000/'
list_dic = ['GCaMP6f_12132024_a2_r2.pkl']
fname = ['06212024_6f_a1_r8_n500_labels.h5']
data = pd.read_pickle(path_dico + list_dic[0])
time_audio_2p = data['time_audio_aligned']

dropbox_dir = "C:/Users/wayan.CHRISTAPNI/Documents/Labs/Murthy/LightBead/Code/audio stimuli/Method paper/"
audio_stim = loadmat(dropbox_dir + 'highspeed_pulse_2_WG_paper_forplotting.mat')
time_audio_2p = audio_stim['stim_time'][0]
pulse_song = audio_stim['pulse_song'][0]


start_block_seconds_LB = np.array([5,25,45,65,84,103,123,143,163,183,202.99894,222.99788,242.99788])
end_block_seconds_LB = np.array([15,35,55,75,94,113,133,153,173,192.99894,212.99788,232.99788,252.99788])

start_block_seconds_2p = np.array([5,25,45,65,84,103,123,143,163,183,202.99894,222.99788,242.99788])
end_block_seconds_2p = np.array([15,35,55,75,94,113,133,153,173,192.99894,212.99788,232.99788,252.99788])


#%% 

################################################################
# Plot distribution of peaks and fourier spectrum panels
################################################################

# For LB
peaks_freq_LB = f.peaks_fourier_ROI_combine(zscore(dffs_LB,axis = 1),start_block_seconds_LB,end_block_seconds_LB,0,28.2893,time_activity_LB,'LB',282,1, None)
f.plot_distribution_peaks_fourier_ROIs(peaks_freq_LB,'LB', 'g',None)


# For 2p
peaks_freq_2p = f.peaks_fourier_ROI_combine(zscore(dffs_2p,axis = 1),start_block_seconds_2p,end_block_seconds_2p,2,2.2,time_activity_2p,'2p',282,1, None)
f.plot_distribution_peaks_fourier_ROIs(peaks_freq_2p,'2p', 'm',None)




#%% 

################################################################
# Plot heatmaps and ROI vs thresold panels
################################################################


list_dic = ['GCaMP6f_04032024_a2_r1.pkl' ,'GCaMP6f_04032024_a2_r5.pkl'] 
path_dico_LB = 'D:/Wayan/LightBead/method paper/dico data/zscored/supervoxels_2000/'

# Threshold to extract audio ROIs
cutoff_corr = [6,6.7]
         
min_dim = 7977

# Create kernel
tau_rise = 0.050  # 50 ms rise time
tau_decay = 0.140  # 140 ms decay time
dt = fr_LB  
kernel_duration = 1.0  
kernel_t = np.arange(0, kernel_duration, dt)
kernel = (1 - np.exp(-kernel_t / tau_rise)) * np.exp(-kernel_t / tau_decay)
kernel /= np.max(kernel)  

# Values for ROIs vs threshold panels
threshold_test = np.arange(0.0,1,0.005)

cutoff_05 = 0.5
line = [0.0287,0.043]



for i, dic in enumerate(list_dic):
    data = pd.read_pickle(path_dico_LB + dic)
    print('Run:', dic)
    
    ####################################################
    ### Extract auditory correlated ROIs and plot heatmap
    ####################################################
    
    dffs = data['dffs_aligned'][:,:min_dim] 
    time_audio = data['time_audio_aligned']
    pulse_song = data['pulse_song']

    time_activity= np.arange(fr_LB,(dffs.shape[1]+fr_LB)*fr_LB,fr_LB)  

    stim = f.create_stim(dffs, start_block_seconds_LB,end_block_seconds_LB,Hz_LB, t_i2c=0)
    #Convolve stimulus with kernel
    continuous_stim = convolve(stim, kernel, mode='full')[:len(stim)] 
    time_filter = np.arange(0,len(stim))/Hz_LB 
    conv = np.convolve (stim, kernel, mode = 'same')
    conv = conv/np.max(conv) 
    
    #Extract audio correlated ROIs
    audio_correlated, coeffs, all_coeffs, sort_i = f.crosscorr_sort(zscore(dffs, axis = 1), conv, cutoff_corr[i] ,Hz_LB,0.0)

    cut_off_audio = int(265*Hz_LB)
    to_plot = np.flip(zscore(dffs[audio_correlated,:cut_off_audio], axis = 1),axis = 0)
    
    plt.figure(figsize = (5,5))
    im = plt.imshow(to_plot, aspect = 'auto', vmin = -1.0, vmax = 1.0,cmap = 'viridis',extent = [0,265,0,3617])
    plt.xticks(fontsize = 18)
    plt.yticks([])
    plt.fill_between(time_audio,y1=pulse_song+3676, y2=pulse_song+3712,where =pulse_song>0,color='r',alpha=1)
    plt.tight_layout()
    plt.xlabel('Time (s)',fontsize = 18)
    plt.ylabel('ROIs',fontsize = 18)
    plt.tight_layout()
    
    ####################################################
    ### Plot ROIs vs threshold
    ####################################################
    n_roi, c = f.crosscorr_sort_corr(zscore(dffs, axis = 1), conv, threshold_test,cutoff_05 ,Hz_LB)

    plt.figure(figsize = (15,10))
    plt.plot(threshold_test,np.array(n_roi)/1000,color = 'k', lw = 3.5)
    plt.xlabel('Correlation coefficient',fontsize = 22)
    plt.ylabel('# ROIs',fontsize = 22)
    plt.xticks(fontsize = 22)
    plt.yticks(fontsize = 22)
    plt.tight_layout()
    plt.axvline(x=line[i], color = 'r', linestyle = '--', alpha = 1,lw = 2.5)
    plt.axvline(x=c[0], color = 'k', linestyle = '--', alpha = 1,lw = 2.5)
    
    # Plot zoom in panel
    plt.figure(figsize = (15,3))
    plt.plot(threshold_test,np.array(n_roi)/1000, color = 'k',lw = 3.5)
    plt.xlabel('Correlation coefficient',fontsize = 22)
    plt.ylabel('# ROIs',fontsize = 22)
    plt.xlim(0,0.2)
    if i == 0:
        plt.ylim(0,7)
    else:   
        plt.ylim(0,15)
    plt.axvline(x=line[i], color = 'r', linestyle = '--', alpha = 1,lw = 2.5)
    plt.axvline(x=c[0], color = 'k', linestyle = '--', alpha = 1,lw = 2.5)
    plt.xticks(fontsize = 22)
    plt.yticks(fontsize = 22)
    plt.locator_params(axis='y', nbins=3)
    plt.tight_layout()   
 
    

    
#%% 

################################################################
# Extract ROIs by comparing to a null distribution
################################################################

## Load data
list_dic = ['GCaMP6f_04032024_a2_r1.pkl' ,'GCaMP6f_04032024_a2_r5.pkl'] 
path_dico_LB = 'D:/Wayan/LightBead/method paper/dico data/zscored/supervoxels_2000/'

n_shuffle = 500

for i, dic in enumerate(list_dic[:1]):

    data = pd.read_pickle(path_dico_LB + dic)
    print('Run:', dic)
    
    dffs = data['dffs_aligned'][:,:min_dim] 
    time_audio = data['time_audio_aligned']
    pulse_song = data['pulse_song']
    
    time_activity= np.arange(fr_LB,(dffs.shape[1]+fr_LB)*fr_LB,fr_LB)  
    
    
    #1) z score dffs and the stimulus
    dffs_z = zscore(dffs,axis = 1)
    
    stim = f.create_stim(dffs_z, start_block_seconds_LB,end_block_seconds_LB,Hz_LB, t_i2c=0)
    # Create kernel
    tau_rise = 0.050  # 50 ms rise time
    tau_decay = 0.140  # 140 ms decay time
    dt = fr_LB  
    kernel_duration = 1.0  
    kernel_t = np.arange(0, kernel_duration, dt)
    kernel = (1 - np.exp(-kernel_t / tau_rise)) * np.exp(-kernel_t / tau_decay)
    kernel /= np.max(kernel)  
    #Convolve stimulus with kernel
    continuous_stim = convolve(stim, kernel, mode='full')[:len(stim)] 
    time_filter = np.arange(0,len(stim))/Hz_LB 
    conv = np.convolve (stim, kernel, mode = 'same')
    #conv = conv/np.max(conv)
    conv_z = zscore(conv) 
    
    #2) compute correlation coefficient of each ROI with the stimulus
    audio_correlated, coeffs, all_coeffs, sort_i = f.crosscorr_sort(dffs_z, conv_z, 100 ,Hz_LB,0.0)
    
    
    
    #3) Randomly shuffle each ROI N times and get correlation coefficient for each shuffle
    
    
    
    coeffs_shuffle_all = np.zeros((dffs_z.shape[0],n_shuffle))
    
    for i in range(n_shuffle):
        # shuffle all rows
        d_shuffled = shuffle_rows_numpy(dffs_z, seed=i, key_dtype=np.float32)
        # extract correlation coefficient
        audio_correlated_shuffle, coeffs_shuffle, all_coeffs_shuffle, sort_i_shuffle = f.crosscorr_sort(d_shuffled, conv_z, 100 ,Hz_LB,0.0)
        # store coefficients
        coeffs_shuffle_all[:,i] = all_coeffs_shuffle
    
    all_coeffs_shuffle.shape    
    #4)  Compute p values
    pvals = permutation_pvals_two_sided(all_coeffs, coeffs_shuffle_all)


#5) Compute FDR
sig_mask, pvals_bh = fdr_bh(pvals, q=0.05)
np.sum(sig_mask)
np.sum(pvals<0.05)
np.min(pvals)





# 6) plot the extracted ROIs

# First we sort them bz coeff
audio_correlated_o, coeffs_o, all_coeffs_o, sort_i_o = f.crosscorr_sort(dffs_z, conv_z, 0.5 ,Hz_LB,0.0)
audio_correlated, coeffs, all_coeffs, sort_i = f.crosscorr_sort(dffs_z[pvals<0.05,:], conv_z, 100 ,Hz_LB,0.0)


to_plot = np.flip(zscore(dffs_z[pvals<0.05,:][audio_correlated,:], axis = 1),axis = 0)

plt.figure(figsize = (5,5))
im = plt.imshow(to_plot, aspect = 'auto', vmin = -1.0, vmax = 1.0,cmap = 'viridis')
plt.xticks(fontsize = 18)
plt.yticks([])
#plt.fill_between(time_audio,y1=pulse_song+3676, y2=pulse_song+3712,where =pulse_song>0,color='r',alpha=1)
plt.tight_layout()
plt.xlabel('Time (s)',fontsize = 18)
plt.ylabel('ROIs',fontsize = 18)
plt.tight_layout()

roiss = np.arange(0,54000)
roiss[pvals<0.05]


count = 0
for i, roi in enumerate(sort_i_o):
    if roi in roiss[pvals<0.05]:
        count +=1

sort_i[-1]
sort_i_o[-1]

plt.figure()
plt.plot(dffs_z[pvals<0.05,:][audio_correlated[-1],:])

def shuffle_rows_numpy(X, seed=0, key_dtype=np.float32):
    rng  = np.random.default_rng(seed)
    keys = rng.random(X.shape, dtype=key_dtype)         # ~ size(X)
    idx  = np.argsort(keys, axis=1, kind='quicksort')   # int64 indices (~2× size(X) if X is float32)
    del keys                                             # free keys before the gather
    Xsh  = np.take_along_axis(X, idx, axis=1)           # new array same dtype/shape as X
    return Xsh



plt.figure(figsize = (15,5))
plt.plot(dffs_z[0,:], color = 'k')

plt.figure(figsize = (15,5))
plt.plot(d_shuffled[0,:], color = 'b')


def permutation_pvals_two_sided(r_obs, r_null):
    """
    Compute two-sided permutation p-values:
      p_i = (1 + #{ |r_null| >= |r_obs_i| }) / (N_i + 1)

    Parameters
    ----------
    r_obs : array-like, shape (n_rois,)
        Observed correlations per ROI.
    r_null : array-like, shape (n_rois, n_perm)  OR  (n_perm,)
        Null correlations from shuffles. If 2D, each row corresponds to that ROI's null.
        If 1D, a pooled null used for all ROIs.

    Returns
    -------
    pvals : ndarray, shape (n_rois,)
        Two-sided permutation p-values for each ROI.
    """
    r_obs = np.asarray(r_obs, dtype=np.float64).reshape(-1)
    abs_obs = np.abs(r_obs)

    r_null = np.asarray(r_null, dtype=np.float64)

    if r_null.ndim == 1:
        # Pooled null, broadcast to all ROIs
        abs_null = np.abs(r_null)[None, :]                     # (1, n_perm)
        valid = ~np.isnan(abs_null)                            # (1, n_perm)
        N = valid.sum(axis=1).astype(np.int64)                 # (1,)
        ge = (abs_null >= abs_obs[:, None]) & valid            # (n_rois, n_perm)
        counts = ge.sum(axis=1)                                # (n_rois,)
        pvals = (1.0 + counts) / (N[0] + 1.0)                  # scalar N for all ROIs
        return pvals

    elif r_null.ndim == 2:
        if r_null.shape[0] != r_obs.shape[0]:
            raise ValueError("For per-ROI nulls, r_null must have shape (n_rois, n_perm).")
        abs_null = np.abs(r_null)                              # (n_rois, n_perm)
        valid = ~np.isnan(abs_null)                            # (n_rois, n_perm)
        N = valid.sum(axis=1).astype(np.int64)                 # (n_rois,)
        # Compare each ROI's |obs| to its own null distribution
        ge = (abs_null >= abs_obs[:, None]) & valid            # (n_rois, n_perm)
        counts = ge.sum(axis=1)                                # (n_rois,)
        pvals = (1.0 + counts) / (N + 1.0)                     # (n_rois,)
        return pvals

    else:
        raise ValueError("r_null must be 1D (pooled) or 2D (per-ROI).")
        
        
def fdr_bh(pvals, q=0.05):
    """
    Benjamini-Hochberg FDR control.
    Returns a boolean mask of discoveries and the BH-adjusted p-values.
    """
    pvals = np.asarray(pvals, dtype=np.float64)
    n = pvals.size
    order = np.argsort(pvals)
    ranked = pvals[order]
    thresh = q * (np.arange(1, n + 1) / n)
    is_sig = ranked <= thresh
    if np.any(is_sig):
        k = np.max(np.where(is_sig)[0])
        cutoff = ranked[k]
        discoveries = pvals <= cutoff
    else:
        discoveries = np.zeros_like(pvals, dtype=bool)

    # Adjusted p-values
    adj = np.empty_like(ranked)
    # monotone step-up
    adj[-1] = ranked[-1] * n / n
    for i in range(n - 2, -1, -1):
        adj[i] = min(adj[i + 1], ranked[i] * n / (i + 1))
    adj_p = np.empty_like(adj)
    adj_p[order] = adj
    return discoveries, adj_p        