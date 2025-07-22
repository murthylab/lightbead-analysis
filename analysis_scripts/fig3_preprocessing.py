# -*- coding: utf-8 -*-
"""
Created on Sat Jun 14 16:18:22 2025

@author: wayan
"""

#%% Imports

import pandas as pd 
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import functions as f
from scipy.stats import zscore
import scipy.fft
from pandas import DataFrame
import pickle
from _aux import loadmat_h5
from scipy.io import loadmat
import os
from scipy.fft import rfftfreq
from scipy.signal import convolve

matplotlib.rcParams['axes.spines.right'] = False
matplotlib.rcParams['axes.spines.top'] = False

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42



#%% Define variables

#############################################################################
# To reproduce figures: 
# GO THROUGH THE TODO AND FOLLOW THE INSTRUCTIONS THERE
#############################################################################

list_dic = ['GCaMP8m_06202024_a1_r1_1000.pkl' ,'GCaMP8m_06202024_a1_r2_1000.pkl' ,'GCaMP8m_06192024_a1_r1_1000.pkl'] 
fname = ['06202024_6f_a1_r1_n1000_labels.h5', '06202024_8m_a1_r2_n1000_labels.h5','06192024_8m_a1_r1_n1000_labels.h5'] 

#TODO CHANGE THIS TO THE DESIRED FOLDER containng the pkl files containing the aligned data and the labels
path_dico = 'D:/Wayan/LightBead/method paper/dico data/zscored/60Hz/1000/'
path_labels = 'D:/Wayan/LightBead/method paper/dico data/Labels/zscored/60Hz/' 

#array containing start and end of each stimulus block
start_block_seconds = np.array([5,25,45,55,65,75,85,95,105,115,125,135,145,155])
end_block_seconds = np.array([15,35,47,57,67,77,87,97,107,117,127,137,147,157])



Hz = 60.041
frame_rate = 1/Hz

#TODO adjust booleean if you want to export
export = False


#%% Load the audio stimuli

#TODO CHANGE THIS TO THE DESIRED FOLDER containing plotting version of the stimulus
audio_dir = "C:/Users/wayan.CHRISTAPNI/Documents/Labs/Murthy/LightBead/Code/audio stimuli/Method paper/"
file = loadmat(audio_dir + '3min_pulse_train.mat')
#file = loadmat(audio_dir + '3min_pulse_train_IPI_70.mat')
stim = file['stim']


#%%

#===============================
# Extract based on absolute power
#==============================
cut_off_power = 30   # Extract top 30%
min_dim = 10111
idx_audio_roi = []
HZ = 27.77  # Frequency of interest for an IPI of 36ms
y_lim_fft = [15, 17,7.5,2, 0.6, 0.6]
y_lim_rfft = [0.04, 0.04,0.04,0.01,0.01, 0.01] 
for i, dic in enumerate(list_dic):
    print(i)
    data = pd.read_pickle(path_dico + dic)
    dffs = data['dffs_aligned'][:,:min_dim]
    dffs = zscore(dffs, axis = 1)
    pulse_song = data['pulse_song'][0]
    time_audio = data['time_audio_aligned']
    time_activity = np.arange(frame_rate,(dffs.shape[1]+frame_rate)*frame_rate,frame_rate) 

    ps = [] 
    audio_correlated = []
    mean_roi = np.zeros((dffs.shape[0],2*int(Hz) ))
    # loop though the ROIs
    for roi in range(dffs.shape[0]):
        # Lopp through the blocks and compute mean
        for k in range(2,14):
            # Grab  block
            t_sart = start_block_seconds[k] 
            t_end = end_block_seconds[k] 
            N = int((t_end-t_sart)*Hz)
            t_act = np.linspace(t_sart,t_end,N)
            start_act = (np.abs(t_sart-time_activity)).argmin()
            end_act = (np.abs(t_end-time_activity)).argmin()
            if (end_act-start_act)>(len(t_act)):
                end_act = end_act - ((end_act-start_act)-len(t_act))
      
            # store the block
            if k == 2:
                activity_block = dffs[roi, start_act:end_act]
            else:
                activity_block = np.vstack((activity_block,dffs[roi, start_act:end_act]))    
                
        ## take the mean across all block for each roi
        mean_across_blocks = np.mean(activity_block, axis = 0)
        mean_roi[roi,:] = mean_across_blocks

    overall_mean = np.mean(mean_roi)
    ################# Compute fourier ################
    fourier_fft = np.abs(scipy.fft.fft(mean_roi, norm = 'ortho'))**2
    normalize = int(N/2)+1

    ################# Slice frequency ################
    f_range = [HZ -0.77, HZ +0.33] 
    freq_axis = rfftfreq(N, d = 1/Hz)
    f0 = np.argmin(np.abs(freq_axis- f_range[0]))
    f1 = np.argmin(np.abs(freq_axis- f_range[1]))
    df = freq_axis[1]-freq_axis[0]
    
    ################# compute absolute power in freq band ################
    
    index_cutoff = int(dffs.shape[0] * (cut_off_power/100))
    p_matrix = np.sum(fourier_fft[:,f0:f1],axis=1) 
    rois = np.arange(0,dffs.shape[0])
    rois = rois[~np.isnan(p_matrix)]
    p_matrix = p_matrix[~np.isnan(p_matrix)]
    sorted_indices = np.argsort(p_matrix)
    sorted_ps = p_matrix[sorted_indices]
    sorted_rois = rois[sorted_indices]
    audio_correlated = sorted_rois[-index_cutoff:]
    ps_sorted = sorted_ps[-index_cutoff:]
    idx_audio_roi.append(audio_correlated)        
    if i == 0:
        dffs_all = dffs[audio_correlated,:]
        ps_sort = ps_sorted
    else:    
        dffs_all = np.vstack((dffs_all,dffs[audio_correlated,:]))
        ps_sort = np.vstack((ps_sort,ps_sorted))


######## Keep ROIs with positive correlation coefficient with the stimulus
# Creat stimulus with same number of time points as activity
stim = f.create_stim_train(dffs_all, start_block_seconds,end_block_seconds ,Hz, t_i2c=0)  

# Create kernel
tau_rise = 0.050  # 50 ms rise time
tau_decay = 0.140  # 140 ms decay time
dt = frame_rate  

total_duration = np.round((dffs_all.shape[1]-1)/Hz) 
t = np.arange(0, total_duration, dt)
binary_stim = np.zeros_like(t)
start_first_block = 5  
block_duration = 10  
silence_duration = 10  
num_blocks = 14

for i in range(num_blocks):
    block_start = start_first_block + i * (block_duration + silence_duration)
    block_end = block_start + block_duration
    start_idx = int(block_start / dt)
    end_idx = int(block_end / dt)
    binary_stim[start_idx:end_idx] = 1

# Define GCaMP6f kernel 
kernel_duration = 1.0  
kernel_t = np.arange(0, kernel_duration, dt)
kernel = (1 - np.exp(-kernel_t / tau_rise)) * np.exp(-kernel_t / tau_decay)
kernel /= np.max(kernel)  # Normalize to peak at 1

#Convolve stimulus with kernel
continuous_stim = convolve(binary_stim, kernel, mode='full')[:len(binary_stim)]   

time_activity= np.arange(frame_rate,(dffs_all.shape[1]+frame_rate)*frame_rate,frame_rate)    

cutoff_corr = 99 #5
audio_correlated_corr, coeffs, all_coeffs,sorted_i = f.crosscorr_sort(dffs_all[:], continuous_stim, cutoff_corr ,Hz,0.0) #dffs_all[8100:16200]
    
idx_pos = np.where(coeffs>0)[0].shape[0]



#%% Compute the time it takes the laser to reach each ROI

#================================================================
### load labels for each dffs and compute location: Facing Up
#=================================================================
# Here we rotate the labels 180 counterclockwise and flip around y

n_pixels = 226*216
t_total = 15 # in ms
# Time per pixels
t_pixels = t_total/n_pixels
n_roi = 1000
# define grid with pixel index
for i, dic in enumerate(list_dic[:]):
    print(i)
    labels = loadmat_h5(os.path.join(path_labels, fname[i]))
    l = labels['labels']
    l = l.reshape((226,216,27))
    l = np.flip(np.rot90(l,k=2,axes = (0,1)),axis = 1)
    u = 0
    t_r = np.zeros((dffs.shape[0]))
    # Loop through the slices
    for z in range(l.shape[2]):
        for r in range(n_roi):
            # get ROI coordinates
            y, x = np.where(l[:,:,z]==r)
            center_y = np.mean(y)
            center_x = np.mean(x)
            col = center_x 
            if col == 0 :
                row = center_y
            else:    
                if (col-1)%2 == 0:
                    row = 226 - center_y
                if (col-1)%2 != 0:    
                    row =  center_y
            idx_roi = int( (col-1)*226 ) + row
            # Time to reach the center of the ROI
            t_roi = t_pixels*idx_roi
            t_r[u] = t_roi    
            u+=1

    if  i == 0:
         time_roi= t_r[idx_audio_roi[i]]
         
    else:
        time_roi= np.hstack(( time_roi,t_r[idx_audio_roi[i]] ))
        
        
        
#%% Export

if export == True:
    dic = {'audio_correlated': dffs_all[audio_correlated_corr[-idx_pos:]],'coeffs': coeffs[-idx_pos:],'time_roi':time_roi[sorted_i[-idx_pos:]]}               
      
    path = 'D:/Wayan/LightBead/method paper/clustering/zscored/60Hz/1000/New trial/'     
    with open(path+'dffs_audio_60hz_corr30_30_abs20_180_flip_row'+'.pkl', 'wb') as f:
    
       pickle.dump(dic, f)

        
        