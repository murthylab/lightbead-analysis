# -*- coding: utf-8 -*-
"""
This script extracts the stimulus correlated ROIs and store them in pkl files for later analyses
It also plots panel B of Fig2
"""

#%% Imports
import pandas as pd 
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import Functions as f
import pickle
from _aux import loadmat_h5
import os
from scipy.signal import convolve
from scipy.stats import zscore

matplotlib.rcParams['axes.spines.right'] = False
matplotlib.rcParams['axes.spines.top'] = False

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42



#%% Define variables


#############################################################################
# To reproduce figures: 
# GO THROUGH THE TODO AND FOLLOW THE INSTRUCTIONS THERE
#############################################################################

#TODO sepcify the scope, 'LB' or '2p'
scope = 'LB'

#TODO sepcify if data will be exported or not
export = False

if scope == 'LB':
    list_dic = ['GCaMP6f_04032024_a2_r1.pkl' ,'GCaMP6f_04032024_a2_r5.pkl' ,'GCaMP6f_04162024_a1_r1.pkl','GCaMP6f_04192024_a1_r2.pkl', 'GCaMP6f_04192024_a1_r6.pkl', 'GCaMP6f_04192024_a1_r9.pkl'] 
    fname = ['04032024_6f_a2_r1_n2000_labels.h5', '04032024_6f_a2_r5_n2000_labels.h5','04162024_6f_a1_r1_n2000_labels.h5', '04192024_6f_a1_r2_n2000_labels.h5', '04192024_6f_a1_r6_n2000_labels.h5', '04192024_6f_a1_r9_n2000_labels.h5'] 
    min_dim = 7977
    Hz, Hz_target = 28.2893, 28.2893  
    frame_rate = 1/Hz
    name_export = 'dffs_audio_LB_corr_top05_all'
    
    depth_a1 =np.arange(275, 5,-10)
    depth_a2 =np.arange(220,-50,-10)
    depth_a3 =np.arange(275, 5,-10)
    list_depth = [depth_a1,depth_a1,depth_a2,depth_a3,depth_a3,depth_a3]
    
    #TODO CHANGE THIS TO THE DESIRED FOLDER containng the pkl files containing the aligned data and the labels
    path_dico = 'D:/Wayan/LightBead/method paper/dico data/zscored/supervoxels_2000/'
    path_labels = 'D:/Wayan/LightBead/method paper/dico data/Labels/zscored/2000/' 
    
             
if scope == '2p':
    list_dic = ['GCaMP6f_12132024_a2_r2.pkl', 'GCaMP6f_12132024_a2_r3.pkl','GCaMP6f_12132024_a2_r4.pkl','GCaMP6f_12202024_a1_r2.pkl']
    fname = ['12132024_6f_a2_r2_n1000_labels.h5','12132024_6f_a2_r3_n1000_labels.h5','12132024_6f_a2_r4_n1000_labels.h5','12202024_6f_a1_r2_n1000_labels.h5']       
    min_dim = 668
    Hz = 2.20337115787 
    Hz_target = 28.2893 
    frame_rate = 1/Hz
    name_export = 'dffs_audio_2p_corr_top05_all'

    #TODO CHANGE THIS TO THE DESIRED FOLDER containng the pkl files containing the aligned data and the labels
    path_dico = 'D:/Wayan/LightBead/method paper/dico data/zscored/RigE/supervoxels_1000/'
    path_labels = 'D:/Wayan/LightBead/method paper/dico data/Labels/zscored/RigE/1000/'
             

# array containing the start and end of each stimulus block
start_block_seconds = np.array([5,25,45,65,84,103,123,143,163,183,202.99894,222.99788,242.99788])
end_block_seconds = np.array([15,35,55,75,94,113,133,153,173,192.99894,212.99788,232.99788,252.99788])
    

# threshold to extract auditory correlated ROIs    
cutoff_corr = 0.5


#%% Merge dffs

###################################################################
# Store each dffs for each trials and animals
###################################################################

for i, dic in enumerate(list_dic):
    data = pd.read_pickle(path_dico + dic)
    print('Run:', dic)
    
    if scope == 'LB':
        dffs = data['dffs_aligned'][:,:min_dim]
        time_audio = data['time_audio_aligned']
        pulse_song = data['pulse_song']
        labels = loadmat_h5(os.path.join(path_labels, fname[i]))
        l = labels['labels']
        l = l.reshape((226,512,27))
        
        depth_rois = assign_depths(dffs.shape[0], list_depth[i])

    if scope == '2p':
        dffs = data['dffs_corrected'][:,:min_dim]
        time_audio = data['time_audio_aligned']
        pulse_song = data['pulse_song']
        sine_song = np.zeros((data['pulse_song'].shape[0]))
        labels = loadmat_h5(os.path.join(path_labels, fname[i]))
        l = labels['labels']
        l = l.reshape((256,128,47))

    if i == 0:
        dffs_all = dffs
        depth_rois_all = depth_rois
    else:    
        dffs_all = np.vstack((dffs_all,dffs))
        depth_rois_all = np.hstack((depth_rois_all,depth_rois))
        

        
#%%  Extract depth

def assign_depths(n_rois: int, slice_depths: np.ndarray) -> np.ndarray:
    """
    Assigns depths to each ROI given the total number of ROIs and slice depths.

    Parameters
    ----------
    n_rois : int
        Total number of ROIs (number of rows in your 2D array).
    slice_depths : np.ndarray
        1D array of shape (n_slices,) containing the depth for each slice.

    Returns
    -------
    roi_depths : np.ndarray
        1D array of shape (n_rois,) where each entry is the depth of the slice
        corresponding to that ROI.
    """
    n_slices = len(slice_depths)
    rois_per_slice = n_rois // n_slices  # assumes equal number of ROIs per slice
    
    if n_rois % n_slices != 0:
        raise ValueError("Number of ROIs is not evenly divisible by number of slices.")

    # Repeat each slice depth rois_per_slice times
    roi_depths = np.repeat(slice_depths, rois_per_slice)
    
    return roi_depths

depth_a1 =np.arange(275, 5,-10)
depth_a2 =np.arange(220,-50,-10)
depth_a3 =np.arange(275, 5,-10)
list_depth = [depth_a1,depth_a1,depth_a2,depth_a3,depth_a3,depth_a3]

for i, dic in enumerate(list_depth[:1]):
    depth_rois = assign_depths(dffs.shape[0], list_depth[i])
        
#%% Extract audio correlated ROIs   
     
# Creat stimulus with same number of time points as activity
stim = f.create_stim(dffs_all, start_block_seconds,end_block_seconds,Hz, t_i2c=0)  

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
num_blocks = 13

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
#continuous_stim = convolve(binary_stim, kernel, mode='full')[:len(binary_stim)]   
continuous_stim = convolve(stim, kernel, mode='full')[:len(stim)]   


time_activity= np.arange(frame_rate,(dffs_all.shape[1]+frame_rate)*frame_rate,frame_rate)    

# Extract the top 0.5% of ROIs with highest correlation coeficient with the stimulus
audio_correlated, coeffs, all_coeffs,sorted_indices = f.crosscorr_sort(dffs_all, continuous_stim, cutoff_corr ,Hz,0.0)

dffs_all.shape, continuous_stim.shape
# Plot the mean
plt.figure(figsize = (15,5))       
plt.plot(time_activity, np.mean(dffs_all[audio_correlated,:],axis=0))    
max_trace = np.max(np.mean(dffs_all[audio_correlated,:],axis=0))
plt.fill_between(time_audio,y1=max_trace*pulse_song,y2=(max_trace*pulse_song)+0.02,where =pulse_song>0,color='r',alpha=0.5)
plt.fill_between(time_audio,y1=max_trace*sine_song,y2=(max_trace*sine_song)+0.02,where =sine_song>0,color='b',alpha=0.5)
plt.title('Mean activity {}'.format(dic)) 
plt.xlabel('Time (s)')
plt.ylabel('DF/F')


#%% Export audio correlated ROIs

if export == True:
    dic = {'audio_correlated': dffs_all[audio_correlated,:]}
     
     #TODO CHANGE THIS TO THE DESIRED FOLDER
    path = 'D:/Wayan/LightBead/method paper/clustering/zscored/supervoxels 2000/'     
    with open(path + name_export +'.pkl', 'wb') as f:
    
       pickle.dump(dic, f)
   


#%% Plot distribution of correlation coefficient

# sort the array

all_coeffs_sorted = np.copy(all_coeffs)
sorted_indices = np.argsort(all_coeffs_sorted)
all_coeffs_sorted = all_coeffs_sorted[sorted_indices]

if scope == 'LB':
    col = 'g.'
else:
    col = 'm.'    

plt.figure()
plt.plot(np.random.normal(0.5,0.005, size = len(all_coeffs)),all_coeffs_sorted,col,alpha = 0.3)
plt.axhline(coeffs[0], linestyle ='--', color = 'k')
plt.xticks([])
plt.yticks(color = 'k')
plt.ylabel('Correlation coefficient',fontsize =24)
plt.ylim(-0.2,0.85)
plt.xlim(0.45, 0.55)
plt.yticks(fontsize =22)
plt.tight_layout()


path_fig = 'C:/Users/wayan.CHRISTAPNI/Princeton Dropbox/Wayan Gauthey/Princeton/Lightbead/Method paper/Figures/Panels/Figure 2/14012025/'
plt.savefig(path_fig + 'Corr_coeff_2p.png', transparent = True)  




#%% Plot correlation coefficient vs depth
all_coeffs_sorted = np.copy(all_coeffs)
sorted_indices_coeff = np.argsort(all_coeffs_sorted)
all_coeffs_sorted = all_coeffs_sorted[sorted_indices_coeff]
all_depths_sorted = depth_rois_all[sorted_indices_coeff]

plt.figure(figsize = (10,10))
plt.scatter(all_depths_sorted, all_coeffs_sorted)

## Only top 0.5
depths_top05 = depth_rois_all[sorted_indices]
depths_top05[np.argsort(depths_top05)]

plt.figure(figsize = (10,10))
plt.scatter(depths_top05, coeffs)
plt.xlabel('Depth (um)', fontsize = 16)
plt.ylabel('Corr coeff', fontsize = 16)
plt.yticks(fontsize = 16)
plt.xticks(fontsize = 16)

all_depths_sorted.shape, coeffs.shape

sorted_indices_depth_top05 = sorted_indices[np.argsort(depths_top05)]

to_plot_LB = np.flip(zscore(dffs_all[sorted_indices_depth_top05],axis = 1),axis = 0)
cmap_base = 'viridis' #gnuplot
vmin, vmax = -0.4, 1.1  # -0.8, 1
cmap = f.truncate_colormap(cmap_base, vmin, vmax)

plt.figure(figsize = (4.7,5.3)) #(4,5)
im = plt.imshow(to_plot_LB, aspect = 'auto', vmin = -1, vmax = 1,cmap = cmap, extent = [0.035,258,0,1700])   
plt.tight_layout()
plt.colorbar(im)
plt.yticks([])
#plt.fill_between(time_audio_LB,y1=pulse_song+1725, y2=pulse_song +1745,where =pulse_song>0,color='r',alpha=1)
plt.xlabel('Time (s)')
plt.tight_layout()

