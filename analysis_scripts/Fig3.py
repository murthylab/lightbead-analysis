# -*- coding: utf-8 -*-
"""
Created on Sat Jun 14 16:15:59 2025

@author: wayan
"""

#%% Imports

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib
import functions as f
from scipy.signal import find_peaks
from scipy.fft import rfftfreq
from scipy.io import loadmat
#from sklearn.utils import shuffle
from scipy.stats import sem
from scipy.stats import ttest_ind  
from statsmodels.stats.multitest import multipletests

matplotlib.rcParams['axes.spines.right'] = False
matplotlib.rcParams['axes.spines.top'] = False

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42



#%% Import data

#############################################################################
# To reproduce figures: 
# GO THROUGH THE TODO AND FOLLOW THE INSTRUCTIONS THERE
#############################################################################

#TODO CHANGE THIS TO THE DESIRED FOLDER containing extracted data
path = 'D:/Wayan/LightBead/method paper/clustering/zscored/60Hz/1000/New trial/' 
data_LB = pd.read_pickle(path + 'dffs_audio_60hz_abs30_pos_corr_facing_U_flip_rot180_row'+'.pkl') 

dffs = data_LB['audio_correlated']
time_roi = data_LB['time_roi']

#TODO CHANGE THIS TO THE DESIRED FOLDER containing aligned data
path_dico = 'D:/Wayan/LightBead/method paper/dico data/'
dic = ['GCaMP8m_06202024_a1_r2.pkl']
data = pd.read_pickle(path_dico + dic[0])
pulse_song = data['pulse_song'][0]
sine_song = data['sine_song'][0]
time_audio = data['time_audio_aligned']

#TODO CHANGE THIS TO THE DESIRED FOLDER containing the auditory stimulus
audio_dir = "C:/Users/wayan.CHRISTAPNI/Documents/Labs/Murthy/LightBead/Code/audio stimuli/Method paper/"
file = loadmat(audio_dir + '3min_pulse_train.mat')
stim = file['stim']



#%% Define variables


### Define start and end of stimulus blocks
start_block_seconds = np.array([5,25,45,55,65,75,85,95,105,115,125,135,145,155])
end_block_seconds = np.array([15,35,47,57,67,77,87,97,107,117,127,137,147,157])

# when stimulus is off
start_block_seconds_off = np.array([1,19,39,49,59,69,79,89,99,109,119,129,139,149])
end_block_seconds_off = np.array([3,21,41,51,61,71,81,91,101,111,121,131,141,151])

Hz = 60.041
frame_rate = 1/Hz

time_activity = np.arange(frame_rate,(dffs.shape[1]+frame_rate)*frame_rate,frame_rate) 

#%% Compute mean fourier spectrum and absolute power

### shuffle activity within each blocks only
shuffled_activity = f.shuffle_within_blocks(
    dffs,
    start_block_seconds,
    end_block_seconds,
    Hz,
    seed=44
)


ps_mean , fourier_roi= f.fourier_and_peaks_mean(dffs,start_block_seconds,end_block_seconds,Hz,stim,time_activity,None)
ps_mean = np.array(ps_mean).flatten()

ps_mean_shuffled , fourier_roi_shuffled= f.fourier_and_peaks_mean(shuffled_activity,start_block_seconds,end_block_seconds,Hz,stim,time_activity,None)
ps_mean_shuffled = np.array(ps_mean_shuffled).flatten()

ps_off_mean , fourier_roi_off= f.fourier_and_peaks_mean(dffs,start_block_seconds_off,end_block_seconds_off,Hz,stim,time_activity,None)
ps_off_mean = np.array(ps_off_mean).flatten()


#################################
# Plot absolute power
#################################

f.plot_power_ROIs_all(ps_mean,ps_mean_shuffled,ps_off_mean, None,'all') # black line is when stimulus is off, magenta is when activity is shuffled

##### Compute pvals abs power
t_stat_12, p_val_12 = ttest_ind(ps_mean, ps_mean_shuffled, equal_var = False)
t_stat_13, p_val_13 = ttest_ind(ps_mean, ps_off_mean, equal_var = False)
t_stat_23, p_val_23 = ttest_ind(ps_off_mean, ps_mean_shuffled, equal_var = False)

p_values = [p_val_12, p_val_13, p_val_23]
comparisons = ['on vs shuffled', 'on vs off', 'off vs shuffled']

alpha = 0.05
reject, pvals_corrected, _, _ = multipletests(p_values, alpha=alpha, method='fdr_bh')


#################################
# Plot mean Fourier spectrum
#################################

## When stim is on
N = fourier_roi.shape[1]
normalize = int(N/2)+1 
freq_axis = rfftfreq(N-1, d = 1/Hz)
plt.figure()
plt.plot(freq_axis,np.mean(fourier_roi[:,1:normalize],axis=0), color = 'g')
plt.fill_between(freq_axis,y1=(np.mean(fourier_roi[:,1:normalize],axis=0) + sem(fourier_roi[:,1:normalize],axis=0)),y2=(np.mean(fourier_roi[:,1:normalize],axis=0) - sem(fourier_roi[:,1:normalize],axis=0)),color='g',alpha=0.3)
plt.xlabel('Frequency (Hz)',fontsize =24)
plt.ylabel('Amplitude',fontsize =24)
plt.locator_params(axis='y', nbins=1)
plt.xticks(fontsize =20)
plt.yticks(fontsize =20)
plt.ylim(0,0.15)
plt.tight_layout()


## When activity is shuffled
N = fourier_roi_shuffled.shape[1]
normalize = int(N/2)+1 
freq_axis = rfftfreq(N-1, d = 1/Hz)
plt.figure()
plt.plot(freq_axis,np.mean(fourier_roi_shuffled[:,1:normalize],axis=0), color = 'm')
plt.fill_between(freq_axis,y1=(np.mean(fourier_roi_shuffled[:,1:normalize],axis=0) + sem(fourier_roi_shuffled[:,1:normalize],axis=0)),y2=(np.mean(fourier_roi_shuffled[:,1:normalize],axis=0) - sem(fourier_roi_shuffled[:,1:normalize],axis=0)),color='m',alpha=0.3)
plt.xlabel('Frequency (Hz)',fontsize =24)
plt.ylabel('Amplitude',fontsize =24)
plt.locator_params(axis='y', nbins=1)
plt.xticks(fontsize =20)
plt.yticks(fontsize =20)
plt.ylim(0,0.15)
plt.tight_layout()


## When stimulus is off
N = fourier_roi_off.shape[1]
normalize = int(N/2)+1 
freq_axis = rfftfreq(N-1, d = 1/Hz)
plt.figure()
plt.plot(freq_axis,np.mean(fourier_roi_off[:,1:normalize],axis=0), color='k')
plt.fill_between(freq_axis,y1=(np.mean(fourier_roi_off[:,1:normalize],axis=0) + sem(fourier_roi_off[:,1:normalize],axis=0)),y2=(np.mean(fourier_roi_off[:,1:normalize],axis=0) - sem(fourier_roi_off[:,1:normalize],axis=0)),color='k',alpha=0.3)
plt.xlabel('Frequency (Hz)',fontsize =24)
plt.ylabel('Amplitude',fontsize =24)
plt.locator_params(axis='y', nbins=1)
plt.xticks(fontsize =20)
plt.yticks(fontsize =20)
plt.ylim(0,0.15)
plt.tight_layout()


    
#%% Compute Pulse triggered average    

to_use = dffs
t_to_use = time_roi

### shuffle within each block only
shuffled = f.shuffle_within_blocks(
    dffs,
    start_block_seconds,
    end_block_seconds,
    Hz,
    seed=44
)

to_use_shuffle = shuffled


# Grab the pulse peak times
peaks_stim = find_peaks(stim[0], prominence = 0.7)[0]
t_peaks = peaks_stim/44100

# Truncate the first 2 blocks
idx_start = np.where(t_peaks>40)[0][0]

# Define the time of the responses. Starts at 1/frame rate and we add the sampling time we computed above for each ROIs
resp_ts = np.zeros((to_use.shape[0], to_use.shape[1])) 
for i in range(to_use.shape[0]):
    resp_ts[i,:] = np.arange(frame_rate,(to_use.shape[1]+frame_rate)*frame_rate, frame_rate) + t_to_use[i]/1000
       
# Define the window around each pulse over which to grab the activity
time_window = 18  # in ms

dist = []  # contains the time difference between the pulse time and response time
r = []     # contains the fluorescence values around each pulse
r_shuffle = []
# Loop through the ROIs
for roi in range(to_use.shape[0]):
    for t_peak in t_peaks[idx_start:][::2]:
        # for each peak, grab all the responses within a given window
        # first get the index of the response times within the window around the pulse
        idx_r = np.abs(resp_ts[roi]-t_peak) <= (time_window/1000)
        # grab the times
        t_r = resp_ts[roi,idx_r]
        # loop through these times
        for t in t_r:
            #store the time difference with the pulse time 
            dist.append(t-t_peak)
            # store the fluorescence value at that time
            idx = (t == resp_ts[roi])
            r.append( to_use[roi,idx][0] )
            r_shuffle.append( to_use_shuffle[roi,idx][0] )


# Convert to arrays
R = np.array(r) 
R_shuffle = np.array(r_shuffle) 
R_ts = np.array(dist) 


# Get the sorted indices of the times array
sorted_indices = np.argsort(R_ts)
# Sort both arrays using the sorted indices
R_ts_sorted = R_ts[sorted_indices]
R_sorted = R[sorted_indices]
R_sorted_shuffle = R_shuffle[sorted_indices]

###### Divide into bins and take the mean fluoresence in each bin
bin_edges = np.arange(-time_window*1000, time_window*1000 + 0.0009*1000, 0.0009*1000)
# Digitize the times array into bins
bin_indices = np.digitize(R_ts_sorted*1000, bins=bin_edges)

# Extract fluorescence values for each bin and calculate the mean
bin_means = []
bin_std = []
bin_centers = []
for i in range(1, len(bin_edges)):  # Iterate over bins
    # Find indices of elements in the current bin
    values_in_bin = R_sorted[bin_indices == i]
    if len(values_in_bin) > 0:  # Avoid empty bins
        bin_means.append(np.mean(values_in_bin))
        bin_std.append(sem(values_in_bin))
        bin_centers.append((bin_edges[i - 1] + bin_edges[i]) / 2)  # Center of the bin


# Convert to arrays
bin_means = np.array(bin_means)
bin_std = np.array(bin_std)
bin_centers = np.array(bin_centers)

# Plot the mean fluorescence values
plt.figure(figsize=(6, 5))
plt.plot(bin_centers,bin_means, linestyle='-', color='g', label='Mean Fluorescence')
plt.fill_between(bin_centers,y1=bin_means+bin_std,y2=bin_means-bin_std,color='g',alpha=0.1)
plt.xlabel('Time from pulse (ms)',fontsize =24)
plt.ylabel('Mean Z(DF/F) per 0.9 ms',fontsize =24)

plt.xticks(fontsize =22)
plt.yticks(fontsize =22)
plt.ylim(-0.02,0.06) 
plt.locator_params(axis='y', nbins=4)
plt.tight_layout()

######################################
#### Plot the shuffled PTA
######################################

## Divide into bins and take the mean fluoresence in each bin
bin_edges = np.arange(-time_window*1000, time_window*1000 + 0.0009*1000, 0.0009*1000)
# Digitize the times array into bins
bin_indices = np.digitize(R_ts_sorted*1000, bins=bin_edges)

# Extract fluorescence values for each bin and calculate the mean
bin_means = []
bin_std = []
bin_centers = []
for i in range(1, len(bin_edges)):  # Iterate over bins
    # Find indices of elements in the current bin
    values_in_bin = R_sorted_shuffle[bin_indices == i]
    if len(values_in_bin) > 0:  # Avoid empty bins
        bin_means.append(np.mean(values_in_bin))
        bin_std.append(sem(values_in_bin))
        bin_centers.append((bin_edges[i - 1] + bin_edges[i]) / 2)  # Center of the bin

# Convert to arrays
bin_means = np.array(bin_means)
bin_std = np.array(bin_std)
bin_centers = np.array(bin_centers)

# Plot the mean fluorescence values
plt.figure(figsize=(6, 5))
plt.plot(bin_centers,bin_means, linestyle='-', color='m', label='Mean Fluorescence') 
plt.fill_between(bin_centers,y1=bin_means+bin_std,y2=bin_means-bin_std,color='m',alpha=0.1)
plt.xlabel('Time from pulse (ms)',fontsize =24)
plt.ylabel('Mean Z(DF/F) per 0.9 ms',fontsize =24)

plt.xticks(fontsize =22)
plt.yticks(fontsize =22)
plt.ylim(-0.02,0.06)
plt.locator_params(axis='y', nbins=4)
plt.tight_layout()


