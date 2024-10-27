# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 16:34:38 2024

@author: wayan
"""


import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from _aux import loadmat_h5, align_t, get_outliers
import pandas as pd
import os
from scipy.io import loadmat
import h5py
from pandas import DataFrame
from scipy.signal import find_peaks
import scipy.fft
from scipy.fft import rfft, rfftfreq
import pickle
from sklearn.decomposition import PCA
from scipy.stats import pearsonr, zscore
from scipy.ndimage import gaussian_filter1d
import functions as f

matplotlib.rcParams['axes.spines.right'] = False
matplotlib.rcParams['axes.spines.top'] = False






# %%
################################
# Define variable for fictrac
################################

idx_frame = 1
idx_time = 21
idx_speed = 18 
DT_BHV = 1


# %%
##########################
# Import fictrac data
##########################


data_dir = "Z:/Wayan/Lightbead/Flies/Method paper/04032024/Fictrac/a2_r5/"


time_speed_smooth = []  
speed_smooth = []  

fname_bhv = data_dir + '/output.txt'

df_bhv = pd.read_csv(fname_bhv, header=None)  # behavior dataframe
time_speed = np.array(df_bhv)[:, idx_time] - np.array(df_bhv)[0, idx_time]
speed = np.array(df_bhv)[:, idx_speed].astype(float)

time_speed_smooth.append(time_speed)
speed_smooth.append(speed)

time_speed_smooth, speed_smooth = align_t(time_speed_smooth, speed_smooth, dt=DT_BHV)


speed_smooth = speed_smooth[0]  

print('time_speed shape: {}, speed shape: {}'.format(time_speed.shape, speed.shape) )
print('time_speed shape: {}, speed shape: {}'.format(time_speed_smooth.shape, speed_smooth.shape) )


# %%
##########################
# Plot behavior
##########################

# raw speed
fig, axs = plt.subplots(2,1, figsize = (15,5))

axs[0].plot(time_speed,speed, color = 'black')
axs[0].set_xlabel('Time (s)')
axs[0].set_ylabel('Walking speed')
axs[0].set_title('Raw speed')

# Smoothen using Rich's method
axs[1].plot(time_speed_smooth,speed_smooth, color = 'black')
axs[1].set_xlabel('Time (s)')
axs[1].set_ylabel('Walking speed')
axs[1].set_title('Smooth speed')

plt.tight_layout()






#####################################
# Extract locomotion correlated ROI
#####################################
Hz = 28
coeff = 0.15
loco_correlated = []   # contains the locomotion correlated ROIs
offset_all=[]          # contains the average lag that gave the best correlation
close_corr=[]
neg_offset = []
offset_neg =[]
for roi in range(dffs.shape[0]):
    d = {'speed': speed_smooth/np.nanmax(speed_smooth), 'activity': dffs[roi,:]}
    df = pd.DataFrame(data=d)
    s = df['speed']
    a = df['activity']
    
    seconds = int(10*Hz)
    x= np.arange(-seconds,seconds+Hz, int(Hz/2))
    corr = [f.crosscorr(s,a, lag) for lag in range(-seconds,seconds+Hz,Hz)]
    offset = np.floor(len(corr)/2)-np.argmax(corr)
    
    if np.max(corr)>coeff:
        loco_correlated.append(roi)
        offset_all.append(offset)
        if 0<=offset<2:
            close_corr.append(roi)
        if -4<offset<0:
            neg_offset.append(roi)
            offset_neg.append(offset)
        
print('number of locomotion correlated ROIs: {}'.format(len(loco_correlated)))
print('average lag {}'.format(np.mean(offset_all)))
print('number of locomotion correlated ROIs close to motion: {}'.format(len(close_corr)))
print('number of ROIs with negative offset: {}'.format(len(neg_offset)))
