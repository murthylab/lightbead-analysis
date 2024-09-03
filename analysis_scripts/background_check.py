# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 13:55:13 2024

@author: wayan
"""

import pandas as pd
import numpy as np
import functions as f
import matplotlib.pyplot as plt
import pickle
import matplotlib

matplotlib.rcParams['axes.spines.right'] = False
matplotlib.rcParams['axes.spines.top'] = False

dic = ['GCaMP6f_06212024_a1_r2.pkl'] # sine pulse
    
path_dico = 'D:/Wayan/LightBead/method paper/dico data/'


scope = '2p'



data = pd.read_pickle(path_dico + dic[0])

if scope == 'LB':
    dffs_aligned = data['dffs_aligned'][:,:] #:,:min_dim
    dffs_raw= data['dffs_raw']
    time_audio_aligned = data['time_audio_aligned']
    time_audio_raw = data['time_audio_raw']
    time_activity_raw = ['time_activity_raw']
    time_activity_aligned = data['time_activity_aligned']
    n_volumes_raw = data['n_volumes_raw']
    n_volumes_aligned = data['n_volumes_aligned']
    
if scope == '2p':
    dffs = data['dffs'][:,:] #:,:min_dim
    time_audio = data['time_audio']
    time_activity = data['time_activity']
    n_volumes = data['n_volumes']
 
 
dffs_detrend = data['dffs_corrected']
sine_song = data['sine_song']
pulse_song = data['pulse_song']
stim_type = data['stimulus'] 
t_i2c = data['t_i2c']
time_start_audio= data['time_start_audio']
Scope = data['Scope']
Frame_rate = data['Frame_rate']



################################################################################
##### Plot mean activity before background correction
################################################################################
#Extract audio correlated ROIs
audio_correlated_before, mean_on = f.filter_threshold(dffs,0.3, 'ON', pulse_song, sine_song, time_audio, time_activity)  
# Plot heatmaps of extracted ROIs
sort = f.sorted_heatmap_audio(dffs,audio_correlated_before,dic, -0.6,0.6,'viridis')
#Plot mean activity of each run
plt.figure(figsize = (15,5))       
plt.plot(time_activity, np.mean(dffs[audio_correlated_before,:],axis=0))    
max_trace = np.max(np.mean(dffs[audio_correlated_before,:],axis=0))
plt.fill_between(time_audio,y1=max_trace*pulse_song,y2=(max_trace*pulse_song)+0.02,where =pulse_song>0,color='r',alpha=0.5)
plt.fill_between(time_audio,y1=max_trace*sine_song,y2=(max_trace*sine_song)+0.02,where =sine_song>0,color='b',alpha=0.5)
plt.title('Mean activity {}'.format(dic))




################################################################################
##### Plot mean activity after background correction
################################################################################
#Extract audio correlated ROIs
audio_correlated_after, mean_after = f.filter_threshold(dffs_detrend,0.3, 'ON', pulse_song, sine_song, time_audio, time_activity)  
# Plot heatmaps of extracted ROIs
sort = f.sorted_heatmap_audio(dffs_detrend,audio_correlated_after,dic, -0.2,0.4,'viridis')
#Plot mean activity of each run
plt.figure(figsize = (15,5))       
plt.plot(time_activity, np.mean(dffs_detrend[audio_correlated_after,:],axis=0))    
max_trace = np.max(np.mean(dffs_detrend[audio_correlated_after,:],axis=0))
plt.fill_between(time_audio,y1=max_trace*pulse_song,y2=(max_trace*pulse_song)+0.004,where =pulse_song>0,color='r',alpha=0.5)
plt.fill_between(time_audio,y1=max_trace*sine_song,y2=(max_trace*sine_song)+0.004,where =sine_song>0,color='b',alpha=0.5)
plt.title('Mean activity after background correction {}'.format(dic))


