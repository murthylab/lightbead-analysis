# -*- coding: utf-8 -*-
"""
This script generates all panels for Fig.2
"""

#%% Imports

import pandas as pd 
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import functions as f
from scipy.stats import zscore
from scipy.io import loadmat

matplotlib.rcParams['axes.spines.right'] = False
matplotlib.rcParams['axes.spines.top'] = False

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


#%% Load data and define variables

#############################################################################
# To reproduce figures: 
# GO THROUGH THE TODO AND FOLLOW THE INSTRUCTIONS THERE
#############################################################################

#TODO CHANGE THIS TO THE DESIRED FOLDER contaning the data with audio correlated ROIs
path = 'D:/Wayan/LightBead/method paper/clustering/zscored/supervoxels 2000/' 
data_LB = pd.read_pickle(path + 'dffs_audio_LB_corr_top05_all'+'.pkl') 
data_2p = pd.read_pickle(path + 'dffs_audio_2p_corr_top05_all'+'.pkl')  

dffs_LB = data_LB['audio_correlated'][:,1:]
dffs_2p = data_2p['audio_correlated'][:,1:]

Hz_LB = 28.2893
fr_LB = 1/Hz_LB 
time_activity_LB = np.arange(fr_LB,(dffs_LB.shape[1]+fr_LB)*fr_LB,fr_LB) 

Hz_2p = 2.20337115787
fr_2p = 1/Hz_2p
time_activity_2p  = np.arange(fr_2p,(dffs_2p.shape[1]+fr_2p)*fr_2p,fr_2p) 

#TODO CHANGE THIS TO THE DESIRED FOLDER contaning the aligned LBM data
path_dico = 'D:/Wayan/LightBead/method paper/dico data/supervoxels_500/'
list_dic = ['GCaMP6f_04032024_a2_r1.pkl']
fname = ['04032024_GCamp6f_a2_r1_w3_n1000_labels.h5']
data = pd.read_pickle(path_dico + list_dic[0])
time_audio_LB = data['time_audio_aligned']
pulse_song = data['pulse_song']

  
#TODO CHANGE THIS TO THE DESIRED FOLDER contaning the aligned 2p data     
path_dico = 'D:/Wayan/LightBead/method paper/dico data/zscored/RigE/supervoxels_1000/'
list_dic = ['GCaMP6f_12132024_a2_r2.pkl']
fname = ['06212024_6f_a1_r8_n500_labels.h5']
data = pd.read_pickle(path_dico + list_dic[0])
time_audio_2p = data['time_audio_aligned']

#TODO CHANGE THIS TO THE DESIRED FOLDER contaning the stimulus for plotting
dropbox_dir = "C:/Users/wayan.CHRISTAPNI/Documents/Labs/Murthy/LightBead/Code/audio stimuli/Method paper/"
audio_stim = loadmat(dropbox_dir + 'highspeed_pulse_2_WG_paper_forplotting.mat')
time_audio_2p = audio_stim['stim_time'][0]
pulse_song = audio_stim['pulse_song'][0]



start_block_seconds_LB = np.array([5,25,45,65,84,103,123,143,163,183,202.99894,222.99788,242.99788])
end_block_seconds_LB = np.array([15,35,55,75,94,113,133,153,173,192.99894,212.99788,232.99788,252.99788])

start_block_seconds_2p = np.array([5,25,45,65,84,103,123,143,163,183,202.99894,222.99788,242.99788])
end_block_seconds_2p = np.array([15,35,55,75,94,113,133,153,173,192.99894,212.99788,232.99788,252.99788])




#%% Plot heatmaps

## For LB
to_plot_LB = np.flip(zscore(dffs_LB,axis = 1)[:,:7300],axis = 0)
cmap_base = 'viridis' #gnuplot
vmin, vmax = -0.4, 1.1  # -0.8, 1
cmap = f.truncate_colormap(cmap_base, vmin, vmax)

plt.figure(figsize = (4.7,5.3)) #(4,5)
im = plt.imshow(to_plot_LB, aspect = 'auto', vmin = -1, vmax = 1,cmap = cmap, extent = [0.035,258,0,1700])   
plt.tight_layout()
plt.colorbar(im)
plt.yticks([])
plt.fill_between(time_audio_LB,y1=pulse_song+1725, y2=pulse_song +1745,where =pulse_song>0,color='r',alpha=1)
plt.xlabel('Time (s)')
plt.tight_layout()


## for 2p
to_plot_2p = np.flip(zscore(dffs_2p,axis = 1)[:,:569],axis = 0) 


cmap_base = 'magma' #gnuplot
vmin, vmax = 0.0, 0.50 # -0.6 , 0.5
cmap = f.truncate_colormap(cmap_base, vmin, vmax)

plt.figure(figsize = (4.7,5.3))
im = plt.imshow(to_plot_2p, aspect = 'auto', vmin = -1, vmax = 1,cmap = cmap,extent = [0.45,258.18,0,950])   
plt.tight_layout()
plt.colorbar(im)
plt.yticks([])
plt.fill_between(time_audio_2p,y1=pulse_song+960, y2=pulse_song+975,where =pulse_song>0,color='r',alpha=1)
plt.xlabel('Time (s)')
plt.tight_layout()



#%% Plot mean activity


### For LBM
mean_trace_per_pair_LB,sem_trace_per_pair_LB = f.compute_mean_time_series_per_block_pair(
    dffs=zscore(dffs_LB,axis =1),
    time_activity=time_activity_LB,
    Hz=Hz_LB,
    start_block_seconds=start_block_seconds_LB[1:],
    end_block_seconds=end_block_seconds_LB[1:],
    t_added = 2,
    scope = 'LB'
)


mean_stim_traces_LB = f.extract_single_stimulus_per_block_pair(
    pulse_song=pulse_song,
    time_audio=time_audio_LB,
    Hz=100,
    start_block_seconds=start_block_seconds_LB[1:],
    end_block_seconds=end_block_seconds_LB[1:],
    t_added = 2
)


f.plot_calcium_with_stimulus_overlay(mean_trace_per_pair_LB,sem_trace_per_pair_LB, mean_stim_traces_LB, Hz_LB,None,'LB' )



### For 2p
mean_trace_per_pair_2p,sem_trace_per_pair_2p = f.compute_mean_time_series_per_block_pair(
    dffs=zscore(dffs_2p,axis =1),
    time_activity=time_activity_2p,
    Hz=Hz_2p,
    start_block_seconds=start_block_seconds_2p[1:],
    end_block_seconds=end_block_seconds_2p[1:],
    t_added = 2,
    scope = '2p'
)


mean_stim_traces_2p = f.extract_single_stimulus_per_block_pair(
    pulse_song=pulse_song,
    time_audio=time_audio_2p,
    Hz=100,
    start_block_seconds=start_block_seconds_2p[1:],
    end_block_seconds=end_block_seconds_2p[1:],
    t_added = 2
)


f.plot_calcium_with_stimulus_overlay(mean_trace_per_pair_2p,sem_trace_per_pair_2p, mean_stim_traces_2p, Hz_2p,None, '2p' )



#%% Plot the fourier of the mean

f.fourier_mean_activity_interpolate(zscore(dffs_LB,axis = 1),start_block_seconds_LB,end_block_seconds_LB,0,Hz_LB,time_activity_LB,'LB', Hz_LB,142, None,14, 'green')
f.fourier_mean_activity_interpolate(zscore(dffs_2p,axis = 1),start_block_seconds_2p,end_block_seconds_2p,2,Hz_2p,time_activity_2p,'2p', Hz_LB,282, None,1.1, 'magenta')



#%% Plot fraction of power
abs_power_LB, frac_power_Lb = f.power_ROI(zscore(dffs_LB,axis = 1),start_block_seconds_LB,end_block_seconds_LB,0,28.2893,time_activity_LB,'LB',282)
abs_power_2p, frac_power_2p = f.power_ROI(zscore(dffs_2p,axis = 1),start_block_seconds_2p,end_block_seconds_2p,2,2.2,time_activity_2p,'2p',282)

f.plot_power_ROIs(abs_power_LB,frac_power_Lb,abs_power_2p,frac_power_2p,None)


#%% Plot single trace for LBM

roi = -2
limit = 7977

plt.figure(figsize = (20,3))
plt.plot(time_activity_LB[:limit], zscore(dffs_LB[roi,:limit],axis=0), color = 'k')
max_trace = np.max(zscore(dffs_LB[roi,:limit],axis=0))
min_trace = np.min(zscore(dffs_LB[roi,:limit],axis=0))
plt.fill_between(time_audio_LB,y1=(max_trace*pulse_song)+0.05,y2=(max_trace*pulse_song)+0.3,where =pulse_song>0,color='r',alpha=1)
plt.xlim(0,time_activity_LB[-1])
#plt.xticks([60,70,80,90,100,110,120,130,140], [0,10,20,30,40,50,60,70,80], fontsize = 14)
#plt.yticks(fontsize = 14)
plt.xlabel('Time (s)', fontsize = 24)
plt.ylabel('Z(DF/F)', fontsize = 24)
plt.xticks(fontsize = 22)
plt.yticks(fontsize = 22)
plt.locator_params(axis='y', nbins=3)
plt.tight_layout()






























