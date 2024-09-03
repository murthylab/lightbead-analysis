# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 15:07:01 2024

This script generates the panels for the main figures

@author: wayan
"""

#%%
import pandas as pd 
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import functions as f
from scipy.stats import zscore
from scipy.stats import sem
import math
from tslearn.clustering import TimeSeriesKMeans
from sklearn.decomposition import PCA
from scipy.fft import rfft, rfftfreq
import scipy.fft
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster, ward
from scipy.spatial.distance import pdist
from sklearn.cluster import AgglomerativeClustering

matplotlib.rcParams['axes.spines.right'] = False
matplotlib.rcParams['axes.spines.top'] = False



#%%


#######################################
## Define variables
#######################################

scope = '2p'

if scope == 'LB':
    list_dic = ['GCaMP6f_04032024_a2_r2.pkl','GCaMP6f_04032024_a2_r6.pkl', 'GCaMP6f_04162024_a1_r2.pkl', 'GCaMP6f_04192024_a1_r5.pkl'] # sine pulse['GCaMP6f_04032024_a2_r2.pkl','GCaMP6f_04032024_a2_r6.pkl','GCaMP6f_04192024_a1_r5.pkl'] 
    #list_dic = ['GCaMP6f_04032024_a2_r1.pkl' ,'GCaMP6f_04032024_a2_r5.pkl','GCaMP6f_04192024_a1_r2.pkl','GCaMP6f_04192024_a1_r6.pkl'] #'GCaMP6f_04032024_a2_r1.pkl' ,'GCaMP6f_04032024_a2_r5.pkl','GCaMP6f_04192024_a1_r2.pkl','GCaMP6f_04192024_a1_r6.pkl'
    #list_dic = ['GCaMP6f_04032024_a2_r1.pkl' ,'GCaMP6f_04032024_a2_r5.pkl' ,'GCaMP6f_04162024_a1_r1.pkl','GCaMP6f_04192024_a1_r2.pkl', 'GCaMP6f_04192024_a1_r6.pkl', 'GCaMP6f_04192024_a1_r9.pkl'] # 'GCaMP6f_04032024_a2_r1.pkl',  'GCaMP6f_04032024_a2_r5.pkl',  'GCaMP6f_04162024_a1_r1.pkl', 'GCaMP6f_04192024_a1_r2.pkl', 'GCaMP6f_04192024_a1_r6.pkl', 'GCaMP6f_04192024_a1_r9.pkl'

if scope == '2p':
    list_dic = ['GCaMP6f_06212024_a1_r2.pkl', 'GCaMP6f_06212024_a1_r8.pkl','GCaMP6f_06212024_a1_r1.pkl', 'GCaMP6f_06212024_a1_r4.pkl']#'GCaMP6f_06212024_a1_r2.pkl','GCaMP6f_06212024_a1_r8.pkl','GCaMP6f_06212024_a1_r1.pkl'
    #list_dic = ['GCaMP6f_06212024_a1_r5.pkl', 'GCaMP6f_06212024_a1_r7.pkl'] # sine pulse
    
path_dico = 'D:/Wayan/LightBead/method paper/dico data/'


## The following makes sure all runs have the same dimension
min_dim = 100000
for i, dic in enumerate(list_dic):

    data = pd.read_pickle(path_dico + dic)
    if scope == 'LB':
        dffs = data['dffs_aligned']
    if scope == '2p':    
        dffs = data['dffs']
    if dffs.shape[1]<min_dim:
        min_dim = dffs.shape[1]

# Define the metric to extract ROIS
filters = 'threshold' #threshold, 'block' or 'filters
# Set the threshold to extract audio correlated ROIs
threshold_audio_T =  [0.3,0.3,0.3,0.3,0.3,0.3 ]
threshold_audio_block = [0.3,0.3,0.3,0.3,0.3,0.3 ]
threshold_audio_corr = [0.3,0.3,0.3,0.3,0.3,0.3 ]
block = [0]
clustering = True
stim_type = 'Pulse'  # sine-pulse, Pulse


#Define the start and end of ech block for each stimulus
if stim_type == 'Pulse':
    #start_block_seconds = np.array([5,25,45,65,84,103,123,143,163,183,203,223,243])
    #end_block_seconds = np.array([15,35,55,75,94,113,133,153,173,193,213,233,253])
    
    start_bloc_seconds = np.array([5,25,45,65,84,103,123,143,163,183,202.99894,222.99788,242.99788])
    end_block_seconds = np.array([15,35,55,75,94,113,133,153,173,192.99894,212.99788,232.99788,252.99788])
    

if stim_type == 'sine-pulse':
    start_block_seconds = np.array([5,25,45,65,	85,	105,125,145,165,185,205,224.9996,244.9992,264.9992])
    end_block_seconds = np.array([15,35,55,75,95,115,135,155,175,195,214.9996,234.9992,254.9992,274.9992])

   # start_bloc_seconds = np.array([5,25,45,	65,	85,	105,125,145,165,185,205,225,245,265])
   # end_block_seconds = np.array([15,35,55,75,95,115,135,155,175,195,215,235,255,275])

# Set the times we want to add or subtract in the fourier analysis
t_subtracted = 0.0
t_added = 0.0

## Design the kernel if we used crosscorelation to extract auditory ROIs
if scope == 'LB':
    Hz, Hz_target = 28.2893, 28.2893  
    frame_rate = 1/Hz
    rise = np.linspace(0,1,int((50/1000)*Hz))
    decay = np.linspace(1, 0, int((140/1000)*Hz))
    kernel = np.concatenate((rise,decay))
    time_kernel = np.array([0,50,125,200])
    
    
if scope == '2p':
    Hz = 2.535 
    Hz_target = 28.2893 
    frame_rate = 1/Hz
    rise = np.linspace(0,1,1)
    decay = np.linspace(1, 0, 3)
    kernel = np.concatenate((rise,decay))
    time_kernel = np.array([0,50,125,200])    

plt.figure()
plt.plot(time_kernel,kernel, color= 'black')
plt.xlabel('Time(s)')

#%%

# Initialize lists and arrays
time_audio_delay = []
auditory_roi_on = np.array([],dtype=np.int32)
auditory_roi_off = np.array([],dtype=np.int32)
audio_correlated_total = np.array([],dtype=np.int32)
name_run = []
roi_audio_per_run = np.array([],dtype=np.int32)
roi_audio_per_run_off = np.array([],dtype=np.int32)

if filters == 'threshold':
    fig4,axs4 = plt.subplots(1,1, figsize = (25,5))
    fig5,axs5 = plt.subplots(1,1, figsize = (25,5))

    

for i, dic in enumerate(list_dic):
    data = pd.read_pickle(path_dico + dic)
    
    if scope == 'LB':
        dffs = data['dffs_aligned'][:,:min_dim] #:,:min_dim
        time_audio = data['time_audio_aligned']
    if scope == '2p':
        dffs = data['dffs'][:,:min_dim] #:,:min_dim
        time_audio = data['time_audio']
        
   # dffs_z = f._zscore(data['dffs_align'])[:,:min_dim]
    #time_activity = data['time_activity']#[:min_dim]
    pulse_song = data['pulse_song'][0]
    sine_song = data['sine_song'][0]
    t_start_audio = data['time_start_audio']
    t_i2c = data['t_i2c']
    time_activity= np.arange(frame_rate,(dffs.shape[1]+frame_rate)*frame_rate,frame_rate)  
    
    if i == 0:
        plt.figure(figsize = (15,5))
        plt.fill_between(time_audio,y1=1.5*pulse_song,y2=pulse_song,where =pulse_song>0,color='r',alpha=0.5)
        plt.fill_between(time_audio,y1=1.5*sine_song,y2=sine_song,where =sine_song>0,color='b',alpha=0.5)
        plt.xlabel('Time(s)')  

    time_audio_delay.append(t_start_audio)
    print('Run:', dic) 
   
    ###################################################################
    # Extract auditory-correlated ROIs and plot average of each run
    ###################################################################
    if filters == 'threshold':
        #Extract audio correlated ROIs
        audio_correlated_on, mean_on = f.filter_threshold(dffs,threshold_audio_T[i], 'ON', pulse_song, sine_song, time_audio, time_activity)  
        audio_correlated_off, mean_off = f.filter_threshold(dffs,threshold_audio_T[i], 'OFF', pulse_song, sine_song, time_audio, time_activity) 
        # Plot heatmaps of extracted ROIs
        sort = f.sorted_heatmap_audio(dffs,audio_correlated_on,dic, -0.6,0.6,'viridis')
        #Plot mean activity of each run
        plt.figure(figsize = (15,5))       
        plt.plot(time_activity, np.mean(dffs[audio_correlated_on,:],axis=0))    
        max_trace = np.max(np.mean(dffs[audio_correlated_on,:],axis=0))
        plt.fill_between(time_audio,y1=max_trace*pulse_song,y2=(max_trace*pulse_song)+0.02,where =pulse_song>0,color='r',alpha=0.5)
        plt.fill_between(time_audio,y1=max_trace*sine_song,y2=(max_trace*sine_song)+0.02,where =sine_song>0,color='b',alpha=0.5)
        plt.title('Mean activity ON {}'.format(dic))
        
        plt.figure(figsize = (15,5))       
        plt.plot(time_activity, np.mean(dffs[audio_correlated_off,:],axis=0))    
        max_trace = np.max(np.mean(dffs[audio_correlated_off,:],axis=0))
        plt.fill_between(time_audio,y1=max_trace*pulse_song,y2=(max_trace*pulse_song)+0.02,where =pulse_song>0,color='r',alpha=0.5)
        plt.fill_between(time_audio,y1=max_trace*sine_song,y2=(max_trace*sine_song)+0.02,where =sine_song>0,color='b',alpha=0.5)
        plt.title('Mean activity OFF {}'.format(dic))
    
        axs4.plot(np.mean(zscore(dffs[audio_correlated_on,:],axis=1),axis=0),label = dic)
        axs4.legend()
        axs4.set_title('zscored')
        axs5.plot(np.mean(dffs[audio_correlated_on,:],axis=0),label = dic)
        axs5.legend()        
    
    if filters == 'block':
        #Extract audio correlated ROIs
        audio_correlated, mean = f.filter_block(dffs,block,threshold_audio_block[i], start_block_seconds, end_block_seconds,Hz, time_audio, pulse_song, sine_song, time_activity)  
        # Plot heatmaps of extracted ROIs
        sort = f.sorted_heatmap_audio(dffs,audio_correlated,dic, -0.6,0.6,'viridis')
        # Plot mean activity of each run
        plt.figure(figsize = (15,5))       
        plt.plot(time_activity, np.mean(dffs[audio_correlated,:],axis=0))    
        max_trace = np.max(np.mean(dffs[audio_correlated,:],axis=0))
        plt.fill_between(time_audio,y1=max_trace*pulse_song,y2=(max_trace*pulse_song)+0.02,where =pulse_song>0,color='r',alpha=0.5)
        plt.fill_between(time_audio,y1=max_trace*sine_song,y2=(max_trace*sine_song)+0.02,where =sine_song>0,color='b',alpha=0.5)
        plt.title('Mean activity {}'.format(dic))       
        
        
    if filters == 'filters':
        stim = f.create_stim(dffs, start_block_seconds,end_block_seconds,Hz, t_i2c=0)    
        time_filter = np.arange(0,len(stim))/Hz 
        conv = np.convolve (stim, kernel, mode = 'same')
        conv = conv/np.max(conv) 
        if i == 0:
            plt.figure()
            plt.plot(time_filter,stim)
            plt.title("Convolved stimulus")
        #Extract audio correlated ROIs
        audio_correlated, mean = f.filter_corr(dffs,conv,threshold_audio_corr[i], Hz,10)
        # Plot heatmaps of extracted ROIs
        sort = f.sorted_heatmap_audio(dffs,audio_correlated,dic, -0.6,0.6,'viridis')
        # Plot mean activity of each run
        plt.figure(figsize = (15,5))       
        plt.plot(time_activity, np.mean(dffs[audio_correlated,:],axis=0))    
        max_trace = np.max(np.mean(dffs[audio_correlated,:],axis=0))
        plt.fill_between(time_audio,y1=max_trace*pulse_song,y2=(max_trace*pulse_song)+0.02,where =pulse_song>0,color='r',alpha=0.5)
        plt.fill_between(time_audio,y1=max_trace*sine_song,y2=(max_trace*sine_song)+0.02,where =sine_song>0,color='b',alpha=0.5)
        plt.title('Mean activity {}'.format(dic))       
        
        
        
    ###################################################################
    ### append and store activity of auditory ROIs
    ###################################################################

    if filters == 'threshold':
        if i == 0:
            auditory_roi_on = dffs[audio_correlated_on,:]
            auditory_roi_off =dffs[audio_correlated_off,:]
        else:    
            auditory_roi_on = np.vstack((auditory_roi_on,dffs[audio_correlated_on,:]))
            auditory_roi_off = np.vstack((auditory_roi_off,dffs[audio_correlated_off,:]))
            
        roi_audio_per_run = np.concatenate((roi_audio_per_run,audio_correlated_on))    
        roi_audio_per_run_off = np.concatenate((roi_audio_per_run,audio_correlated_off))  
        for r in range(len(audio_correlated_on)):
            name_run.append(dic)
    
    else:
        if i == 0:
            audio_correlated_total = dffs[audio_correlated,:]
        else:    
            audio_correlated_total = np.vstack((audio_correlated_total,dffs[audio_correlated,:]))
            
            
        roi_audio_per_run = np.concatenate((roi_audio_per_run,audio_correlated))  
        for r in range(len(audio_correlated)):
            name_run.append(dic)
                
        
        
        
        
        


########################################################################
## Plot mean activity across all flies
########################################################################
if filters == 'threshold':
    to_plot = auditory_roi_on
else:    
    to_plot = audio_correlated_total

plt.figure(figsize = (22,4))       
#plt.plot(time_activity, np.mean(dffs[auditory_roi_on,:],axis=0),color='forestgreen')    
plt.plot(time_activity, np.mean(to_plot,axis=0),color='black')    
max_trace = np.max(np.mean(to_plot,axis=0))
plt.fill_between(time_audio,y1=max_trace*pulse_song,y2=(max_trace*pulse_song)+0.025,where =pulse_song>0,color='r',alpha=0.9)
plt.fill_between(time_audio,y1=max_trace*sine_song,y2=(max_trace*sine_song)+0.025,where =sine_song>0,color='b',alpha=0.5)
plt.fill_between(time_activity,y1=(np.mean(to_plot,axis=0) + sem(to_plot,axis=0)),y2=(np.mean(to_plot,axis=0) - sem(to_plot,axis=0)),color='black',alpha=0.3)
plt.xlabel('Time (s)') 
plt.ylabel('DF/F')
plt.title('Mean activity ON all flies')
#plt.xlim(19,33)

if filters == 'threshold':
    to_plot = auditory_roi_off

    plt.figure(figsize = (22,4))       
    plt.plot(time_activity, np.mean(to_plot,axis=0),color = 'maroon')    
    #plt.plot(time_filter[:],conv/4, color = 'black')
    max_trace = np.max(np.mean(to_plot,axis=0))
    plt.fill_between(time_audio,y1=max_trace*pulse_song,y2=(max_trace*pulse_song)+0.025,where =pulse_song>0,color='r',alpha=0.9)
    plt.fill_between(time_audio,y1=max_trace*sine_song,y2=sine_song,where =sine_song>0,color='b',alpha=0.5)
    plt.fill_between(time_activity,y1=(np.mean(to_plot,axis=0) + sem(to_plot,axis=0)),y2=(np.mean(to_plot,axis=0) - sem(to_plot,axis=0)),color='maroon',alpha=0.3)
    plt.xlabel('Time (s)') 
    plt.ylabel('DF/F')
    plt.title('Mean activity Off all flies')
            
            
            
            
 

###########################################################
# Plot the mean activity for each bloc across all flies
###########################################################
if filters == 'threshold':
    to_use = auditory_roi_on
else:    
    to_use = audio_correlated_total

y_lim = [0.02, 0.022, 0.02, 0.01, 0.0075,0.005]
HZ = [0.3,0.5,1,2,3,5]
fig,axs = plt.subplots(1,6, figsize = (25,5))
#to_plot = [np.mean(activity_12,axis=0), np.mean(activity_34,axis=0), np.mean(activity_56,axis=0), np.mean(activity_78,axis=0), np.mean(activity_910,axis=0), np.mean(activity_1112,axis=0)]
#to_plot = [activity_12, activity_34, activity_56, activity_78, activity_910, activity_1112]

color = 'black'

if stim_type == 'Pulse':
    b = 1
if stim_type == 'sine-pulse':
    b = 2
    
t_delay_audio = np.mean(time_audio_delay)   
t_delay_audio = np.min(time_audio_delay) 
    

for k in range(6):
    
        # Grab first block
        t_sart = start_block_seconds[b] -t_subtracted
        t_end = end_block_seconds[b] +t_added  
        
        N = int((t_end-t_sart)*Hz)
        
        t_act = np.linspace(t_sart,t_end,N)
        start_act = np.argwhere((t_sart-time_activity)<0.001)[0][0]
        end_act = np.argwhere((t_end-time_activity)<0.001)[0][0] 
        if (end_act-start_act)>(len(t_act)):
            end_act = end_act - ((end_act-start_act)-len(t_act))
        
        activity_block1 = to_use[:,start_act:end_act] 
        
        # Grab second block
        t_sart = start_block_seconds[b+1] - t_subtracted
        t_end = end_block_seconds[b+1] + t_added  
        
        N = int((t_end-t_sart)*Hz)
        
        t_act = np.linspace(t_sart,t_end,N)
        start_act = np.argwhere((t_sart-time_activity)<0.001)[0][0]
        end_act = np.argwhere((t_end-time_activity)<0.001)[0][0] 
        if (end_act-start_act)>(len(t_act)):
            end_act = end_act - ((end_act-start_act)-len(t_act))
        
        activity_block2 = to_use[:,start_act:end_act]     
        
        # append and take the mean
        act_both_block = np.vstack((activity_block1,activity_block2))  
        # plot
        
        time_activity_block = np.linspace(0,10+t_subtracted+t_added, N)

        start = np.argwhere((t_sart-time_audio)<0.001)[0][0]
        end = np.argwhere((t_end-time_audio)<0.001)[0][0] 
        t_stim = np.linspace(0,10+t_subtracted+t_added,end-start)


        axs[k].plot(time_activity_block,np.mean(act_both_block,axis = 0),color=color,alpha = 1, lw = 2)
        axs[k].set_xlabel('Time(s)')
        axs[k].set_ylabel('dF/F')
        axs[k].set_title('{0}Hz'.format(HZ[k]))
        
        max_trace = np.max((np.mean(act_both_block,axis = 0) + sem(act_both_block,axis=0)))
        min_trace = np.min((np.mean(act_both_block,axis = 0) + sem(act_both_block,axis=0)))
        
        axs[k].fill_between(t_stim,y1=(max_trace*pulse_song[start:end])+0.0060,y2=(max_trace*pulse_song[start:end])+0.009,where =pulse_song[start:end]>0,color='r',alpha=0.9)
        axs[k].fill_between(t_stim,y1=(max_trace*sine_song[start:end])+0.0060,y2=(max_trace*sine_song[start:end])+0.009,where =sine_song[start:end]>0,color='blue',alpha=0.9)
       
        axs[k].fill_between(time_activity_block,y1=(np.mean(act_both_block,axis = 0) + sem(act_both_block,axis=0)),y2=(np.mean(act_both_block,axis = 0) - sem(act_both_block,axis=0)),color=color,alpha=0.5)
        b=b+2
        
        plt.figure(figsize = (4,5))
        plt.plot(time_activity_block, np.mean(act_both_block,axis = 0),color= color,alpha = 1, lw = 2)
        plt.fill_between(t_stim,y1=(max_trace*pulse_song[start:end])+0.0060,y2=(max_trace*pulse_song[start:end])+0.009,where =pulse_song[start:end]>0,color='r',alpha=0.9)
        plt.fill_between(t_stim,y1=(max_trace*sine_song[start:end])+0.0060,y2=(max_trace*sine_song[start:end])+0.009,where =sine_song[start:end]>0,color='blue',alpha=0.9)
        plt.fill_between(time_activity_block,y1=(np.mean(act_both_block,axis = 0) + sem(act_both_block,axis=0)),y2=( np.mean(act_both_block,axis = 0) - sem(act_both_block,axis=0)),color=color,alpha=0.5)
        plt.xlabel('Time(s)')
        plt.ylabel('dF/F')  
        plt.tight_layout()
        #plt.title(HZ[k])
        #axs[k].set_xlim(0,10)
        #axs[k].set_ylim(0,y_lim[k])#
        
    
fig.suptitle('Mean activity')       
        
        
        
     

########################################
# Clustering
########################################

    
if clustering == True:
    
    if filters == 'threshold':
        #to_cluster = np.concatenate((auditory_roi_on,auditory_roi_off))
        to_cluster = auditory_roi_on

    else:
        to_cluster = audio_correlated_total

    t_add = 3      
    normalize = True   
    smooth = True
    if scope == 'LB':
        sigma = 18
    if scope == '2p':
        sigma = 4
    full_trace = False
    block_start = 0
    block_end = 0
    metric = 'euclidean' #'euclidean'
    random_state = 66
    raw = True
    
        
    data_clustering, n_clusters = f.prepare_data(to_cluster, start_block_seconds, end_block_seconds,Hz, t_add, normalize, smooth, sigma, full_trace, block_start, block_end)
      
    labels = f.compute_clustering_Kmeans(data_clustering, n_clusters, metric, random_state) #n_clusters
        
    f.plot_all_clusters(to_cluster,data_clustering,labels, n_clusters ,start_block_seconds,end_block_seconds,t_add,time_activity,time_audio,pulse_song,sine_song, full_trace, raw, sigma, block_start, block_end) 
      
    #f.plot_individual_clusters(to_cluster,data_clustering,labels,n_clusters,start_block_seconds,end_block_seconds,t_add,time_activity,time_audio,pulse_song,sine_song, full_trace, raw, sigma, block_start, block_end) 
    
    f.distribution_cluster(labels, n_clusters)
    
    
#######################################
# Fourier on mean activity
#######################################  

f.fourier_mean_activity(np.mean(auditory_roi_on,axis = 0), stim_type,start_block_seconds,end_block_seconds,Hz,time_activity,scope,Hz_target)            

#######################################
# Plot mean activity of given cluster
#######################################

cluster = 4
cluster_indices = np.where(labels == cluster)[0]
to_use = to_cluster[cluster_indices,:]  

plt.figure(figsize = (22,4))       
#plt.plot(time_activity, np.mean(dffs[auditory_roi_on,:],axis=0),color='forestgreen')    
plt.plot(time_activity, np.mean(to_use,axis=0),color='black')    
max_trace = np.max(np.mean(to_use,axis=0))
plt.fill_between(time_audio,y1=max_trace*pulse_song,y2=(max_trace*pulse_song)+0.025,where =pulse_song>0,color='r',alpha=0.9)
plt.fill_between(time_audio,y1=max_trace*sine_song,y2=(max_trace*sine_song)+0.025,where =sine_song>0,color='b',alpha=0.5)
plt.fill_between(time_activity,y1=(np.mean(to_use,axis=0) + sem(to_use,axis=0)),y2=(np.mean(to_use,axis=0) - sem(to_use,axis=0)),color='black',alpha=0.3)
plt.xlabel('Time (s)') 
plt.ylabel('DF/F')
plt.title(cluster)
#plt.xlim(19,33)


#######################################################################
# combine clusters
#######################################################################   
to_merge = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
to_merge = [0,1,2,3,4]
for i, cluster in enumerate(to_merge):
       cluster_indices = np.where(labels == cluster)[0]
       
       if i == 0:
           cluster_indices_combine = cluster_indices
       else:
           cluster_indices_combine = np.hstack((cluster_indices_combine,cluster_indices))
           
plt.figure(figsize = (22,4))       
#plt.plot(time_activity, np.mean(dffs[auditory_roi_on,:],axis=0),color='forestgreen')    
plt.plot(time_activity, np.mean(to_cluster[cluster_indices_combine,:],axis=0),color='black')    
max_trace = np.max(np.mean(to_cluster[cluster_indices_combine,:],axis=0))
plt.fill_between(time_audio,y1=max_trace*pulse_song,y2=(max_trace*pulse_song)+0.025,where =pulse_song>0,color='r',alpha=0.9)
plt.fill_between(time_audio,y1=max_trace*sine_song,y2=(max_trace*sine_song)+0.025,where =sine_song>0,color='b',alpha=0.5)
plt.fill_between(time_activity,y1=(np.mean(to_cluster[cluster_indices_combine,:],axis=0) + sem(to_cluster[cluster_indices_combine,:],axis=0)),y2=(np.mean(to_cluster[cluster_indices_combine,:],axis=0) - sem(to_cluster[cluster_indices_combine,:],axis=0)),color='black',alpha=0.3)
plt.xlabel('Time (s)') 
plt.ylabel('DF/F')  
plt.title('-19')         

f.fourier_mean_activity(to_cluster[cluster_indices_combine,:], stim_type,start_block_seconds,end_block_seconds,Hz,time_activity,scope,Hz_target)            
absolute_power, frac_power, peaks  = f.fourier_individual_roi(to_cluster[cluster_indices_combine,:],stim_type,start_block_seconds,end_block_seconds, Hz, time_activity,scope, Hz_target)        

#######################################################################
# compute and plot the fourier of the mean activity of a given cluster
#######################################################################   

cluster = 4
cluster_indices = np.where(labels == cluster)[0]
to_use = to_cluster[cluster_indices,:]  

f.fourier_mean_activity(to_use, stim_type,start_block_seconds,end_block_seconds,Hz,time_activity,scope,Hz_target)        
absolute_power, frac_power, peaks  = f.fourier_individual_roi(to_use,stim_type,start_block_seconds,end_block_seconds, Hz, time_activity,scope, Hz_target)        

HZ = [0.3,0.5,1,2,3,5]

plt.figure()
for k in range(len(absolute_power)):   
    plt.plot(np.random.normal(k,0.01, size = len(absolute_power[k])),absolute_power[k],'r.',alpha = 0.5)
    plt.hlines(y=np.mean(absolute_power[k]),xmin = k-0.09,xmax =k+ 0.09, color = 'black',lw =2)
    plt.xticks(np.arange(0, len(absolute_power)),labels= HZ)
    plt.xlabel('Stimulus frequency (Hz)')
    plt.ylabel('Absolute power in frequency range')

plt.title('Absolute power in frequency range')

   
plt.figure()
for k in range(len(frac_power)):   
    plt.plot(np.random.normal(k,0.01, size = len(frac_power[k])),frac_power[k],'r.',alpha = 0.5)
    plt.hlines(y=np.mean(frac_power[k]),xmin = k-0.09,xmax =k+ 0.09, color = 'black',lw =2)
    plt.xticks(np.arange(0, len(frac_power)),labels= HZ)
    plt.xlabel('Stimulus frequency (Hz)')
    plt.ylabel('Fraction of power in frequency range')
    
plt.title('fraction of power in frequency range')   


plt.figure()
for k in range(len(peaks)):   
    plt.plot(np.random.normal(k,0.01, size = len(peaks[k])),peaks[k],'r.',alpha = 0.5)
    plt.hlines(y=np.mean(peaks[k]),xmin = k-0.09,xmax =k+ 0.09, color = 'black',lw =2) 
    plt.xticks(np.arange(0, len(peaks)),labels= HZ)
    plt.xlabel('Stimulus frequency (Hz)')
    plt.ylabel('fourier peak values')
    
plt.title('Peaks values') 
  
for k in range(len(peaks)):   
    plt.figure()
    plt.hist(peaks[k], bins=[0, 0.4, 0.8, 1.2, 1.6,2,2.4,2.8,3.2,3.6,4,4.4,4.8,5.2,5.6])
    plt.title(HZ[k])
    

plt.figure()
average_series = np.mean(to_cluster[cluster_indices,:],axis = 0)
for r in cluster_indices:
    plt.plot(time_activity,to_cluster[r,:],color = 'black', alpha = 0.05)
plt.plot(time_activity,average_series, color = 'black')
plt.plot(time_activity,gaussian_filter1d(average_series,sigma = 6), label = cluster,alpha = 0.0)#,color = 'red'
max_trace = np.max(average_series)
plt.legend()

plt.fill_between(time_audio,y1=(max_trace*pulse_song[:])+0.005,y2=(max_trace*pulse_song[:])+0.015,where =pulse_song[:]>0,color='r',alpha=0.9)
plt.fill_between(time_audio,y1=(max_trace*sine_song[:])+0.005,y2=(max_trace*sine_song[:])+0.015,where =sine_song[:]>0,color='blue',alpha=0.9)


###############################################
# Hierarchical clustering
###############################################


# Compute clustering
n_clusters= 2
distance = 15
#hierarchical_cluster = AgglomerativeClustering(n_clusters=n_clusters, metric='euclidean', linkage='ward')#
hierarchical_cluster = AgglomerativeClustering( n_clusters = None, distance_threshold = distance, metric='euclidean', linkage='ward')#
labels = hierarchical_cluster.fit_predict(data_clustering)
n_clusters = len(np.unique(labels))
f.plot_all_clusters(to_cluster,data_clustering,labels, n_clusters ,start_block_seconds,end_block_seconds,t_add,time_activity,time_audio,pulse_song,sine_song, full_trace, raw, sigma, block_start, block_end) 


# Plot the dendogram
labels_clusters = np.arange(0,n_clusters).tolist()
plt.figure()
linkage_data = linkage(data_clustering, method='ward', metric='euclidean', optimal_ordering =False)
f_cluster = fcluster(linkage_data,n_clusters, criterion = 'maxclust')
dendrogram(linkage_data,p=n_clusters, truncate_mode = 'lastp',show_leaf_counts=False,labels =labels_clusters)
plt.show()
len(labels_clusters)

f_cluster.shape



