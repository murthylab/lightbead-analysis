# -*- coding: utf-8 -*-
"""
Created on Tue Jul  1 12:51:13 2025

@author: wayan
"""


####################
# Import 
####################

import numpy as np


import matplotlib.pyplot as plt
from scipy.stats import zscore
from scipy.stats import sem
from scipy.fft import rfftfreq
import scipy.fft
from scipy.signal import find_peaks
import matplotlib
from scipy.interpolate import interp1d

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42



def create_stim(dffs, start_block_seconds,end_block_seconds ,frame_rate, t_i2c=0):
    """
    Description
    ----------
    This function creates a continuous version of the stimulus (array of 0 or 1) with the same shape as the calcium trace for Figure 2,S3,S4
    ----------

    Parameters
    ----------
    dffs (np.ndarray)
        Array containing the calcium activity over time, each row is an ROI.
    start_block_seconds (np.ndarray)
        array containing the start of each block of stimulus in seconds
    end_block_seconds (np.ndarray)
        array containing the end of each block of stimulus in seconds
    t_i2c (float)
        The first time point received by I2C. Set to 0 is data is already aligned
    frame_rate (float)
        frame rate of the scope
    ----------

    Returns
    ----------
    continuous_stim
        An array of 0 and 1s when stimulus is off and on respectively
    ----------   
    
    """
    
    # Get index of start and end
    s = (start_block_seconds)*frame_rate  
    e = (end_block_seconds )*frame_rate
    
    continuous_stim = []
    for ii in range(int((t_i2c)*frame_rate)):
        continuous_stim.append(0)
    
    for ii in range(dffs.shape[1]-int((t_i2c)*frame_rate)):
        if  (s[0]<ii<e[0]) or (s[1]<ii<e[1]) or (s[2]<ii<e[2]) or (s[3]<ii<e[3]) or (s[4]<ii<e[4]) or (s[5]<ii<e[5]) or (s[6]<ii<e[6]) or (s[7]<ii<e[7]) or (s[8]<ii<e[8]) or (s[9]<ii<e[9]) or (s[10]<ii<e[10]) or (s[11]<ii<e[11]) or (s[12]<ii<e[12]):
            continuous_stim.append(1)
        else:
            continuous_stim.append(0)
            
    return (continuous_stim) 



def create_stim_train(dffs, start_block_seconds,end_block_seconds ,frame_rate, t_i2c=0):
    """
    Description
    ----------
    This function creates a continuous version of the stimulus (array of 0 or 1) with the same shape as the calcium trace for Figure 3
    ----------

    Parameters
    ----------
    dffs (np.ndarray)
        Array containing the calcium activity over time, each row is an ROI.
    start_block_seconds (np.ndarray)
        array containing the start of each block in seconds
    end_block_seconds (np.ndarray)
        array containing the end of each block in seconds
    t_i2c (float)
        The first time point received by I2C. Set to 0 is data is already aligned
    frame_rate (float)
        frame rate of the scope
    ----------

    Returns
    ----------
    continuous_stim
        An array of 0 and 1s when stimulus is off and on respectively
    ----------   
    
    """
    # Get index of start and end
    s = (start_block_seconds)*frame_rate  
    e = (end_block_seconds )*frame_rate
    
    continuous_stim = []
    for ii in range(int((t_i2c)*frame_rate)):
        continuous_stim.append(0)
    
    for ii in range(dffs.shape[1]-int((t_i2c)*frame_rate)):
        if  (s[0]<ii<e[0]) or (s[1]<ii<e[1]) or (s[2]<ii<e[2]) or (s[3]<ii<e[3]) or (s[4]<ii<e[4]) or (s[5]<ii<e[5]) or (s[6]<ii<e[6]) or (s[7]<ii<e[7]) or (s[8]<ii<e[8]) or (s[9]<ii<e[9]) or (s[10]<ii<e[10]) or (s[11]<ii<e[11]) or (s[12]<ii<e[12]) or (s[13]<ii<e[13]):
            continuous_stim.append(0.15)
        else:
            continuous_stim.append(0)
            
    return (continuous_stim)      



def crosscorr_sort(dffs, stimulus,cutoff,frame_rate,max_lag=0):
    """
    Description
    ----------
    This function computes the cross correlation between the calcium activity and the auditory stimulus and extract the top cutoff % of ROIs based on the correlation coefficient.
    ----------

    Parameters
    ----------
    dffs (np.ndarray)
        Array containing the calcium activity over time, each row is an ROI.
    stimulus (np.ndarray)
        array containing the auditory stimulus
    cutoff (float)
        threshold in percentage to use when extracting the top X% based on correlation
    max_lag (float)
        the lag over which to compute the cross correlation
    frame_rate (float)
        frame rate of the scope
    ----------

    Returns
    ----------
    audio_correlated, corr_coeff, correlations, sorted_indices
        array containing the index of the top 'cutoff'% ROIs with the highest correlation coefficient with the stimulus
        
    corr_coeff
        array containing the correlation coefficient of the extracted ROIs in audio_correlated
        
    correlations
        array containing the correlation coefficients of all ROIs in dffs   
    
    ----------   
    
    """    
    
    # Normalize the activity and the stimulus
    dffs_mean = dffs.mean(axis=1, keepdims=True)
    dffs_std = dffs.std(axis=1, keepdims=True)
    dffs_normalized = (dffs - dffs_mean) / dffs_std

    
    stimulus_normalized = (stimulus - stimulus.mean()) / stimulus.std()
    
    # define index cutoff
    index_cutoff = int(dffs.shape[0] * (cutoff/100))
    
    # If we don't want any lag
    if max_lag == 0:
        # Compute correlations
        correlations = np.dot(dffs_normalized, stimulus_normalized.T) / dffs_normalized.shape[1]
        correlations = correlations.flatten()
        
        rois = np.arange(0,dffs.shape[0])
        rois = rois[~np.isnan(correlations)]
        
        correlations = correlations[~np.isnan(correlations)]
        
        #sort the correlation coefficients and rois 
        sorted_indices = np.argsort(correlations)
        sorted_corr = correlations[sorted_indices]
        sorted_rois = rois[sorted_indices]
        
        # Extract the top X%
        audio_correlated = sorted_rois[-index_cutoff:]
        corr_coeff = sorted_corr[-index_cutoff:]
        # Result: correlations is a 1D array of shape (number of ROIs,)

    else:
        # Compute correlations for different lags
        num_rois, time_points = dffs.shape
        lags = np.arange(int(-max_lag*frame_rate), int(max_lag*frame_rate) + 1,int(0.5*frame_rate))
        correlation_matrix = np.zeros((num_rois, len(lags)))
        for i, lag in enumerate(lags):
            if lag < 0:
                # Shift auditory stimulus forward
                shifted_stimulus = stimulus_normalized[-lag:]
                activity_subset = dffs_normalized[:, :len(shifted_stimulus)]
            elif lag > 0:
                # Shift auditory stimulus backward
                shifted_stimulus = stimulus_normalized[:-lag]
                activity_subset = dffs_normalized[:, lag:]

            # Compute correlations for the current lag
            correlation_matrix[:, i] = (
                np.dot(activity_subset, shifted_stimulus) / len(shifted_stimulus)
            )
         
        # Find the best correlation across all lags for each ROI
        correlations = np.max(np.abs(correlation_matrix), axis=1)
        #best_lags = lags[np.argmax(np.abs(correlation_matrix), axis=1)]  
        rois = np.arange(0,dffs.shape[0])
        rois = rois[~np.isnan(correlations)]
        correlations = correlations[~np.isnan(correlations)]
        #sort the correlation and rois
        sorted_indices = np.argsort(correlations)
        sorted_corr = correlations[sorted_indices]
        sorted_rois = rois[sorted_indices]
        
        audio_correlated = sorted_rois[-index_cutoff:]
        corr_coeff = sorted_corr[-index_cutoff:]     

    print('number of audio correlated ROIs: {}'.format(len(audio_correlated)))  
    return(audio_correlated, corr_coeff, correlations, sorted_indices[-index_cutoff:])  



def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)
    new_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap 




def compute_mean_time_series_per_block_pair(dffs, time_activity, frame_rate, start_block_seconds, end_block_seconds, t_added, scope):
    """
    Description
    ----------
    This function computes the mean activity across all ROIs during the two presentation of stimulus block with the same frequency.
    ----------

    Parameters
    ----------
    dffs (np.ndarray)
        Array containing the calcium activity over time, each row is an ROI.
    time_activity (np.ndarray)
        array containing the time for each calcium trace
    frame_rate (float)
        frame rate of the scope    
    start_block_seconds (np.ndarray)
        array containing the start of each block in seconds
    end_block_seconds (np.ndarray)
        array containing the end of each block in seconds
    t_added (float)
        Time to add around each block for visualization purposes
    scope (str)    
        'LB' or '2p' to specify which scope was used to aquire the data

    ----------

    Returns
    ----------
    block_pair_traces (shape: (n_block_pairs, samples_per_block)) 
        array containing the mean activity across all ROIs during each block of stimulus with different frequencies
        
    block_pair_sem (shape: (n_block_pairs, samples_per_block)) 
        array containing the sem of the activity across all ROIs during each block of stimulus with different frequencies
    
    ----------   
    
    """   
    
    n_blocks = len(start_block_seconds)
    assert n_blocks % 2 == 0, "Number of blocks must be even"

    
    block_pair_traces = []
    block_pair_sem=[]

    for i in range(0, n_blocks, 2):
        pair_traces = []
        pair_sem = []

        for j in [i, i + 1]:  # handle each of the two blocks in the pair
            start_time = start_block_seconds[j]-t_added
            end_time = end_block_seconds[j]+t_added  
            samples_per_block = int((end_time-start_time) * frame_rate)

            # Get mask for time points in this block
            mask = (time_activity >= start_time) & (time_activity <= end_time)
            if scope == '2p':
                samples_per_block+=1
                if (i==10) and (j==11):
                    mask = np.hstack(( np.array(np.where(mask)[0][0]-1) ,   np.where(mask)[0] ))
            block_data = dffs[:, mask]  

            if block_data.shape[1] != samples_per_block:
                raise ValueError(f"Block {j+1} does not contain {samples_per_block} samples. Got {block_data.shape[1]}.")

            # Average across ROIs
            mean_trace = np.mean(block_data, axis=0)  
            pair_traces.append(mean_trace)
            sem_trace = sem(block_data, axis=0)
            pair_sem.append(sem_trace)
            
        # Average the two blocks 
        mean_pair_trace = np.mean(pair_traces, axis=0) 
        block_pair_traces.append(mean_pair_trace)
        sem_pair_trace = np.mean(sem_trace, axis=0)
        block_pair_sem.append(sem_pair_trace)
        
    return np.stack(block_pair_traces, axis=0),np.stack(block_pair_sem, axis=0)  



def extract_single_stimulus_per_block_pair(stimulus, time_audio, frame_rate, start_block_seconds,end_block_seconds,t_added):
    """
    Description
    ----------
    This function extract the auditory stimulus during each block presentation for plotting purposes in other functions.
    ----------

    Parameters
    ----------
    stimulus (np.ndarray)
        array containing the auditory stimulus
    time_audio (np.ndarray)
        array containing the time for the auditory stimulus
    frame_rate (float)
        frame rate of the scope    
    start_block_seconds (np.ndarray)
        array containing the start of each block in seconds
    end_block_seconds (np.ndarray)
        array containing the end of each block in seconds
    t_added (float)
        Time to add around each block for visualization purposes


    ----------

    Returns
    ----------
    stim_traces (shape: (n_block_pairs, samples_per_block)) 
        array containing the stimulus during each block of stimulus of a given frequency
        
    
    ----------   
    
    """   
    samples_per_block = int((10+t_added+t_added) * frame_rate)
    n_blocks = len(start_block_seconds)
    assert n_blocks % 2 == 0, "Number of blocks must be even"

    stim_traces = []

    for i in range(0, n_blocks, 2):  # take the first block in each pair
        start_time = start_block_seconds[i]-t_added
        end_time = end_block_seconds[i] +t_added # 10s block

        mask = (time_audio >= start_time) & (time_audio < end_time)
        stim_block = stimulus[mask]

        if stim_block.shape[0] != samples_per_block:
            raise ValueError(f"Stimulus block {i+1} has {stim_block.shape[0]} samples; expected {samples_per_block}.")

        stim_traces.append(stim_block)

    return np.stack(stim_traces, axis=0)  





def plot_calcium_with_stimulus_overlay(mean_traces,sem_traces, stim_traces, frame_rate,path, scope):
    """
    Description
    ----------
    This function plots the mean activity across all ROIs during the two presentation of stimulus block of a given frequency overlayed with the auditory stimulus.
    ----------

    Parameters
    ----------
    mean_traces (nd.array) 
        array containing the mean activity across all ROIs during each block of stimulus with different frequencies
    sem_traces (nd.array) 
        array containing the sem of the activity across all ROIs during each block of stimulus with different frequencies
    stim_traces (np.ndarray)
        array containing the stimulus during each block of stimulus of a given frequency
    time_activity (np.ndarray)
        array containing the time for each calcium trace
    frame_rate (float)
        frame rate of the scope    
    path (str)
        Path to folder where to save the plots. If set to None, plots won't be saved.
    scope (str)    
        'LB' or '2p' to specify which scope was used to aquire the data

    ----------
    """
    
    if scope == 'LB':
        col = 'g'
    else:
        col = 'm'
        
    fr = 1/frame_rate
    n_pairs, samples_per_block = mean_traces.shape
    time_axis = np.arange(0,(samples_per_block)*fr,fr)  

    fr_stim = 1/100
    samples_per_block_stim = np.shape(stim_traces[0])[0]
    t_stim = np.arange(0,(float(samples_per_block_stim))*fr_stim,fr_stim)


    for i in range(n_pairs):
        max_trace = np.max(mean_traces[i] + sem_traces[i])
        #min_trace = np.min(mean_traces[i] + sem_traces[i])
        height1 = 0.006 
        height2 = 0.085  
        plt.figure(figsize = (8,5))
        plt.plot(time_axis, mean_traces[i],color= col,alpha = 1, lw = 2.5)
        plt.fill_between(t_stim,y1=(max_trace*stim_traces[i])+ height1,y2=(max_trace*stim_traces[i])+height2,where =stim_traces[i]>0,color='r',alpha=1)
        plt.fill_between(time_axis,y1=(mean_traces[i] + sem_traces[i]),y2=( mean_traces[i] - sem_traces[i]),color=col,alpha=0.4)
        plt.xlabel('Time (s)', fontsize = 36)
        plt.ylabel('Z(DF/F)', fontsize = 36)  
        plt.locator_params(axis='y', nbins=5)
        plt.xlim(0,time_axis[-1])
        plt.xticks(fontsize = 34)
        plt.yticks(fontsize = 34)
        plt.locator_params(axis='y', nbins=3)
        plt.tight_layout()
        
        if path != None:
                plt.savefig(path + 'mean_activity_cluster_block_' + str(i) + '_2p' + '.pdf', transparent = True)
        

        
        
        
        
def fourier_mean_activity_interpolate(dffs,start_block_seconds,end_block_seconds,t_added,frame_rate,time_activity,scope, target_frame_rate,N_target, path,xlim, col):
    """
    Description
    ----------
    This function computes and plots the fourier spectrum of the mean activity for each block pair of a given frequency
    ----------

    Parameters
    ----------
    dffs (np.ndarray)
        Array containing the calcium activity over time, each row is an ROI. 
    start_block_seconds (np.ndarray)
        array containing the start of each block in seconds
    end_block_seconds (np.ndarray)
        array containing the end of each block in seconds
    t_added (float)
        Time to add around each block for visualization purposes
    frame_rate (float)
        frame rate of the scope    
    time_activity (np.ndarray)
        array containing the time for each calcium trace
    scope (str)
        Either 'LB' or '2p'
    Hz_target (float)
        Target frame rate when interpolating 2p
    N_target (float)
        Target number of samples when interpolating 2p
    path (float)
        Path to folder where to save the plots. If set to None, plots won't be saved.
    xlim (float)
        maximum of x axis for plotting 
    col (float)
        color for plotting    
    ---------- 
    """    
    b = 1   # Index that keeps track of the stimulus blocks
    HZ = [0.25,0.5,1,2,3,5]
    # Loop through each pair of stimulus blocks
    for k in range(6):
            # Grab first block
            t_sart = start_block_seconds[b] 
            t_end = end_block_seconds[b] +t_added  
            N = int((t_end-t_sart)*frame_rate)
            t_act = np.linspace(t_sart,t_end,N)
            start_act = np.argwhere((t_sart-time_activity)<0.001)[0][0]
            end_act = np.argwhere((t_end-time_activity)<0.001)[0][0] 
            if (end_act-start_act)>(len(t_act)):
                end_act = end_act - ((end_act-start_act)-len(t_act))
            if len(dffs.shape)>1:
                activity_block1 = dffs[:,start_act:end_act] 
            else:
                activity_block1 = dffs[start_act:end_act] 
            # Grab second block
            t_sart = start_block_seconds[b+1] 
            t_end = end_block_seconds[b+1] + t_added  
            N = int((t_end-t_sart)*frame_rate)
            t_act = np.linspace(t_sart,t_end,N)
            start_act = np.argwhere((t_sart-time_activity)<0.001)[0][0]
            end_act = np.argwhere((t_end-time_activity)<0.001)[0][0] 
            if (end_act-start_act)>(len(t_act)):
                end_act = end_act - ((end_act-start_act)-len(t_act))
            
            if len(dffs.shape)>1:
                activity_block2 = dffs[:,start_act:end_act] 
            else:
                activity_block2 = dffs[start_act:end_act] 
             
            # append both blocks and take the mean
            act_both_block = np.vstack((activity_block1,activity_block2))  
            # Compute the mean
            mean_block = np.mean(act_both_block,axis = 0)
            # subtract the mean
            activity = mean_block-np.mean(mean_block)
            normalize = int(N/2)+1
            
            #### compute fourier
            fourier = scipy.fft.fft(activity) 
            fourier = np.abs(fourier)**2
            ff = scipy.fft.fft(activity)  
            ff = np.abs(ff)**2
            
            # Interpolate 2-photon spectrum to match light bead frequency axis
            if scope == '2p':
                time_inter = rfftfreq(N_target, d = 1/frame_rate) 
                freq_2p = rfftfreq(N, d = 1/frame_rate) 
                interp_func = interp1d(freq_2p, ff[:normalize], kind='linear')
                fourier_interp = interp_func(time_inter)
            
            # plot the Fourier spectrums
            N = len(activity)               
            plt.figure()
            if scope == '2p':
                plt.plot(rfftfreq(N_target, d = 1/frame_rate),fourier_interp[:int(N_target/2)+1], color =  col,lw = 2.5 )
            if scope == 'LB':    
                plt.plot(rfftfreq(N, d = 1/target_frame_rate), fourier[:normalize], color =  col,lw = 2.5)
            plt.xlabel('Frequency (Hz)',fontsize =36)
            plt.ylabel('Amplitude',fontsize =36)
            plt.xlim(0,xlim)
            plt.xticks(fontsize =34)
            plt.yticks(fontsize =34,)
            plt.locator_params(axis='y', nbins=4)
            plt.title('Spectrum block {0}Hz'.format(HZ[k]))
            plt.tight_layout()
            if path != None:
                plt.savefig(path + 'spectrum_block_' + str(k) + '_'+ scope + '.pdf', transparent = True)
            
            b=b+2 # Move to the next block pair
      
    plt.tight_layout()
    
           


def power_ROI(dffs,start_block_seconds,end_block_seconds,t_added,frame_rate,time_activity,scope,N_target):
    """
    Description
    ----------
    This function computes the absolute and fraction of power at each frequencies for each ROIs.
    ----------

    Parameters
    ----------
    dffs (np.ndarray)
        Array containing the calcium activity over time, each row is an ROI. 
    start_block_seconds (np.ndarray)
        array containing the start of each block in seconds
    end_block_seconds (np.ndarray)
        array containing the end of each block in seconds
    t_added (float)
        Time to add around each block for visualization purposes
    frame_rate (float)
        frame rate of the scope    
    time_activity (np.ndarray)
        array containing the time for each calcium trace
    scope (str)
        Either 'LB' or '2p'
    N_target (float)
        Target number of samples when interpolating 2p  
    ---------- 
    
        Returns
    ----------
    ps (nd.array) 
        array containing the absolute power at each frequency for all ROIs
    fracps (nd.array) 
        array containing the fraction of power at each frequency for all ROIs    
    """  
    
    mean_blocks = []
    ### First we grab the mean activity during of each stimulus block pair to subtract later
    b = 1  # Index that keeps track of the stimulus blocks
    # Loop through each block pairs
    for k in range(6):
         # extract first block in the pair
         t_sart = start_block_seconds[b]
         t_end = end_block_seconds[b] +t_added  
         N = int((t_end-t_sart)*frame_rate)         
         t_act = np.linspace(t_sart,t_end,N)
         start_act = np.argwhere((t_sart-time_activity)<0.001)[0][0]
         end_act = np.argwhere((t_end-time_activity)<0.001)[0][0] 
         if (end_act-start_act)>(len(t_act)):
             end_act = end_act - ((end_act-start_act)-len(t_act))
         activity_block1 =  dffs[:,start_act:end_act]
         
         # Grab second block in the pair
         t_sart = start_block_seconds[b+1]
         t_end = end_block_seconds[b+1] + t_added  
         N = int((t_end-t_sart)*frame_rate)
         t_act = np.linspace(t_sart,t_end,N)
         start_act = np.argwhere((t_sart-time_activity)<0.001)[0][0]
         end_act = np.argwhere((t_end-time_activity)<0.001)[0][0] 
         if (end_act-start_act)>(len(t_act)):
             end_act = end_act - ((end_act-start_act)-len(t_act))
         activity_block2 =  dffs[:,start_act:end_act]
         # Combine both blocks
         activity_both_block = np.vstack((activity_block1,activity_block2))
         # store the mean
         mean_blocks.append(np.mean(activity_both_block))
         b +=2 # Move to the next block pair
        
    HZ = [0.25,0.5,1,2,3,5]
    ps = np.zeros((6,dffs.shape[0])) # will contain the absolute power
    fracps = np.zeros((6,dffs.shape[0])) # will contain the fraction of power
    
    for roi, i in enumerate(dffs):  
        b=1   # Index that keeps track of the stimulus blocks
        # Loop through each block pairs
        for k in range(6):
            # extract first block in the pair
            t_sart = start_block_seconds[b] 
            t_end = end_block_seconds[b] +t_added  
            N = int((t_end-t_sart)*frame_rate)
            t_act = np.linspace(t_sart,t_end,N)
            start_act = np.argwhere((t_sart-time_activity)<0.001)[0][0]
            end_act = np.argwhere((t_end-time_activity)<0.001)[0][0] 
            if (end_act-start_act)>(len(t_act)):
                end_act = end_act - ((end_act-start_act)-len(t_act))
            activity_block1 =  dffs[:,start_act:end_act][roi,:]
            # Grab second block in the pair
            t_sart = start_block_seconds[b+1] 
            t_end = end_block_seconds[b+1] + t_added  
            N = int((t_end-t_sart)*frame_rate)
            t_act = np.linspace(t_sart,t_end,N)
            start_act = np.argwhere((t_sart-time_activity)<0.001)[0][0]
            end_act = np.argwhere((t_end-time_activity)<0.001)[0][0] 
            if (end_act-start_act)>(len(t_act)):
                end_act = end_act - ((end_act-start_act)-len(t_act))
            activity_block2 =  dffs[:,start_act:end_act][roi,:]
            # Combine both blocks
            activity_both_block = np.vstack((activity_block1,activity_block2))
            # Compute the mean
            mean_both_block = np.mean(activity_both_block,axis = 0)
            
            #### Compute fourier spectrum
            # subtract the overall mean during these two blocks
            ff_mean = scipy.fft.fft(mean_both_block - mean_blocks[k], norm = 'ortho' )
            ff_mean = np.abs(ff_mean)**2  
            N_mean = len(mean_both_block) 
            normalize = int(N_mean/2)+1
            
            ### slice frequency range
            f_range = [HZ[k]-0.1, HZ[k]+0.1]
            freq_axis = rfftfreq(N, d = 1/frame_rate)
            f0 = np.argmin(np.abs(freq_axis- f_range[0]))
            f1 = np.argmin(np.abs(freq_axis- f_range[1]))
            f_cutoff = np.argmin(np.abs(freq_axis- 1.1))
            
            if scope == 'LB':    
                p = np.sum(ff_mean[f0:f1])
                if HZ[k]<2:
                    totp = np.sum(ff_mean[1:f_cutoff])
                else:
                    totp = np.sum(ff_mean[1:normalize])
                    
            if scope == '2p': 
                if (HZ[k]-freq_axis[f0])>0.8:
                    p = 0
                else:    
                    p = np.sum(ff_mean[f0:f1])

                
                totp = np.sum(ff_mean[1:normalize])           
            
            ps[k,roi] = p
            fracp = (p/totp)*100
            fracps[k,roi] = fracp
        
    return(ps,fracps)



    
def plot_power_ROIs(ps,fracps,ps_2p, fracps_2p,path):
    """
    Description
    ----------
    This function plot the fraction of power at each frequency across all ROIs for both 2p and LB
    ----------

    Parameters
    ----------
    ps (nd.array) 
        array containing the absolute power at each frequency for all ROIs aquired with LB
    fracps (nd.array) 
        array containing the fraction of power at each frequency for all ROIs aquired with LB
    ps_2p (nd.array) 
        array containing the absolute power at each frequency for all ROIs aquired with 2p
    fracps_2p (nd.array) 
        array containing the fraction of power at each frequency for all ROIs aquired with 2p   
    path (str)
        Path to folder where to save the plots. If set to None, plots won't be saved.
    ---------- 
   
    """  
    
    HZ = [0.25,0.5,1,2,3,5]

    # Loop through each frequencies
    for k in range(len(ps)):
        plt.figure(figsize=(3.5, 6))
        plt.plot(np.random.normal(0,0.03, size = len(fracps[k])),fracps[k],'g.',alpha = 0.1)
        plt.plot(np.random.normal(0.3,0.03, size = len(fracps_2p[k])),fracps_2p[k],'m.',alpha = 0.1)
        plt.bar([0],np.mean(fracps[k]),yerr = np.std(fracps[k]),capsize = 5, color = 'white',edgecolor = 'g', width = 0.2)
        plt.bar([0.3],np.mean(fracps_2p[k]),yerr = np.std(fracps_2p[k]),capsize = 5, color = 'white', edgecolor = 'm', width = 0.2)
        plt.xticks([])
        plt.ylabel('Fraction of power at {} Hz'.format(HZ[k]), fontsize = 32)  
        plt.yticks(fontsize = 32)
        plt.locator_params(axis='y', nbins=4)
        plt.tight_layout()
        
        if path != None:
            plt.savefig(path + 'frac_power' + str(HZ[k]) + '_' + '.pdf', transparent = True)



def shuffle_within_blocks(dffs, start_block_seconds, end_block_seconds, frame_rate, seed=None):
    """
    This function returns a copy of the activity where, for each stimulus block,
    the time‐points within that block are independently shuffled for each ROI.

    Parameters
    ----------
    dffs (np.ndarray)
        Array containing the calcium activity over time, each row is an ROI.
    start_block_seconds (np.ndarray)
        array containing the start of each block in seconds
    end_block_seconds (np.ndarray)
        array containing the end of each block in seconds
    frame_rate (float)
        frame rate of the scope   
    seed : int or None
        If given, seeds the RNG 

    Returns
    -------
    shuffled : np.ndarray, shape (n_rois, n_timepoints)
        A copy of `dffs` with time‐points shuffled within each block for each ROI.
    """
    if seed is not None:
        np.random.seed(seed)

    n_rois, n_time = dffs.shape
    shuffled = dffs.copy()

    for start_sec, end_sec in zip(start_block_seconds, end_block_seconds):
        # Convert seconds to integer frame indices
        start_idx = int(np.floor(start_sec * frame_rate))
        end_idx   = int(np.ceil (end_sec   * frame_rate))

        # Clip to valid range
        start_idx = max(start_idx, 0)
        end_idx   = min(end_idx,   n_time)

        block_len = end_idx - start_idx
        if block_len <= 1:
            continue  

        # For each ROI, shuffle the values within [start_idx:end_idx]
        for r in range(n_rois):
            block_vals = dffs[r, start_idx:end_idx]
            permuted   = block_vals[np.random.permutation(block_len)]
            shuffled[r, start_idx:end_idx] = permuted

    return shuffled




def fourier_and_peaks_mean(dffs,start_block_seconds,end_block_seconds,frame_rate,time_activity, path):   
    """
    Description
    ----------
    This function the absolute at27-28Hz and the fourier spectrum for each ROIs
    ----------

    Parameters
    ----------
    dffs (np.ndarray)
        Array containing the calcium activity over time, each row is an ROI. 
    start_block_seconds (np.ndarray)
        array containing the start of each block in seconds
    end_block_seconds (np.ndarray)
        array containing the end of each block in seconds
    frame_rate (float)
        frame rate of the scope 
    time_activity (np.ndarray)
        array containing the time for each calcium trace
     path (str)
        Path to folder where to save the plots. If set to None, plots won't be saved.
    ---------- 
    
        Returns
    ----------
    ps (nd.array) 
        array containing the absolute power at each frequency for all ROIs
    ff_all_roi (nd.array) 
        array containing the mean Fourier spectrum for each ROI
    """  
  
    HZ = 27.77   # frequency of pulses within the stimulus (36ms IPI)
    ps = [] 
    mean_roi = np.zeros((dffs.shape[0],2*int(frame_rate) ))

    ff_all_roi = np.zeros((dffs.shape[0],2*int(frame_rate) )) # Contains the mean ff spectrum of each ROI
    # First we compute the mean activity over all blocks for each ROIs to subtract later
    for roi in range(dffs.shape[0]):
        ## Loop through the blocks
        for k in range(2,14):
            # Grab  block
            t_sart = start_block_seconds[k]
            t_end = end_block_seconds[k]
            N = int((t_end-t_sart)*frame_rate)
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
    
    # compute the mean of all these means
    overall_mean = np.mean(mean_roi)
    # compute fourier
    for roi in range(mean_roi.shape[0]):
        # extract activity of each roi and subtract the overall mean
        activity_fourier = mean_roi[roi,:] - overall_mean
        N = len(activity_fourier) 
        #Slice frequency  range
        f_range = [HZ -0.77, HZ +0.33]  # Frequency range of interesest (27-28Hz)
        freq_axis = rfftfreq(N, d = 1/frame_rate)
        f0 = np.argmin(np.abs(freq_axis- f_range[0]))
        f1 = np.argmin(np.abs(freq_axis- f_range[1]))
        #Compute fourier
        ff = np.abs(scipy.fft.fft(activity_fourier, norm = 'ortho' ))**2
        if roi == 0:
            ff_all_roi = ff
        else:
            ff_all_roi = np.vstack((ff_all_roi,ff))
        
        #compute abstolute power 
        p = np.sum(ff[f0:f1]) 
        ps.append(p)

     
    return  ps , ff_all_roi      




def plot_power_ROIs_all(ps,ps_shuffled,ps_off, path,title):
    """
    Description
    ----------
    This function plots the absolute at 27-28Hz for the three groups (stim on, stim off and shuffled activity)
    ----------

    Parameters
    ----------
    ps (nd.array) 
        array containing the absolute power at 27-28Hz for all ROIs when stimulus is on
    ps_shuffled (nd.array) 
        array containing the absolute power at 27-28Hz for all ROIs when stimulus is on and the activity is shuffled
    ps_off (nd.array) 
        array containing the absolute power at 27-28Hz for all ROIs when stimulus is off
     path (str)
        Path to folder where to save the plots. If set to None, plots won't be saved.
    ---------- 

    """  
    plt.figure(figsize=(11, 6))
    plt.plot(np.random.normal(0.1,0.025, size = len(ps)),ps,'g.',alpha = 0.3)#+'.'
    plt.plot(np.random.normal(0.9,0.025, size = len(ps_shuffled)),ps_shuffled,'m.',alpha = 0.3)#+'.'
    plt.plot(np.random.normal(0.5,0.025, size = len(ps_off)),ps_off,'k.',alpha = 0.3)#+'.'

    plt.bar([0.1],np.mean(ps),yerr = np.std(ps),capsize = 5, color = 'white',edgecolor = 'g', width = 0.1)
    plt.bar([0.9],np.mean(ps_shuffled),yerr = np.std(ps_shuffled),capsize = 5, color = 'white',edgecolor = 'm', width = 0.1)
    plt.bar([0.5],np.mean(ps_off),yerr = np.std(ps_off),capsize = 5, color = 'white',edgecolor = 'k', width = 0.1)

    plt.xticks([])
    plt.yticks(fontsize = 22)
    plt.ylabel('Absolute power at [27-28] Hz', fontsize = 20)
    plt.locator_params(axis='y', nbins=3)
    plt.ylim(0,0.9)
    plt.tight_layout()
    
    if path != None:
        plt.savefig(path + 'absolute_power'+ title +'.pdf', transparent = True) 





def peaks_fourier_ROI_combine(dffs,start_block_seconds,end_block_seconds,t_added,Hz,time_activity,scope,N_target,top_peaks,path_fig):
    """
    Description
    ----------
    This function computes the fourier of each roi for each block and extract the peak frequency of the spectrum.
    It also plots the fourier spectrum for each block of a representative ROI.

    ----------

    Parameters
    ----------
    dffs (np.ndarray)
        Array containing the calcium activity over time, each row is an ROI.
    start_block_seconds (np.ndarray)
        array containing the start of each block in seconds
    end_block_seconds (np.ndarray)
        array containing the end of each block in seconds
    t_added (float)
        Time to add around each block for visualization purposes
    frame_rate (float)
        frame rate of the scope 
    time_activity (np.ndarray)
        array containing the time for each calcium trace
    scope (str)    
        'LB' or '2p' to specify which scope was used to aquire the data   
    N_target (float)
        Target number of samples when interpolating 2p
    top_peaks (int)
        the number of peaks to extract per spectrum
    path_fig (float)
        Path to folder where to save the plots. If set to None, plots won't be saved.
    ----------

    Returns
    ----------
    peak_freq (np.ndarray)
        contains the absolute peak frequency of the Fourier for each block of each ROI.

    ----------   
    
    """ 
    if scope == 'LB':
        roi_to_plot = 1596
        col = 'g'
    else:
        roi_to_plot = 890
        col = 'm'
        
    mean_blocks = []
    # First we grab mean of each block to subtract later
    b = 1 # Index that keeps track of the stimulus blocks
    for k in range(6):
         # Grab first block 
         t_sart = start_block_seconds[b] 
         t_end = end_block_seconds[b] +t_added  
         N = int((t_end-t_sart)*Hz)
         t_act = np.linspace(t_sart,t_end,N)
         start_act = np.argwhere((t_sart-time_activity)<0.001)[0][0]
         end_act = np.argwhere((t_end-time_activity)<0.001)[0][0] 
         if (end_act-start_act)>(len(t_act)):
             end_act = end_act - ((end_act-start_act)-len(t_act))
         activity_block1 =  zscore(dffs[:,start_act:end_act],axis = 1)
         # Grab second block
         t_sart = start_block_seconds[b+1] 
         t_end = end_block_seconds[b+1] + t_added    
         N = int((t_end-t_sart)*Hz)  
         t_act = np.linspace(t_sart,t_end,N)
         start_act = np.argwhere((t_sart-time_activity)<0.001)[0][0]
         end_act = np.argwhere((t_end-time_activity)<0.001)[0][0] 
         if (end_act-start_act)>(len(t_act)):
             end_act = end_act - ((end_act-start_act)-len(t_act))
         activity_block2 =  zscore(dffs[:,start_act:end_act],axis = 1)
         # Combine both blocks
         activity_both_block = np.vstack((activity_block1,activity_block2))
         mean_blocks.append(np.mean(activity_both_block))
         b +=2 # move to next pair
        
    HZ = [0.25,0.5,1,2,3,5]
    peak_freq = np.zeros((6,top_peaks*dffs.shape[0]))
    
    for roi, i in enumerate(dffs):   
        b=1  # Index that keeps track of the stimulus blocks  
        for k in range(6):
             #extract first  block
            t_sart = start_block_seconds[b]
            t_end = end_block_seconds[b] +t_added      
            N = int((t_end-t_sart)*Hz)
            t_act = np.linspace(t_sart,t_end,N)
            start_act = np.argwhere((t_sart-time_activity)<0.001)[0][0]
            end_act = np.argwhere((t_end-time_activity)<0.001)[0][0] 
            if (end_act-start_act)>(len(t_act)):
                end_act = end_act - ((end_act-start_act)-len(t_act))
            activity_block1 =  zscore(dffs[:,start_act:end_act][roi,:])
            # Grab second block
            t_sart = start_block_seconds[b+1]
            t_end = end_block_seconds[b+1] + t_added   
            N = int((t_end-t_sart)*Hz) 
            t_act = np.linspace(t_sart,t_end,N)
            start_act = np.argwhere((t_sart-time_activity)<0.001)[0][0]
            end_act = np.argwhere((t_end-time_activity)<0.001)[0][0] 
            if (end_act-start_act)>(len(t_act)):
                end_act = end_act - ((end_act-start_act)-len(t_act))
            activity_block2 =  zscore(dffs[:,start_act:end_act][roi,:])
            #Combine both blocks
            activity_both_block = np.vstack((activity_block1,activity_block2))
            mean_both_block = np.mean(activity_both_block,axis = 0)
            # subtract the overall mean during these two blocks
            ff_mean = np.abs(scipy.fft.fft(mean_both_block - mean_blocks[k] ))**2  #
            N_mean = len(mean_both_block) 
            normalize = int(N_mean/2)+1

            # Interpolate 2-photon spectrum to match light bead frequency axis
            if scope == '2p':
                # Frequency axes
                time_inter = rfftfreq(N_target, d = 1/Hz)
                freq_2p = rfftfreq(N, d = 1/Hz)
                interp_func = interp1d(freq_2p, ff_mean[:normalize], kind='linear')
                ff_mean = interp_func(time_inter)
                normalize = int(N_target/2)+1
                
            if roi == roi_to_plot:
                   plt.figure()
                   plt.plot(rfftfreq(N_target, d = 1/Hz), ff_mean[:normalize], color =  col)
                   plt.xlabel('Frequency[Hz]',fontsize = 24)
                   plt.ylabel('Amplitude',fontsize = 24)
                   plt.xticks(fontsize =22)
                   plt.yticks(fontsize =22)
                   plt.tight_layout()         
                   if path_fig != None:
                       plt.savefig(path_fig + 'spectrum_ROI_' + str(roi) + '_' + str(HZ[k]) + '_LB.pdf', transparent = True)    

            # extract peaks 
            if scope == 'LB':
                peaks = find_peaks(ff_mean[:normalize],prominence = 0.7)
            else:
                peaks = find_peaks(ff_mean,prominence = 0.7)
            if len(peaks[0])>0:
               # Find the index from the maximum peak
               i_max_peak = peaks[0][np.argmax(ff_mean[peaks[0]])]
               
               #second_highest_peak_index = peaks[0][np.argpartition(ff_mean[peaks[0]],-2)[-2]]
               '''
               third_highest_peak_index = peaks[0][np.argpartition(ff_mean[peaks[0]],-3)[-3]]
               fourth_highest_peak_index = peaks[0][np.argpartition(ff_mean[peaks[0]],-4)[-4]]
               fifth_highest_peak_index = peaks[0][np.argpartition(ff_mean[peaks[0]],-5)[-5]]
               '''
               # Find the x value from that index
               # x_max = (2*np.abs(rfft(data))/N)[i_max_peak]
               x_max = rfftfreq(N_target, d = 1/Hz)[i_max_peak]

               #freq_run_mean.append(x_max)
               peak_freq[k,roi] = x_max
               if top_peaks == 2:
                   peak_freq[k,roi+dffs.shape[0]] = rfftfreq(N_target, d = 1/Hz) [second_highest_peak_index]     
               #peak_freq[k,roi+2*sorter.shape[0]] = rfftfreq(N, d = 1/Hz_target) [third_highest_peak_index]
               #peak_freq[k,roi+3*sorter.shape[0]] = rfftfreq(N, d = 1/Hz_target) [fourth_highest_peak_index]
               #peak_freq[k,roi+4*sorter.shape[0]] = rfftfreq(N, d = 1/Hz_target) [fifth_highest_peak_index]
            
            b=b+2  # move to next pair

    return(peak_freq)



def crosscorr_sort_corr(dffs,stimulus,threshold_test,cutoff,frame_rate):
    """
    Description
    ----------
    This function computes extract the number of ROIs for each correlation coefficient between activity and the auditory stimulus
    ----------

    Parameters
    ----------
    dffs (np.ndarray)
        Array containing the calcium activity over time, each row is an ROI.
    stimulus (np.ndarray)
        array containing the auditory stimulus
    threshold_test (np.ndarray)
        Contains all of the coefficient values over which to extract the number of ROIs
    cutoff (float)
        threshold in percentage to use when extracting the top X% based on correlation
    max_lag (float)
        the lag over which to compute the cross correlation
    frame_rate (float)
        frame rate of the scope
    ----------

    Returns
    ----------
    n_roi
        the number of ROIs extracted for each correlation coefficient
        
    corr_coeff
        array containing the correlation coefficient of the top 'cutoff'% of ROIs with the highest correlation with the stimulus
        
    
    ----------   
    
    """    
    # Normalize dffs and the stimulus
    dffs_mean = dffs.mean(axis=1, keepdims=True)
    dffs_std = dffs.std(axis=1, keepdims=True)
    dffs_normalized = (dffs - dffs_mean) / dffs_std

    stimulus_normalized = (stimulus - stimulus.mean()) / stimulus.std()
    # Compute correlations
    correlations = np.dot(dffs_normalized, stimulus_normalized.T) / dffs_normalized.shape[1]
    correlations = correlations.flatten()
    rois = np.arange(0,dffs.shape[0])
    rois = rois[~np.isnan(correlations)]
    correlations = correlations[~np.isnan(correlations)]
    #sort the correlation and roi index
    sorted_indices = np.argsort(correlations)
    sorted_corr = correlations[sorted_indices]
    index_cutoff = int(dffs.shape[0] * (cutoff/100))
    corr_coeff = sorted_corr[-index_cutoff:]
    n_roi = []
    for t in threshold_test:
        n_roi.append( np.where(sorted_corr>t)[0].shape[0] )  
    return(n_roi, corr_coeff)  





