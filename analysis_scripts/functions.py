# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 11:21:27 2024

This scripts contains the functions used to generate the main panels

@author: wayan
"""

####################
# Import 
####################

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import math
from scipy.ndimage import gaussian_filter1d
from scipy.stats import zscore
from scipy.stats import sem
from tslearn.clustering import TimeSeriesKMeans
from scipy.fft import rfft, rfftfreq
import scipy.fft
from scipy.signal import find_peaks
from scipy.interpolate import splrep, splev
from scipy import sparse
from scipy.sparse import linalg
from numpy.linalg import norm



######################################################################################### 
# Preprocessing
#########################################################################################

def _zscore(x):
    """
    input:
        x: 2D array (n, t)
        
    Return:
        x: zscored array
    """
    x_mean = np.mean(x, axis=-1)
    x_std = np.std(x, axis=-1)
    x = (x - x_mean[:, None]) / x_std[:, None]
    return x

def background_correction(roi, tolerance=1e-6, lam=100, niter=10):
    """
    Description
    ----------
    This function used Asymetric Least Squares Smoothing from P.Eilers and H.Boelens, 2005, to 
    compute backgroung correction for a given ROI.
    ----------

    Parameters
    ----------
    roi (np.ndarray)
        Array containing the ROI that needs to be background corrected
    tolerance (float)
        Tolerence used to determine termination. Used to check whether the weights show any changes. ||w_new - w|| < tolerance
    lam (int)
        Smoothness parameter. Paper suggests to stay in the 10^2 ≤ λ ≤ 10^9 range   
    niter (int)
        number of iteration. Default set to 10 as suggested in original paper
    ----------

    Returns
    ----------
    
    z (np.ndarray)
        Smoothed baseline
    ----------   
    
    """    
    length = len(roi) # length of the roi

    diag = np.ones(length - 2)
    D = sparse.spdiags([diag, -2*diag, diag], [0, -1, -2], length, length - 2) #  initialize the Difference matrix

    H = lam * D.dot(D.T) 

    w = np.ones(length)   # initialize the weigths to 1
    W = sparse.spdiags(w, 0, length, length)  # diagonal matrix with the weights

    weigth_change = 1 # initialize the change in weights
    count = 0

    while weigth_change > tolerance:
        z = linalg.spsolve(W + H, W * roi)
        d = roi - z        # update baseline
        dneg = d[d < 0]

        mean_dneg = np.mean(dneg)
        std_dneg = np.std(dneg)

        w_new = 1 / (1 + np.exp(2 * (d - (2*std_dneg - mean_dneg))/std_dneg))    # Compute new weigths

        weigth_change = norm(w_new - w) / norm(w)    # Compute changes in weights

        w = w_new
        W.setdiag(w)  # update diagonal values

        count += 1

        if count > niter:
            #print('Maximum number of iterations exceeded')
            break

    
    info = {'num_iter': count, 'stop_criterion': weigth_change}
    return z, d, info
    
        



#########################################################################################
## Extracting auditory ROIs
#########################################################################################
def filter_threshold(dffs, threshold, stim, pulse_song, sine_song, time_audio, time_activity, Hz):
    """
    Description
    ----------
    This function takes the mean of the activity when stimulus is ON and if this mean is higher than the mean of the 
    activity when the stimulus is OFF times a chosen threshold, it extracts it as audio correlated.
    ----------

    Parameters
    ----------
    dffs (np.ndarray)
        Array containing the ROIs, each row is an roi.
    threshold (int)
        Threshold to be used when computing the metric and extracting ROIs
    stim (str)
        either 'ON' of 'OFF' to specifiy if we want to extract ROIs correlated with stimulus or with silence    
    pulse_song (np.ndarray)
        array containing either 1 or 0 if pulse song was On of OFf at each time point
    sine_song (np.ndarray)
        array containing either 1 or 0 if sine song was On of OFf at each time point
    time_audio (np.ndarray)
        array containing the time of the audio stimulus
    time_activity (np.ndarray)
        array containing the time of the activity   
    ----------

    Returns
    ----------
    audio_correlated_on or audio_correlated_off (ndarray)
        array containing the index in dffs of the roi that are correlated with the auditory stimulus
        
    mean_activity (float)
        mean activity when the stimulus was ON or OFF depending on stim
    ----------   
    
    """
    #Combine both pulse and sine song to have the time when any songs were present
    tot_stim = pulse_song + sine_song
    #Checks when any song is presented and extracts the corresponding time in seconds
    time_stim = np.unique(np.int_(time_audio[np.where(tot_stim>0)]))
    #truncate time_stim if longer than recording
    time_stim = time_stim[np.where(time_stim < dffs.shape[1]/Hz)]
    # now we check when the time vector in activity equals these values and get the indexes
    index_activity_tot = time_activity.searchsorted(time_stim)    
    
        
    # eliminate values for which activity went on after microscope acquisition
    index_activity_tot = index_activity_tot[np.where(index_activity_tot != dffs.shape[1])]
    
    # get the index in dffs activity when no audio is present
    index_no_audio = []
    for index,t in enumerate(np.int_(time_activity)):
        if t not in time_stim:
            index_no_audio.append(index)
            
    index_no_audio = np.array(index_no_audio)
    
    mean_stim = dffs[:,index_activity_tot].mean(axis = 1) 
    mean_no_stim = dffs[:,index_no_audio].mean(axis = 1)
    
    if stim == 'ON':
        audio_correlated_on = np.where( (mean_stim>(mean_no_stim*threshold+mean_no_stim)) &  (np.amax(dffs,axis = 1)<2)  &  (dffs[:,index_no_audio].mean(axis=1)>0) )[0].tolist()
        mean_activity = mean_stim[audio_correlated_on].tolist()
        print('number of audio correlated ROIs: {}'.format(len(audio_correlated_on)))
        return (audio_correlated_on,mean_activity)
    
    if stim == 'OFF':
        audio_correlated_off = np.where( (mean_no_stim>(mean_stim*threshold+mean_stim)) &  (np.amax(dffs,axis = 1)<2)  &  (dffs[:,index_no_audio].mean(axis=1)>0) )[0].tolist()
        mean_activity = mean_stim[audio_correlated_off].tolist()
        print('number of audio correlated ROIs offset: {}'.format(len(audio_correlated_off))) 
        return  (audio_correlated_off,mean_activity)
       
    else:
        raise Exception('wrong input')




def filter_block(dffs, block, threshold, start_block_seconds, end_block_seconds, frame_rate, time_audio, pulse_song, sine_song, time_activity) :
    """
    Description
    ----------
    This function takes the mean during a chosen block of stimulus and if it is higher than
    the activity from the start of the run to the first block times a chosen threshold it extract
    the ROIs as audio correlated
    ----------

    Parameters
    ----------
    dffs (np.ndarray)
        Array containing the ROIs, each row is an roi.
    block (list)
        list containing the index of the block to use to extract ROIs
    threshold (int)
        Threshold to be used when computing the metric and extracting ROIs
    start_block_seconds (np.ndarray)
        array containing the start of each block in seconds
    end_block_seconds (np.ndarray)
        array containing the end of each block in seconds
    frame_rate (float)
        frame rate of the microscope    
    time_audio (np.ndarray)
        array containing the time of the audio stimulus        
    pulse_song (np.ndarray)
        array containing either 1 or 0 if pulse song was On of OFf at each time point
    sine_song (np.ndarray)
        array containing either 1 or 0 if sine song was On of OFf at each time point
    time_activity (np.ndarray)
        array containing the time of the activity   
    ----------

    Returns
    ----------
    audio_correlated_on or audio_correlated_off (ndarray)
        array containing the index in dffs of the roi that are correlated with the auditory stimulus
    mean_activity (float)
        mean activity when the stimulus was ON or OFF depending on stim
    ----------   
  
    """
    start_first = int((start_block_seconds[0])*frame_rate)
    
    if len(block) == 1:  
        # get the index of the start and end of first block
        start = int((start_block_seconds[block[0]])*frame_rate)
        stop = int((end_block_seconds[block[0]])*frame_rate)  
        activity_block = dffs[:,start:stop]
        
     
    if len(block) > 1:
        # get the index of the start and end of each block and combine activity
        for i in block:
            start = int((start_block_seconds[i])*frame_rate)
            stop = int((end_block_seconds[i])*frame_rate)  
            if i == 0:
                activity_block = dffs[:,start:stop]
                
            else:    
                activity_block = np.hstack((activity_block, dffs[:,start:stop]))
            
    mean_stim = activity_block.mean(axis = 1) 
    mean_no_stim = dffs[:,:start_first].mean(axis = 1)
    
    audio_correlated = np.where( (mean_stim>(mean_no_stim*threshold+mean_no_stim)) &  (np.amax(dffs,axis = 1)<2)  &  (dffs[:,:start_first].mean(axis=1)>0) )[0].tolist()
    mean_activity = mean_stim[audio_correlated].tolist()
                
    print('number of audio correlated ROIs: {}'.format(len(audio_correlated)))  
    return (audio_correlated, mean_activity)    
    
    
 
    
def crosscorr(speed, activity, lag=0):
    """ 
    Description
    ----------
    This functions computes the cross correlation between speed and activity with a inputed lag
    ----------
    
    Parameters
    ----------
    lag (int)
        lag to use between speed and activity
    speed, activity : pandas.Series objects of equal length containing walking speed and activity for a given ROI
    
    Returns
    ----------
    crosscorrelation between speed and activity
    """

    return speed.corr(activity.shift(lag))



def create_stim(dffs, start_block_seconds,end_block_seconds ,frame_rate, t_i2c=0):
    """
    Description
    ----------
    This function creates a continuous signal (array) of 0 or 1 using the start and end of each blocks entered as inputs
    ----------

    Parameters
    ----------
    dffs (np.ndarray)
        Array containing the ROIs, each row is an roi.
    start_block_seconds (np.ndarray)
        array containing the start of each block in seconds
    end_block_seconds (np.ndarray)
        array containing the end of each block in seconds
    t_i2c (float)
        The first time point received by I2C
    frame_rate (float)
        frame rate
    ----------

    Returns
    ----------
    time_stim_aligned
        An array of 0 and 1s
    ----------   
    
    """
    
    s = (start_block_seconds)*frame_rate  
    e = (end_block_seconds )*frame_rate
    
    time_stim_aligned = []
    for ii in range(int((t_i2c)*frame_rate)):
        time_stim_aligned.append(0)
    
    for ii in range(dffs.shape[1]-int((t_i2c)*frame_rate)):
        if  (s[0]<ii<e[0]) or (s[1]<ii<e[1]) or (s[2]<ii<e[2]) or (s[3]<ii<e[3]) or (s[4]<ii<e[4]) or (s[5]<ii<e[5]) or (s[6]<ii<e[6]) or (s[7]<ii<e[7]) or (s[8]<ii<e[8]) or (s[9]<ii<e[9]) or (s[10]<ii<e[10]) or (s[11]<ii<e[11]) or (s[12]<ii<e[12]):
            time_stim_aligned.append(0.15)
        else:
            time_stim_aligned.append(0)
            
    return (time_stim_aligned)      
                
    

  
def filter_corr(dffs,stim_conv,threshold, frame_rate,sec):
    """
    Description
    ----------
    This function computes the crosscorrelated between the convolved stimulus and the activity for different lags.
    If the cross correlated is higher than a given threshold, it extract this ROIs
    ----------

    Parameters
    ----------
    dffs (np.ndarray)
        Array containing the ROIs, each row is an roi.
    stim_conv (np.ndarray)
        convolved stimulus
   threshold (float)
        threshold to use for correlation
    frame_rate (float)
        frame rate
    sec (float)
        number of second for the sliding window. In other words how many seconds we shift the activity to compute the crosscorrelations    
    ----------

    Returns
    ----------
    audio_correlated
        array containing indexes of ROIs in dffs that are correalted with the stimulus
    corr_coeff
     the correlatoin coefficient for each extracted ROIs
    ----------   
    
    """

    ### Get correlation between convolved filter with traces
    audio_correlated = []   # contains the locomotion correlated ROIs
    offset_all=[]          # contains the average lag that gave the best correlation
    corr_coeff = []
    
    for roi in range(dffs.shape[0]):
        d = {'convolved_signal': stim_conv, 'activity': dffs[roi,:]}
        df = pd.DataFrame(data=d)
        s = df['convolved_signal']
        a = df['activity']
        
        seconds = int(sec*frame_rate)
        corr = [crosscorr(s,a, lag) for lag in range(-seconds,seconds+int(frame_rate),int(frame_rate/2))]
        offset = np.floor(len(corr)/2)-np.argmax(corr)
        
        if np.max(corr)>threshold:
            audio_correlated.append(roi)
            offset_all.append(offset) 
            corr_coeff.append(np.max(corr))
    
            
    print('number of audio correlated ROIs: {}'.format(len(audio_correlated)))
    print('average lag {}'.format(np.mean(offset_all)))
    
    return (audio_correlated, corr_coeff)


def sorted_heatmap_audio(dffs, audio_correlated,title,vmin,vmax, cmap):
    """
    Description
    ----------
    This functions computes a heat maps of the audio correlated ROIs
    ----------

    Parameters
    ----------
    dffs (np.ndarray)
        Array containing the ROIs, each row is an roi.
    audio_correlated
        array containing indexes of ROIs in dffs that are correalted with the stimulus    
    title (str)
        title of the figure
    vmin and vmax (float)
        range of values to use for the colorbar 
    cmap (str)
        color map to use    
    ----------

    Returns
    ----------
    Plot the heatmap sorted by weights on the first PC
    sort (ndarray)
        index of the plotted ROIs sorted by weights on the first PC
    ----------   
    
    """    
   #### Plot heatmaps of the audio correlated ROIs
    pca = PCA()
    ROIS = pca.fit_transform(dffs[audio_correlated,:].T)
    weights = pca.components_
    sorter = np.argsort(weights[0])[::1]

    sort = []
    for el in sorter:
        sort.append(audio_correlated[el])
    
    plt.figure(figsize = (5,5))
    im = plt.imshow(dffs[sort,:], aspect = 'auto', vmin = vmin, vmax = vmax,cmap = cmap)
    plt.tight_layout()
    plt.xlabel('Frames')
    plt.colorbar(im)
    plt.ylabel('ROIs')
    plt.title('dic')

    return(sort)    
    
    
#########################################################################################
## Clustering
#########################################################################################

def prepare_data(dffs, start_block_seconds,end_block_seconds, Hz, t_add, normalize, smooth, sigma, full_trace, block_start, block_end):
    """
    Description
    ----------
    This function extracts chunk of activity to be used for clustering and compute the estimated number of clusters
    ----------

    Parameters
    ----------
    dffs (np.ndarray)
        Array containing the ROIs, each row is an roi.
    start_block_seconds (np.ndarray)
        Array that contains the time in seconds of the start of each audio block
    Hz (float)
        frame rate of the microscope
    t_add (int)
        time to be added before and after audio block that will be used in clustering    
    normalize (Booleean)
        If True the input data will be zscored before clustering
    smooth (Booleean)
        If True the input data will be smoothened before clustering  
    sigma (int)
        coeff to used when smoothing the data 
    full_trace (Booleean)  
        if True the full trace will be used for clustering
    block_start (int)    
        if full_trace is Flase then this contains the index of the first block to use for clustering
    block_end (int)    
        if full_trace is Flase then this contains the index of the last block to use for clustering

    ----------

    Returns
    ----------
    data_clustering (np.ndarray)
        array containing the data ready for clustering
    n_clusters (int)
        estimated number of cluster to use    
    ----------   
  
    """
    
    # Define the portion of the trace we want to use
    start = int((start_block_seconds[block_start] - t_add)*Hz)+1
    stop = int((end_block_seconds[block_end] + t_add)*Hz)+1
    
    if normalize == True:
        normalized_data = zscore(dffs,axis=1)
    else:
        normalized_data = dffs
    
    if full_trace == True:
        data_clustering = gaussian_filter1d(normalized_data[:, ], sigma = sigma)        
    else:
        data_clustering = gaussian_filter1d(normalized_data[:, start:stop], sigma = sigma) 

    
    ## Plot the mean activity of input dffs for the time chosen for clustering as sanity check
    plt.figure()
    plt.plot(np.mean(data_clustering,axis = 0))

    #Determine the number of cluster:  square root of the number of points 
    cluster_count = math.ceil(math.sqrt(len(data_clustering))) 
    
    return (data_clustering, cluster_count)
   
    
    
def compute_clustering_Kmeans(dffs,n_clusters, metric, random_state):  
    """
    Description
    ----------
    This function compute kmeans clustering using tslearn
    ----------

    Parameters
    ----------
    dffs (np.ndarray)
        Array containing the ROIs, each row is an roi.
    n_cluster (int)
        number of clusters to use for clustering
    metric (str)
        metric to use in clustering
    random_state (int)
        random_state to use in clustering
    ----------

    Returns
    ----------
    labels (np.ndarray)
        labels computed after clustering
    ----------   
    """
    # compute  clustering
    km = TimeSeriesKMeans(n_clusters=n_clusters, metric=metric, random_state=random_state)
    labels = km.fit_predict(dffs)

    return (labels)    
    
    
def plot_all_clusters(dffs_raw, dffs,labels,cluster_count,start_block_seconds,end_block_seconds,t_add,time_activity,time_audio,pulse_song,sine_song, full_trace, raw, sigma, block_start, block_end):    
    """
    Description
    ----------
    This function plots all of the clusters
    ----------

    Parameters
    ----------
    dffs (np.ndarray)
        Array containing the ROIs, each row is an roi.
    labels (np.ndarray)
        labels computed after clustering        
    cluster_count (int)
        number of clusters used for clustering
    start_block_seconds (np.ndarray)
        Array that contains the time in seconds of the start of each audio block
    end_block_seconds (np.ndarray)
        Array that contains the time in seconds of the end of each audio block  
    t_add (int)
        time to be added before and after audio block that will be used in clustering 
    time_activity (np.ndarray)
        array containing the time of the activity            
    time_audio (np.ndarray)
        array containing the time of the audio stimulus
    pulse_song (np.ndarray)
        array containing either 1 or 0 if pulse song was On of OFf at each time point
    sine_song (np.ndarray)
        array containing either 1 or 0 if sine song was On of OFf at each time point    
    full_trace (Booleean)  
        if True the full trace will be used for clustering 
    raw (Booleean)  
        if True the mean of the raw data is plotted 
    sigma (int)
        coeff to used when smoothing the data         
    block_start (int) 
        index of the first block we want to plot
    block_end (int) 
        index of the last block we want to plot        
    ----------

    Returns
    ----------
    Plots all of the clusters
    ----------   
    """    
    
    plot_count = som_y= math.ceil(math.sqrt(cluster_count))
    
    fig, axs = plt.subplots(plot_count,plot_count,figsize=(25,25))
    fig.suptitle('Clusters')
    row_i=0
    column_j=0

    for label in set(labels):
        cluster = []
        cluster_raw = []
        t_sart = start_block_seconds[block_start]-t_add
        t_end = end_block_seconds[block_end]+t_add
        start_act = np.argwhere((t_sart-time_activity)<0.001)[0][0]
        end_act = np.argwhere((t_end-time_activity)<0.001)[0][0] 
        if end_act-start_act > dffs.shape[1]:
            end_act = end_act - ((end_act-start_act) - dffs.shape[1])
        if end_act-start_act < dffs.shape[1]:
            end_act = end_act + (dffs.shape[1] - (end_act-start_act ))    
        for i in range(len(labels)): 
                if(labels[i]==label):
                    if full_trace:
                        if raw == True:
                            axs[row_i, column_j].plot(time_activity[:],dffs_raw[i,:],c="grey",alpha=0.4)
                            cluster_raw.append(dffs_raw[i,:])
                            
                        else:    
                            axs[row_i, column_j].plot(time_activity[:],gaussian_filter1d( dffs[i,:], sigma = sigma),c="grey",alpha=0.4)
                        
                    else: 
                        if raw == True:
                            cluster_raw.append(dffs_raw[i,start_act:end_act])
                         #   axs[row_i, column_j].plot(time_activity[start_act:end_act],dffs_raw[i,start_act:end_act],c="grey",alpha=0.4)
                        if raw == False:                            
                            axs[row_i, column_j].plot(time_activity[start_act:end_act],gaussian_filter1d( dffs[i,:], sigma = sigma),c="grey",alpha=0.4)
                            
                    cluster.append(dffs[i,:])

                        
                    
        if len(cluster) > 0:
            if full_trace:
                if raw == True:
                    axs[row_i, column_j].plot(time_activity[:],np.average(np.vstack(cluster_raw),axis=0),c="red")
                else:    
                    axs[row_i, column_j].plot(time_activity[:],np.average(np.vstack(gaussian_filter1d(cluster,sigma = sigma)), axis=0), c="red")

            else:
                if raw == True:
                    axs[row_i, column_j].plot(time_activity[start_act:end_act],np.average(np.vstack(cluster_raw),axis=0),c="black")
                    axs[row_i, column_j].fill_between(time_activity[start_act:end_act],y1= (np.average(np.vstack(cluster_raw),axis=0) + sem(np.vstack(cluster_raw),axis=0)),y2=(np.average(np.vstack(cluster_raw),axis=0)- sem(np.vstack(cluster_raw),axis=0)),color='black',alpha=0.3)
                else:
                    axs[row_i, column_j].plot(time_activity[start_act:end_act],np.average(np.vstack(gaussian_filter1d(cluster,sigma = sigma)),axis=0),c="red")
                    
        axs[row_i, column_j].set_title("Cluster "+str(row_i*som_y+column_j))
        
        plt.tight_layout()
        
        t_sart = start_block_seconds[block_start]-t_add
        t_end = end_block_seconds[block_end]+t_add
        start_audio = np.argwhere((t_sart-time_audio)<0.001)[0][0]
        end_audio = np.argwhere((t_end-time_audio)<0.001)[0][0] 
        if raw == True:
            max_trace = np.max(np.mean(np.array(cluster_raw),axis=0) + sem((cluster_raw),axis=0) )
            min_trace = np.min(np.mean(np.array(cluster_raw),axis=0))
            shift = 0
        else:
            max_trace = np.max(cluster)  
            min_trace = np.min(cluster)
            shift = 0.3
            
        if full_trace:
            axs[row_i, column_j].fill_between(time_audio[:],y1=(max_trace*pulse_song[:])+0.15,y2=(max_trace*pulse_song[:])+0.3,where =pulse_song[:]>0,color='r',alpha=0.9)
            axs[row_i, column_j].fill_between(time_audio[:],y1=(max_trace*sine_song[:])+0.15,y2=(max_trace*sine_song[:])+0.3,where =sine_song[:]>0,color='blue',alpha=0.9)
        else:
            axs[row_i, column_j].fill_between(time_audio[start_audio:end_audio],y1=(max_trace*pulse_song[start_audio:end_audio])+ shift,y2=(max_trace*pulse_song[start_audio:end_audio]) + 0.1*max_trace,where =pulse_song[start_audio:end_audio]>0,color='r',alpha=0.9)
            axs[row_i, column_j].fill_between(time_audio[start_audio:end_audio],y1=(max_trace*sine_song[start_audio:end_audio])+ shift,y2=(max_trace*sine_song[start_audio:end_audio]) + 0.1*max_trace,where =sine_song[start_audio:end_audio]>0,color='blue',alpha=0.9)
            #axs[row_i, column_j].fill_between(time_audio[start_audio:end_audio],y1=min_trace,y2=(max_trace*pulse_song[start_audio:end_audio]),where =pulse_song[start_audio:end_audio]>0,color='r',alpha=0.2)
            #axs[row_i, column_j].fill_between(time_audio[start_audio:end_audio],y1=min_trace,y2=(max_trace*sine_song[start_audio:end_audio]),where =sine_song[start_audio:end_audio]>0,color='blue',alpha=0.2)
            
        column_j+=1
        if column_j%plot_count == 0:
            row_i+=1
            column_j=0
            
            
    plt.show() 
    
   
def plot_individual_clusters(dffs_raw, dffs,labels,cluster_count,start_block_seconds,end_block_seconds,t_add,time_activity,time_audio,pulse_song,sine_song, full_trace, raw, sigma, block_start, block_end):    
    """
    Description
    ----------
    This function plot each cluster one at a time
    ----------

    Parameters
    ----------
    dffs (np.ndarray)
        Array containing the ROIs, each row is an roi.
    labels (np.ndarray)
        labels computed after clustering        
    cluster_count (int)
        number of clusters used for clustering
    start_block_seconds (np.ndarray)
        Array that contains the time in seconds of the start of each audio block
    end_block_seconds (np.ndarray)
        Array that contains the time in seconds of the end of each audio block  
    t_add (int)
        time to be added before and after audio block that will be used in clustering 
    time_activity (np.ndarray)
        array containing the time of the activity            
    time_audio (np.ndarray)
        array containing the time of the audio stimulus
    pulse_song (np.ndarray)
        array containing either 1 or 0 if pulse song was On of OFf at each time point
    sine_song (np.ndarray)
        array containing either 1 or 0 if sine song was On of OFf at each time point    
    full_trace (Booleean)  
        if True the full trace will be used for clustering 
    raw (Booleean)  
        if True the mean of the raw data is plotted  
    block_start (int) 
        index of the first block we want to plot
    block_end (int) 
        index of the last block we want to plot        
    ----------

    Returns
    ----------
    Plots all of the clusters
    ----------   
    """    
    
    plot_count = som_y= math.ceil(math.sqrt(cluster_count))

    # For each label there is,
    # plots every series with that label
    for label in set(labels):
        plt.figure(figsize = (8,5))
        cluster = []
        cluster_raw = []
        t_sart = start_block_seconds[block_start]-t_add
        t_end = end_block_seconds[block_end]+t_add
        start_act = np.argwhere((t_sart-time_activity)<0.001)[0][0]
        end_act = np.argwhere((t_end-time_activity)<0.001)[0][0] 
        if end_act-start_act > dffs.shape[1]:
            end_act = end_act - ((end_act-start_act) - dffs.shape[1])
        if end_act-start_act < dffs.shape[1]:
            end_act = end_act + (dffs.shape[1] - (end_act-start_act ))    
        for i in range(len(labels)): 
                if(labels[i]==label):
                    if full_trace:
                        if raw == True:
                            plt.plot(time_activity[:],dffs_raw[i,:],c="grey",alpha=0.4)
                            cluster_raw.append(dffs_raw[i,:])
                            
                        else:    
                            plt.plot(time_activity[:],gaussian_filter1d( dffs[i,:], sigma = sigma),c="grey",alpha=0.4)
                        
                    else: 
                        if raw == True:
                            cluster_raw.append(dffs_raw[i,start_act:end_act])
                         #   axs[row_i, column_j].plot(time_activity[start_act:end_act],dffs_raw[i,start_act:end_act],c="grey",alpha=0.4)
                        if raw == False:                            
                            plt.plot(time_activity[start_act:end_act],gaussian_filter1d( dffs[i,:], sigma = sigma),c="grey",alpha=0.4)
                            
                    cluster.append(dffs[i,:])

                        
                    
        if len(cluster) > 0:
            if full_trace:
                if raw == True:
                    plt.plot(time_activity[:],np.average(np.vstack(cluster_raw),axis=0),c="red")
                else:    
                    plt.plot(time_activity[:],np.average(np.vstack(gaussian_filter1d(cluster,sigma = sigma)), axis=0), c="red")

            else:
                if raw == True:
                    plt.plot(time_activity[start_act:end_act],np.average(np.vstack(cluster_raw),axis=0),c="black")
                    plt.fill_between(time_activity[start_act:end_act],y1= (np.average(np.vstack(cluster_raw),axis=0) + sem(np.vstack(cluster_raw),axis=0)),y2=(np.average(np.vstack(cluster_raw),axis=0)- sem(np.vstack(cluster_raw),axis=0)),color='black',alpha=0.3)
                else:
                    plt.plot(time_activity[start_act:end_act],np.average(np.vstack(gaussian_filter1d(cluster,sigma = sigma)),axis=0),c="red")

        t_sart = start_block_seconds[block_start]-t_add
        t_end = end_block_seconds[block_end]+t_add
        start_audio = np.argwhere((t_sart-time_audio)<0.001)[0][0]
        end_audio = np.argwhere((t_end-time_audio)<0.001)[0][0] 
        if raw == True:
            max_trace = np.max(cluster_raw)
            max_trace = np.max(np.mean(np.array(cluster_raw),axis=0) + sem((cluster_raw),axis=0) )
            min_trace = np.min(np.mean(np.array(cluster_raw),axis=0))
            shift = 0
        else:
            max_trace = np.max(cluster)
            min_trace = np.min(cluster)
            shift = 0.3
            
        if full_trace:
            plt.fill_between(time_audio[:],y1=(max_trace*pulse_song[:])+0.15,y2=(max_trace*pulse_song[:])+0.3,where =pulse_song[:]>0,color='r',alpha=0.9)
            plt.fill_between(time_audio[:],y1=(max_trace*sine_song[:])+0.15,y2=(max_trace*sine_song[:])+0.3,where =sine_song[:]>0,color='blue',alpha=0.9)
        else:
            plt.fill_between(time_audio[start_audio:end_audio],y1=(max_trace*pulse_song[start_audio:end_audio])+ shift,y2=(max_trace*pulse_song[start_audio:end_audio]) + 0.1*max_trace ,where =pulse_song[start_audio:end_audio]>0,color='r',alpha=0.9)
            plt.fill_between(time_audio[start_audio:end_audio],y1=(max_trace*sine_song[start_audio:end_audio])+ shift,y2=(max_trace*sine_song[start_audio:end_audio]) + 0.1*max_trace, where =sine_song[start_audio:end_audio]>0,color='blue',alpha=0.9)
            #axs[row_i, column_j].fill_between(time_audio[start_audio:end_audio],y1=min_trace,y2=(max_trace*pulse_song[start_audio:end_audio]),where =pulse_song[start_audio:end_audio]>0,color='r',alpha=0.2)
            #axs[row_i, column_j].fill_between(time_audio[start_audio:end_audio],y1=min_trace,y2=(max_trace*sine_song[start_audio:end_audio]),where =sine_song[start_audio:end_audio]>0,color='blue',alpha=0.2)
        
            
    plt.tight_layout()        
    plt.show() 
        
 
    
 
    
def distribution_cluster(labels, n_clusters):
    """
    Description
    ----------
    This function compute a bar plot of the distribution of clusters
    ----------

    Parameters
    ----------

    labels (np.ndarray)
        labels computed after clustering        
    n_clusters (int)
        number of clusters used for clustering
    ----------

    Returns
    ----------
    Plots the distribution of clusters
    ----------   
    """    
    
    cluster_c = [len(labels[labels==i]) for i in range(n_clusters)]
    cluster_n = ["Cluster "+str(i) for i in range(n_clusters)]
    plt.figure(figsize=(15,5))
    plt.title("Cluster Distribution for KMeans")
    plt.bar(cluster_n,cluster_c)
    plt.xticks(rotation=90)
    plt.show()    
    
    
    
    
    
#################################################################################################
## Compute Fourier   
#################################################################################################  
    
def fourier_mean_activity(dffs, stim_type,start_block_seconds,end_block_seconds,Hz,time_activity,scope, Hz_target):
    """
    Description
    ----------
    This function computes the fourier of the mean activity for each block
    ----------

    Parameters
    ----------
    dffs (np.ndarray)
        Array containing the ROIs, each row is an roi.
    stim_type (str)
        Either Pulse or sine-pulse depending on which auditory stimuli was presented    
    start_block_seconds (np.ndarray)
        array containing the start of each block in seconds
    end_block_seconds (np.ndarray)
        array containing the end of each block in seconds
    Hz (float)
        frame_rate    
    time_activity (np.ndarray)
        array containing the time of the activity   
    scope (str)
        Either 'LB' or '2p'
    Hz_target (float)
        if scope == '2p' it will use inpterpolation to generate a signal with the targeted frame rate Hz_target   
    ----------

    Returns
    ----------
    Plot the fourier of the mean activity
    ----------   
    
    """    
    fig,axs = plt.subplots(1,6, figsize = (25,5))  # rfft
    fig.suptitle('rfft') 
    fig2,axs2 = plt.subplots(1,6, figsize = (25,5))  # fft
    fig2.suptitle('fft') 
    fig3,axs3 = plt.subplots(3,6, figsize = (25,10))  # all
    
    color = 'black'
    
    if stim_type == 'Pulse':
        b = 1
    if stim_type == 'sine-pulse':
        b = 2
        
    HZ = [0.3,0.5,1,2,3,5]
    y_lim_fft = [15, 17,7.5,2, 0.6, 0.6]
    y_lim_rfft = [0.04, 0.04,0.04,0.01,0.01, 0.01]
    for k in range(6):
        
            # Grab first block
            t_sart = start_block_seconds[b] #-t_subtracted
            t_end = end_block_seconds[b] #+t_added  
            
            N = int((t_end-t_sart)*Hz)
            
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
            t_sart = start_block_seconds[b+1] #- t_subtracted
            t_end = end_block_seconds[b+1] #+ t_added  
            
            N = int((t_end-t_sart)*Hz)
            
            t_act = np.linspace(t_sart,t_end,N)
            start_act = np.argwhere((t_sart-time_activity)<0.001)[0][0]
            end_act = np.argwhere((t_end-time_activity)<0.001)[0][0] 
            if (end_act-start_act)>(len(t_act)):
                end_act = end_act - ((end_act-start_act)-len(t_act))
            
            
            if len(dffs.shape)>1:
                activity_block2 = dffs[:,start_act:end_act] 
            else:
                activity_block2 = dffs[start_act:end_act] 
               
            
            # append and take the mean
            act_both_block = np.vstack((activity_block1,activity_block2))  
            # Compute the mean
            mean_block = np.mean(act_both_block,axis = 0)
            time_block = time_activity[start_act:end_act]
            axs3[0,k].plot(time_block, mean_block )
            if scope == '2p':
                # Compute spline representation of the curve
                spl = splrep(time_activity[start_act:end_act], mean_block)
            
                # Create new x axis *similar to LB)
                fr = 1/Hz_target
                time_inter = np.arange(time_activity[start_act:end_act][0],(time_activity[start_act:end_act][-1]),fr)  
                mean_interp = splev(time_inter, spl)
                mean_block = mean_interp
                time_block = time_inter
            
            #axs3[0,k].plot(time_activity[start_act:end_act], mean_block )
            if scope == '2p':
                axs3[0,k].plot(time_inter, mean_interp, color = 'red', label = 'Interpolation', alpha = 0.4 )
                axs3[0,k].legend()
                
            N = len(mean_block)            
            
            axs[k].plot(rfftfreq(N, d = 1/Hz_target), 2*np.abs(rfft(mean_block-np.mean(mean_block)))/N, color =  color)
            axs[k].set_xlabel('Frequency[Hz]')
            axs[k].set_ylabel('Amplitude')
            axs[k].set_title('Spectrum block {0}Hz'.format(HZ[k]))
            #axs[k].set_xlim(0,10)
            #axs[k].set_ylim(0,y_lim[k])#
            
            axs3[1,k].plot(rfftfreq(N, d = 1/Hz_target), 2*np.abs(rfft(mean_block-np.mean(mean_block)))/N, color =  color)
            axs3[1,k].set_xlabel('Frequency[Hz]')
            axs3[1,k].set_ylabel('Amplitude')
            axs3[1,k].set_title('Spectrum block {0}Hz'.format(HZ[k]))
            #axs[k].set_xlim(0,10)
            axs3[1,k].set_ylim(0,y_lim_rfft[k])#
            axs3[1,k].set_title('rfft')
            
            '''plt.figure()
            plt.plot(rfftfreq(N, d = 1/sampling_rate), 2*np.abs(rfft(f))/N)
            plt.xlabel('Frequency[Hz]')
            plt.ylabel('Amplitude')
            plt.ylim(0,y_lim[k])'''
            
            # plot using fft
            fourier = scipy.fft.fft(mean_block-np.mean(mean_block)) #
            fourier = np.abs(fourier)**2
            normalize = int(N/2)+1
            axs2[k].plot(rfftfreq(N, d = 1/Hz_target),fourier[:normalize], color =  color)
            axs2[k].set_xlabel('Frequency[Hz]')
            axs2[k].set_ylabel('Amplitude')
            axs2[k].set_title('Spectrum block {0}Hz'.format(HZ[k]))
            #if scope == 'LB':
                #axs2[k].set_ylim(0,y_lim_fft[k])
            
            axs3[2,k].plot(rfftfreq(N, d = 1/Hz_target),fourier[:normalize], color =  color)
            axs3[2,k].set_xlabel('Frequency[Hz]')
            axs3[2,k].set_ylabel('Amplitude')
            axs3[2,k].set_title('Spectrum block {0}Hz'.format(HZ[k]))
            axs3[2,k].set_title('fft')
            #axs[k].set_xlim(0,10)
            if scope == 'LB':
                axs3[2,k].set_ylim(0,y_lim_fft[k])#
            
            plt.figure()
            plt.plot(rfftfreq(N, d = 1/Hz_target), fourier[:normalize], color =  color)
            plt.xlabel('Frequency[Hz]')
            plt.ylabel('Amplitude')
            plt.title('Spectrum block {0}Hz'.format(HZ[k]))
            #plt.ylim(0,y_lim[k])
            
            b=b+2
            
    plt.tight_layout()
            
    
    
def fourier_individual_roi(dffs,stim_type,start_block_seconds,end_block_seconds, Hz, time_activity, scope, Hz_target):
    """
    Description
    ----------
    This function computes the fourier of each roi for each block 
    and returns the absolute power in frequency range, fraction of power and distribution of peaks
    ----------

    Parameters
    ----------
    dffs (np.ndarray)
        Array containing the ROIs, each row is an roi.
    stim_type (str)
        Either Pulse or sine-pulse depending on which auditory stimuli was presented    
    start_block_seconds (np.ndarray)
        array containing the start of each block in seconds
    end_block_seconds (np.ndarray)
        array containing the end of each block in seconds
    Hz (float)
        frame_rate    
    time_activity (np.ndarray)
        array containing the time of the activity   
    scope (str)
        Either 'LB' or '2p'        
    Hz_target (float)
        if scope == '2p' it will use inpterpolation to generate a signal with the targeted frame rate Hz_target       
    ----------

    Returns
    ----------
    mean_absolute_power (list)
        contains the absolute power for each roi and for each block
    mean_frac_power (list)
        contains the fraction of power for each roi and for each block
    freq_block (list)
        contains the peaks value for each roi and for each block
    ----------   
    
    """       
    HZ = [0.3,0.5,1,2,3,5] # frequency of each stimulus block
    
    if stim_type == 'Pulse':
        b = 1
    if stim_type == 'sine-pulse':
        b = 2
    
    # Initialize lists
    mean_absolute_power = []    # store the absolte power in frequency range for each block
    mean_frac_power = []        # store the fraction of power in frequency range for each block
    freq_block =[]              # store the peak frequency of the fourier for each block 
    #fracps_matrix = []
    #ps_matrix = []
    for k in range(6):
        ps = []                 # store the absolte power in frequency range for each roi
        fracps = []             # store the fraction of power in frequency range for each roi
        freq_run = []           # store the peak frequency of the fourier for each roi 
        f_range = [HZ[k]-0.1, HZ[k]+0.1]  # Frequency range of interesest
        
        
        ################ extract block ################
        t_sart = start_block_seconds[b] #-t_subtracted
        t_end = end_block_seconds[b] #+t_added  
        
        N = int((t_end-t_sart)*Hz)
        
        t_act = np.linspace(t_sart,t_end,N)
        start_act = np.argwhere((t_sart-time_activity)<0.001)[0][0]
        end_act = np.argwhere((t_end-time_activity)<0.001)[0][0] 
        if (end_act-start_act)>(len(t_act)):
            end_act = end_act - ((end_act-start_act)-len(t_act))
        
        activity_block1 = dffs[:,start_act:end_act] 
        
        # Grab second block
        t_sart = start_block_seconds[b+1] #- t_subtracted
        t_end = end_block_seconds[b+1] #+ t_added  
        
        N = int((t_end-t_sart)*Hz)
        
        t_act = np.linspace(t_sart,t_end,N)
        start_act = np.argwhere((t_sart-time_activity)<0.001)[0][0]
        end_act = np.argwhere((t_end-time_activity)<0.001)[0][0] 
        if (end_act-start_act)>(len(t_act)):
            end_act = end_act - ((end_act-start_act)-len(t_act))
        
        activity_block2 = dffs[:,start_act:end_act]     
        activity_both_block = np.vstack((activity_block1,activity_block2))
        
        '''fourier_input = activity_both_block-np.mean(activity_both_block)
       
        ################# Compute fourier ################
        fourier_rfft = 2*np.abs(rfft(fourier_input))/N
        fourier_fft = np.abs(scipy.fft.fft(fourier_input))**2
        normalize = int(N/2)+1

        ################# Slice frequency ################
        freq_axis = scipy.fft.fftfreq(N, d = 1.0/Hz)
        f0 = np.argmin(np.abs(freq_axis- f_range[0]))
        f1 = np.argmin(np.abs(freq_axis- f_range[1]))
        df = freq_axis[1]-freq_axis[0]
        
        ################# compute fraction of power and abstolute power ################
        p_matrix = np.sum(fourier_fft[:,f0:f1],axis=1) 
        ps_matrix.append(p_matrix)
        
        totp_matrix = np.sum(fourier_fft[:,1:normalize],axis = 1)
        
        fracp_matrix = (p_matrix/totp_matrix)*100
        fracps_matrix.append(fracp_matrix)
        '''
        b=b+2
        for roi in range(activity_both_block.shape[0]):
            
            ################ extract activity of each roi ################
            activity = activity_both_block[roi,:] - np.mean(activity_both_block)
            
            ################ interpolate for 2p ################
            if scope == '2p':
                spl = splrep(time_activity[start_act:end_act], activity)
            
                # Create new x axis *similar to LB)
                fr = 1/Hz_target
                time_inter = np.arange(time_activity[start_act:end_act][0],(time_activity[start_act:end_act][-1]),fr)  
                mean_interp = splev(time_inter, spl)
                activity = mean_interp
                
                
            N = len(activity) 
            normalize = int(N/2)+1
            
            ################# Slice frequency  range################
            freq_axis = scipy.fft.fftfreq(N, d = 1.0/Hz_target)
            f0 = np.argmin(np.abs(freq_axis- f_range[0]))
            f1 = np.argmin(np.abs(freq_axis- f_range[1]))
            df = freq_axis[1]-freq_axis[0]
            
            ################# Compute fourier ################
            ff = np.abs(scipy.fft.fft(activity))**2
            
            ################# compute fraction of power and abstolute power ################
            p = np.sum(ff[f0:f1]) 
            ps.append(p)
            totp = np.sum(ff[1:normalize])
            fracp = (p/totp)*100
            fracps.append(fracp)

            ################# get peak ################
            ## find peaks for each roi
            peaks = find_peaks(ff[0:normalize])
            if len(peaks[0])>0:
                # Find the index from the maximum peak
                i_max_peak = peaks[0][np.argmax(ff[peaks[0]])]
                # Find the x value from that index
                # x_max = (2*np.abs(rfft(data))/N)[i_max_peak]
                x_max = rfftfreq(N, d = 1/Hz_target)[i_max_peak]
                
                freq_run.append(x_max)
         
        mean_absolute_power.append(ps)
        mean_frac_power.append(fracps)
        freq_block.append(freq_run)     
      
        
    return (mean_absolute_power,mean_frac_power,freq_block)  
            

    