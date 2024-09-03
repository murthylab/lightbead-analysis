# -*- coding: utf-8 -*-
"""
Created on Tue May 21 12:22:30 2024

@author: wayan

This script takes the output of supervoxels, the sound server files,
and the decoded message form I2C and align everything.
Then it creates a dictionary of the data for Rig E data
"""
#%%
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

#%%
### Fictrac
data_dir = "Z:/Wayan/Lightbead/Flies/Method paper/2p upstairs/07152024/Fictrac_alignment/a2_r2/"

##### Sound alignment
sound_dir = "Z:/Wayan/Lightbead/Flies/Method paper/2p upstairs/07152024/Fictrac_alignment/a2_r2/"
data = h5py.File(sound_dir +'20240715_1421.sound_server.h5', 'r')

########### I2C
I2C_dir = 'Z:/Wayan/Lightbead/Flies/Method paper/2p upstairs/07152024/I2C_frames/a2_r2/'

# Load the activity
data_dir_dff = "Z:/Wayan/Albert_to_run/2P RigE/wayan_a2_r2_green/"
data_dir_fictrac = "Z:/Wayan/Lightbead/Flies/Method paper/2p upstairs/07152024/Fictrac_alignment/a2_r2/"

generate_dic = False

Scope = '2p'
#define the metric to use to extract adn plot audio roi as a sanity check for alignment
stim_type = 'Pulse' #Pulse, sine-pulse
filters = 'threshold' # threshold, block, filters

#Define the type of stimulus that was presented
Natstim1 = False
Pulse2 = True
Sine_Pulse = False
Recorded = False

threshold_audio = 0.1
threshold_filters_audio = 0.6
threshold_audio_block = 0.3

Hz = 2.54
frame_rate = 1/Hz #0.0353356890459364 # 1/28.3


file = 'a2_r2.txt'

fly = ['wayan_a2_r2_green']   
n_vol = 649


fly_dic ='GCaMP6f_07152024_a2_r2_green'


if stim_type == 'Pulse':
    start_block_seconds = np.array([5,25,45,65,84,103,123,143,163,183,203,223,243])
    end_block_seconds = np.array([15,35,55,75,94,113,133,153,173,193,213,233,253])
    
   # start_index = start_bloc_seconds*44100
   # end_index = end_block_seconds*44100

if stim_type == 'sine-pulse':
    start_block_seconds = np.array([5,25,45,	65,	85,	105,125,145,165,185,205,224.9996,244.9992,264.9992])
    end_block_seconds = np.array([15,35,55,75,95,115,135,155,175,195,214.9996,234.9992,254.9992,274.9992])

    #start_bloc_seconds = np.array([5,25,45,	65,	85,	105,125,145,165,185,205,225,245,265])
    #end_block_seconds = np.array([15,35,55,75,95,115,135,155,175,195,215,235,255,275])

    #start_index = start_bloc_seconds*44100
    #end_index = end_block_seconds*44100   



# %%
##########################
# Import auditory stimuli
##########################

#Natstim1 = False
#Pulse2 = True
#Sine_Pulse = False
#Recorded = False

dropbox_dir = "C:/Users/wayan.CHRISTAPNI/Documents/Labs/Murthy/LightBead/Code/audio stimuli/Method paper/"

### natstim 1 (From Albert)
if Natstim1:
    audio_stim = loadmat(dropbox_dir + 'natural_stim_1_WG_paper_forplotting.mat')
    stimTime = audio_stim['stim_time']
    
### Natural stim 2 (From Albert)
if Pulse2:   
    audio_stim = loadmat(dropbox_dir + 'highspeed_pulse_2_WG_paper_forplotting.mat')
    stimTime = audio_stim['stim_time']
    
### Sine-Pulse (From Albert)
if Sine_Pulse:   
    audio_stim = loadmat(dropbox_dir + 'highspeed_sinepulse_3_WG_paper_forplotting.mat')
    stimTime = audio_stim['stim_time']
    
    
### Recorded song (From Albert)
if Recorded:   
    audio_stim = loadmat(dropbox_dir + 'Recorded_song_WG_paper_forplotting.mat')
    stimTime = audio_stim['stim_time']
    

    
    
 
# %%    
##########################
# Plot audio stim
##########################

plt.figure(figsize = (15,5))

plt.fill_between(stimTime[0],y1=1.5*audio_stim['pulse_song'][0],y2=audio_stim['pulse_song'][0],where =audio_stim['pulse_song'][0]>0,color='r',alpha=0.5)
plt.fill_between(stimTime[0],y1=1.5*audio_stim['sine_song'][0],y2=audio_stim['sine_song'][0],where =audio_stim['sine_song'][0]>0,color='b',alpha=0.5)
plt.xlabel('Time(s)')    


# %%
##########################
# Import sound info from the h5 files
##########################

for group in data.keys() :
    print ('group:',group)
    for dset in data[group].keys():      
        print ('dset:',dset)
        ds_data = data[group][dset] # returns HDF5 dataset object
        print ('ds_data:',ds_data)
        print (ds_data.shape, ds_data.dtype)
        arr = data[group][dset][:] # adding [:] returns a numpy array
        print (arr.shape, arr.dtype)
        print (arr)



# %%
############################################################################################
# import the text file with the time and frames from I2C
############################################################################################


File_data = np.loadtxt(I2C_dir + file, dtype=float) 
time = File_data[:,0]
frames = File_data[:,1]

### here we sort the time and frame arrays
paired_arrays = list(zip(time, frames))
paired_arrays_sorted = sorted(paired_arrays, key=lambda x: x[0])
time_sorted, frames_sorted = zip(*paired_arrays_sorted)
time_sorted = np.array(time_sorted)
frames_sorted = np.array(frames_sorted)

### Get the time when I2C starts, ie first time point in the message
t_i2c = time_sorted[0]
t_i2c


### Get the time when audio starts playing. For that we get the frame number when it starts and grab the corresponding time
frame_audio = arr[:,0][0]
print('Frame: ',frame_audio)
time_audio = time_sorted[frames_sorted == frame_audio][0]
print('Time:', time_audio)



# %%
############################################################################################
# Plot the frames we get from I2C and the audio h5 file as a sanity check
############################################################################################

figure, axs = plt.subplots(1,2, figsize = (15,5))

axs[0].plot(time, frames)
axs[0].set_xlabel('time (s)')
axs[0].set_ylabel('Frame number')
axs[0].set_title('Frames from I2C')

axs[1].plot(arr[:,3], arr[:,0])
axs[1].set_xlabel('Audio value')
axs[1].set_ylabel('Frame number')
axs[1].set_title('Frames and start audio')

plt.show()


# info frames
frames[0], frames[-1], time[0], time[-1]

# info stim audio
stimTime[0][0], stimTime[0][-1]

# info from h5 audio file
arr[0,0], arr[-1,0],arr[0,3], arr[-1,3]

stimTime.shape, File_data.shape, time.shape



# %%
############################################################################################
# Here we load the activity from the pipeline (supervoxel output)
############################################################################################

  
h5_file = data_dir_dff +  "/supervoxels1000/"                                     ##### change folder here
dffs = []
dffs_zscored = []


for slice in range(3, 48):
    fname = f'{fly[0]}_n1000_t{n_vol}_slice{slice}.h5'                                         ###### change n rois here
    z_dff = loadmat_h5(os.path.join(h5_file, fname))['z_dff']

    for i, dff in enumerate(z_dff.T):
        dff[get_outliers(dff, 3)] = np.nan
        #interpolate over the outliers
        dff=pd.DataFrame(dff,columns=["dff"])
        dff = dff.interpolate(method='linear', limit_direction = 'both')
        dff = dff["dff"].to_numpy()
        
        # z-score
        dff_zscore = (dff - np.nanmean(dff, axis=0)) / np.nanstd(dff, axis=0)
        
        #smoothing
        #dffs.append(gaussian_filter1d(dff, sigma=1))   
        
        dffs.append(dff)
        dffs_zscored.append(dff_zscore)
dffs = np.array(dffs)      
dffs_zscored = np.array(dffs_zscored)   


########### Here we background correct the activity for the aligned dffs
print('start detrending')
dffs_detrend = np.zeros((dffs.shape[0],dffs.shape[1]))    
# Compute background correction
for roi in range(dffs_detrend.shape[0]):
    _, spectra_arPLS, info = f.background_correction(dffs[roi,:], lam=1e4, niter=10)
    dffs_detrend[roi,:] = spectra_arPLS
    if (roi/1000)%2 ==0 :
            print(roi)
print('detrending done')



# %%
##########################
# Here we create the x axis for everything so that it's aligned
##########################

# Generate the time array for the activity based on the frame rate
#Hz = 28.2884

time_activity= np.arange(frame_rate,(dffs.shape[1]+frame_rate)*frame_rate,frame_rate)
#time_activity= np.linspace(frame_rate,(dffs.shape[1]+frame_rate)*frame_rate,n_vol)

#time_activity= np.linspace(0,int(time_sorted[-1])+1,n_vol)
#time_activity= np.arange(frame_rate,(dffs_truncated.shape[1]+frame_rate)*frame_rate,frame_rate)   
time_activity.shape

#t_diff = (time[0] * Hz) - int(time[0] * Hz)

#time_activity= np.arange(0,(dffs.shape[1])*frame_rate,frame_rate)
print('time_activity:',time_activity)
print('time_activity shape:',time_activity.shape)
print('dffs shape:',dffs.shape)


# shift the time array for audio by the time it took the audio to start
StimTime_shift = stimTime[0]   # +time_audio t_i2c
print('StimTime_shift:',StimTime_shift)
print('StimTime_shift shape:',StimTime_shift.shape)




# %%
##############################################
# Extract auditory-correlated ROIs at 28 Hz
##############################################


############### The folllowing takes the activity when stim is on or off and extract ROIs is activity when on is higher than odd by a given threshold
#Combine both pulse and sine song to have the time when any songs were present ############################


    # Plot mean activity of each run
    
if filters == 'threshold':    
    audio_correlated, mean_on = f.filter_threshold(dffs,threshold_audio, 'ON', audio_stim['pulse_song'][0], audio_stim['sine_song'][0], StimTime_shift, time_activity)  



if filters == 'block':
    audio_correlated, mean_activity = f.filter_block(dffs,[0],threshold_audio_block, start_block_seconds, end_block_seconds,Hz, StimTime_shift, audio_stim['pulse_song'][0], audio_stim['sine_song'][0], time_activity)




########### Plot mean activity
plt.figure(figsize = (15,5))       
plt.plot(time_activity, np.mean(dffs[audio_correlated,:],axis=0))
#plt.plot(time_activity, dffs[949,:]) 
    
#plt.plot(time_filter[:],conv/4, color = 'black')
max_trace = np.max(np.mean(dffs[audio_correlated,:],axis=0))
plt.fill_between(StimTime_shift,y1=max_trace*audio_stim['pulse_song'][0],y2=(max_trace*audio_stim['pulse_song'][0])+0.02,where =audio_stim['pulse_song'][0]>0,color='r',alpha=0.5)
plt.fill_between(StimTime_shift,y1=max_trace*audio_stim['sine_song'][0],y2=max_trace*audio_stim['sine_song'][0]+0.02,where =audio_stim['sine_song'][0]>0,color='b',alpha=0.5)
plt.title(Hz)  

f.sorted_heatmap_audio(dffs, audio_correlated,'1',0.3,0, 'viridis')




########################################################## 
## Export data into dictionary
#########################################################  


if generate_dic:
     
    ### Data   
    dic = {'time_audio': StimTime_shift, 'sine_song': audio_stim['sine_song'],'pulse_song': audio_stim['pulse_song'], 'dffs': dffs,'dffs_corrected' : dffs_detrend,'time_activity': time_activity,\
     't_i2c':t_i2c, 'time_start_audio':time_audio,  'stimulus': stim_type, 'n_volumes':dffs.shape[1], 'Scope': Scope, 'Frame_rate': Hz}

            
        #'time_speed':time_speed, 'speed':speed, , 'dffs_zscored':dffs_zscored

    path = 'D:/Wayan/LightBead/method paper/dico data/'     
    with open(path+fly_dic +'.pkl', 'wb') as f:
       pickle.dump(dic, f)
