# -*- coding: utf-8 -*-
"""
This script takes the raw conventional 2P data as input, align the stimulus with calcium activity
and stores the data in pkl files for analysis
"""

#%%  imports
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from _aux import loadmat_h5, get_outliers
import pandas as pd
import os
from scipy.io import loadmat
import h5py
import pickle
import functions as f
from scipy.signal import convolve

matplotlib.rcParams['axes.spines.right'] = False
matplotlib.rcParams['axes.spines.top'] = False


#%%

#############################################################################
# To reproduce figures: 
# GO THROUGH THE TODO AND FOLLOW THE INSTRUCTIONS THERE
#############################################################################


##### path to sound files
# TODO CHANGE THIS TO THE DESIRED FOLDER and file name
sound_dir = "Z:/Wayan/Lightbead/Flies/Method paper/2p upstairs/13122024/Fictrac_alignment/a2_r2/"
data = h5py.File(sound_dir +'20241213_1557.sound_server.h5', 'r')

#### path to I2C file
# TODO CHANGE THIS TO THE DESIRED FOLDER AND FILE NAME
I2C_dir = "Z:/Wayan/Lightbead/Flies/Method paper/2p upstairs/13122024/I2C_frames/a2_r2/"
# Name of I2C file
file = 'a2_r2.txt'

#### path to calcium data
# TODO CHANGE THIS TO THE DESIRED FOLDER
data_dir_dff = "V:/Wayan/LightBeadsMicroscopeDATA/Flies/Method paper data/Analyzed/2p upstairs/zscored/12132024/a2_r2/"
# Trial name
# TODO CHANGE THIS TO THE TRIAL'S NAME CONTAINED IN RAW TIFF FILE NAME
fly = ['12132024_6f_a2_r2'] 
### Number of volumes 
# TODO CHANGE THIS TO THE TRIAL's NUMBER OF VOLUME 
n_vol = 700

#### path to folder containing the auditory stimulus for plotting
# TODO CHANGE THIS TO THE DESIRED FOLDER
dropbox_dir = "C:/Users/wayan.CHRISTAPNI/Documents/Labs/Murthy/LightBead/Code/audio stimuli/Method paper/"

#### path to where data should be exported
# TODO CHANGE THIS TO THE DESIRED FOLDER
path_export = 'D:/Wayan/LightBead/method paper/dico data/zscored/RigE/supervoxels_1000/'

## Booleean to export data or not
generate_dic = False

### Specify which scope the data is coming from (LBM or 2p)
Scope = '2p'

### Name of the data to be exported
# TODO CHANGE THIS TO THE DESIRED NAME
fly_dic ='GCaMP6f_12132024_a2_r2'


### Define frame rate
if Scope == '2p':
    Hz = 2.20337115787
    frame_rate = 1/Hz 
    n_slice = 47
if Scope == 'LBM':
    Hz = 28.3
    frame_rate = 1/Hz 
    n_slice = 27

start_block_seconds = np.array([5,25,45,65,84,103,123,143,163,183,203,223,243])
end_block_seconds = np.array([15,35,55,75,94,113,133,153,173,193,213,233,253])
    



    
# %%
##########################
# Import auditory stimulus
##########################
#dropbox_dir = "C:/Users/wayan.CHRISTAPNI/Documents/Labs/Murthy/LightBead/Code/audio stimuli/Method paper/"

audio_stim = loadmat(dropbox_dir + 'highspeed_pulse_2_WG_paper_forplotting.mat')
stimTime = audio_stim['stim_time']

# Plot it
plt.figure(figsize = (15,5))

plt.fill_between(stimTime[0],y1=1.5*audio_stim['pulse_song'][0],y2=audio_stim['pulse_song'][0],where =audio_stim['pulse_song'][0]>0,color='r',alpha=0.5)
plt.fill_between(stimTime[0],y1=1.5*audio_stim['sine_song'][0],y2=audio_stim['sine_song'][0],where =audio_stim['sine_song'][0]>0,color='b',alpha=0.5)
plt.xlabel('Time(s)')    

    
# %%
####################################################
# Import sound info from the h5 files
####################################################

#sound_dir = "Z:/Wayan/Lightbead/Flies/Method paper/2p upstairs/13122024/Fictrac_alignment/a2_r2/"
#data = h5py.File(sound_dir +'20241213_1557.sound_server.h5', 'r')

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

#I2C_dir = "Z:/Wayan/Lightbead/Flies/Method paper/2p upstairs/13122024/I2C_frames/a2_r2/"
#file = 'a2_r2.txt'

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

if Scope == 'LBM':
    h5_file = data_dir_dff +  "/supervoxels_2000/" 
else:   
    h5_file = data_dir_dff +  "/supervoxels_1000/"                                 


dffs = []
for slice in range(0, n_slice):
    if Scope == 'LBM':
        fname = f'{fly[0]}_n2000_t{n_vol}_slice{slice}.h5'  
    else:  
        fname = f'{fly[0]}_n1000_t{n_vol}_slice{slice}.h5'    
                                   
    z_dff = loadmat_h5(os.path.join(h5_file, fname))['dff']

    for i, dff in enumerate(z_dff.T):
        dff[get_outliers(dff, 3)] = np.nan
        dff=pd.DataFrame(dff,columns=["dff"])
        dff = dff.interpolate(method='linear', limit_direction = 'both')
        dff = dff["dff"].to_numpy()
        dffs.append(dff)

dffs = np.array(dffs)      



################# Detrending
if Scope == '2p':
    X= np.arange(frame_rate,(dffs.shape[1]+frame_rate)*frame_rate,frame_rate)
    to_use = dffs[:,:]
    print('start detrending')
    dffs_detrend2 = np.zeros((to_use.shape[0],to_use.shape[1])) 
    for roi in range(dffs.shape[0]):
        z = np.polyfit(X,dffs[roi,:],3)
        p = np.poly1d(z)
        dffs_detrend2[roi,:] = dffs[roi,:]-p(X)
        if (roi/1000)%2 ==0 :
            print(roi)

#####################################################
# Truncate the times until the stimulus starts
#####################################################

index_start = int((time_audio-frames[0]/100) * Hz)+1

dffs_align =  dffs[:,index_start:]
if Scope == '2p':
    dffs_corrected = dffs_detrend2[:,index_start:]

time_activity_align = np.arange(frame_rate,(dffs_align.shape[1]+frame_rate)*frame_rate,frame_rate)  
time_activity= np.arange(frame_rate,(dffs.shape[1]+frame_rate)*frame_rate,frame_rate)



# %%
##############################################
# Extract auditory-correlated ROIs at 28 Hz
##############################################

# Creat stimulus with same number of time points as activity
stim = f.create_stim(dffs_align[:,:], start_block_seconds,end_block_seconds,Hz, t_i2c=0)  

# Create kernel
tau_rise = 0.050  # 50 ms rise time
tau_decay = 0.140  # 140 ms decay time
dt = frame_rate  

total_duration = np.round((dffs_align.shape[1]-1)/Hz) 
t = np.arange(0, total_duration, dt)


binary_stim = np.zeros_like(t)
start_first_block = 5  # seconds
block_duration = 10  # seconds
silence_duration = 10  # seconds
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
continuous_stim = convolve(binary_stim, kernel, mode='full')[:len(binary_stim)]

# Extract audio correlated ROIs to check alignment
if Scope == '2p':
    audio_correlated, coeffs, all_coeffs,sorted_indices = f.crosscorr_sort(dffs_corrected[:,:], continuous_stim, 0.2 ,Hz,0.0)
else:
    audio_correlated, coeffs, all_coeffs,sorted_indices = f.crosscorr_sort(dffs_align[:,:], continuous_stim, 0.2 ,Hz,0.0)
    

### Plot stim and activity to double check alignement
if Scope == '2p':
    plt.figure(figsize = (15,5))       
    plt.plot(time_activity_align, np.mean(dffs_corrected[audio_correlated,:],axis=0))
    max_trace = np.max(np.mean(dffs_corrected[audio_correlated,:],axis=0))
    plt.fill_between(stimTime[0],y1=max_trace*audio_stim['pulse_song'][0],y2=(max_trace*audio_stim['pulse_song'][0])+0.02,where =audio_stim['pulse_song'][0]>0,color='r',alpha=0.5)
    plt.fill_between(stimTime[0],y1=max_trace*audio_stim['sine_song'][0],y2=max_trace*audio_stim['sine_song'][0]+0.02,where =audio_stim['sine_song'][0]>0,color='b',alpha=0.5)
    plt.xlabel('Time (s)') 
    plt.ylabel('DF/F')

else:
    plt.figure(figsize = (15,5))       
    plt.plot(time_activity_align, np.mean(dffs_align[audio_correlated,:],axis=0))
    max_trace = np.max(np.mean(dffs_align[audio_correlated,:],axis=0))
    plt.fill_between(stimTime[0],y1=max_trace*audio_stim['pulse_song'][0],y2=(max_trace*audio_stim['pulse_song'][0])+0.02,where =audio_stim['pulse_song'][0]>0,color='r',alpha=0.5)
    plt.fill_between(stimTime[0],y1=max_trace*audio_stim['sine_song'][0],y2=max_trace*audio_stim['sine_song'][0]+0.02,where =audio_stim['sine_song'][0]>0,color='b',alpha=0.5)
    plt.xlabel('Time (s)')
    plt.ylabel('DF/F')


#%% Export data


if generate_dic:
    if Scope == '2p': 
        ### Data   
        dic = {'time_audio_aligned': stimTime[0] ,'time_audio_raw': stimTime[0] +time_audio,'pulse_song': audio_stim['pulse_song'][0], 'dffs_raw': dffs,\
              'time_activity_raw': time_activity,' time_activity_aligned': time_activity_align, 'dffs_corrected': dffs_corrected,  'dffs_aligned': dffs_align,  't_i2c':t_i2c, 'time_start_audio':time_audio,\
             'n_volumes_raw':dffs.shape[1],'n_volumes_aligned':dffs_align.shape[1], 'Scope': Scope, 'Frame_rate': Hz}
   
    else:
         dic = {'time_audio_aligned': stimTime[0],'time_audio_raw': stimTime[0] +time_audio,'pulse_song': audio_stim['pulse_song'][0], 'dffs_raw': dffs,\
          'time_activity_raw': time_activity,' time_activity_aligned': time_activity_align,  'dffs_aligned': dffs_align, 't_i2c':t_i2c, 'time_start_audio':time_audio,\
        'n_volumes_raw':dffs.shape[1],'n_volumes_aligned':dffs_align.shape[1], 'Scope': Scope, 'Frame_rate': Hz}
    
         
    with open(path_export+fly_dic +'.pkl', 'wb') as f:
       pickle.dump(dic, f)

    

