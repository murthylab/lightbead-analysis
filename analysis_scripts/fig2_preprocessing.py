# -*- coding: utf-8 -*-
"""
This script extracts the stimulus correlated ROIs and store them in pkl files for later analyses in Figure 3
It plots panel B of Fig3, and the associated supplemental figure S7 and S8
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
from matplotlib import cm, colors

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
    
    #depth at which each plane was aquired
    depth_a1 =np.array([275,267.1,259.2,250.3,240.4,230.5,220.6,210.7,199.8,189.9,179,168.2,157.3,147.4,168.2,158.3,148.4,137.5,126.6,115.7,103.9,92,80.1,67.2,53.4,38.6,24.7])
    depth_a2 =np.array([220,212.1,204.2,195.3,185.4,175.5,165.6,155.7,144.8,134.9,123,113.2,102.3,92.4,113.2,103.3,93.4,82.5,71.6,60.7,48.9,37,25.1,12.2,0,0,0])
    depth_a3 = depth_a1
    list_depth = [depth_a1,depth_a1,depth_a2,depth_a3,depth_a3,depth_a3]
    
    #TODO CHANGE THIS TO THE DESIRED FOLDER containng the pkl files containing the aligned data and the labels
    path_dico = 'D:/Wayan/LightBead/method paper/dico data/zscored/supervoxels_2000/'
    path_dico_red = 'D:/Wayan/LightBead/method paper/dico data/zscored/supervoxels_2000/Red/'
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
    data_red = pd.read_pickle(path_dico_red + dic)
    print('Run:', dic)
    
    if scope == 'LB':
        dffs = data['dffs_aligned'][:,:min_dim]
        dffs_red = data_red['dffs_aligned'][:,:min_dim]
        time_audio = data['time_audio_aligned']
        pulse_song = data['pulse_song']
        labels = loadmat_h5(os.path.join(path_labels, fname[i]))
        l = labels['labels']
        l = l.reshape((226,512,27))
        
        depth_rois = f.assign_depths(dffs.shape[0], list_depth[i])

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
        dffs_all_red = dffs_red
        depth_rois_all = depth_rois
    else:    
        dffs_all = np.vstack((dffs_all,dffs))
        dffs_all_red = np.vstack((dffs_all_red,dffs_red))
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

# Define GCaMP6f kernel 
kernel_duration = 1.0  
kernel_t = np.arange(0, kernel_duration, dt)
kernel = (1 - np.exp(-kernel_t / tau_rise)) * np.exp(-kernel_t / tau_decay)
kernel /= np.max(kernel)  # Normalize to peak at 1

#Convolve stimulus with kernel
continuous_stim = convolve(stim, kernel, mode='full')[:len(stim)]   


time_activity= np.arange(frame_rate,(dffs_all.shape[1]+frame_rate)*frame_rate,frame_rate)    

# Extract the top 0.5% of ROIs with highest correlation coeficient with the stimulus
audio_correlated, coeffs, all_coeffs,sorted_indices = f.crosscorr_sort(dffs_all[:], continuous_stim, cutoff_corr ,Hz,0.0)

# Plot the mean
plt.figure(figsize = (15,5))       
plt.plot(time_activity, np.mean(dffs_all[audio_correlated,:],axis=0))    
max_trace = np.max(np.mean(dffs_all[audio_correlated,:],axis=0))
plt.fill_between(time_audio,y1=max_trace*pulse_song,y2=(max_trace*pulse_song)+0.02,where =pulse_song>0,color='r',alpha=0.5)
plt.fill_between(time_audio,y1=max_trace*sine_song,y2=(max_trace*sine_song)+0.02,where =sine_song>0,color='b',alpha=0.5)
plt.title('Mean activity {}'.format(dic)) 
plt.xlabel('Time (s)')
plt.ylabel('DF/F')
'''
## Get the trial at which the ROIs was extracted from
count_1, count_2,count_3, count_4,count_5, count_6 = 0,0,0,0,0,0
trial1,trial2,trial3,trial4,trial5,trial6 = [],[],[],[],[],[]
for roi in sorted_indices:
    if roi<54000:
        count_1 += 1
        trial1.append(roi)
    if (roi>54000) and (roi<108000):
        count_2 += 1
        trial2.append(roi)
    if (roi>108000) and (roi<162000):
        count_3 += 1
        trial3.append(roi)
    if (roi>162000) and (roi<216000):
        count_4 += 1
        trial4.append(roi)
    if (roi>216000) and (roi<270000):
        count_5 += 1
        trial5.append(roi)
    if roi>270000:
        count_6 += 1
        trial6.append(roi)
        
27950//2000

## Here we check if any auditory ROIs were in slice 13,14,27
slice_13, slice_14,slice_27 = [],[],[]
count_13, count_14,count_27 = 0,0,0
for roi in sorted_indices:
    if (roi>24000) and (roi<26000):
        count_13 += 1
        slice_13.append(roi)
    if (roi>26000) and (roi<28000):
        count_14 += 1
        slice_14.append(roi)
    if (roi>52000) and (roi<54000):
        count_27 += 1
        slice_27.append(roi)
        
        
## Here we check if any auditory ROIs were in slice 1,2,15, 16      
slice_1, slice_2,slice_15,slice_16 = [],[],[],[]
count_1, count_2,count_15,count_16 = 0,0,0,0
for roi in sorted_indices:
    if (roi>0) and (roi<2000):
        count_1 += 1
        slice_1.append(roi)
    if (roi>2000) and (roi<4000):
        count_2 += 1
        slice_2.append(roi)
    if (roi>28000) and (roi<30000):
        count_15 += 1
        slice_15.append(roi)        
    if (roi>30000) and (roi<32000):
        count_16 += 1
        slice_16.append(roi)        


slice_15[10:]


slice_13, slice_14,slice_27 = [],[],[]
count_13, count_14,count_27 = 0,0,0
for roi in sorted_indices:
    if (roi>24000) and (roi<26000):
        count_13 += 1
        slice_13.append(roi)
    if (roi>78000) and (roi<80000):
        count_13 += 1
        slice_13.append(roi)
    if (roi>132000) and (roi<134000):
        count_13 += 1
        slice_13.append(roi)
    if (roi>186000) and (roi<188000):
        count_13 += 1
        slice_13.append(roi)
    if (roi>240000) and (roi<242000):
        count_13 += 1
        slice_13.append(roi)    
    if (roi>294000) and (roi<296000):
        count_13 += 1
        slice_13.append(roi)  
        
    if (roi>26000) and (roi<28000):
        count_14 += 1
        slice_14.append(roi)
    if (roi>80000) and (roi<82000):
        count_14 += 1
        slice_14.append(roi)
    if (roi>134000) and (roi<136000):
        count_14 += 1
        slice_14.append(roi)
    if (roi>188000) and (roi<190000):
        count_14 += 1
        slice_14.append(roi)
    if (roi>242000) and (roi<244000):
        count_14 += 1
        slice_14.append(roi)    
    if (roi>296000) and (roi<298000):
        count_14 += 1
        slice_14.append(roi)        
        
        
    if (roi>52000) and (roi<54000):
        count_27 += 1
        slice_27.append(roi)
    if (roi>106000) and (roi<108000):
        count_27 += 1
        slice_27.append(roi)
    if (roi>111000) and (roi<113000):
        count_27 += 1
        slice_27.append(roi)
    if (roi>165000) and (roi<167000):
        count_27 += 1
        slice_27.append(roi)
    if (roi>219000) and (roi<221000):
        count_27 += 1
        slice_27.append(roi)    
    if (roi>273000) and (roi<275000):
        count_27 += 1
        slice_27.append(roi) 
'''


       
#%% Export audio correlated ROIs

if export == True:
    dic = {'audio_correlated': dffs_all[audio_correlated,:]}
     
     #TODO CHANGE THIS TO THE DESIRED FOLDER
    path = 'D:/Wayan/LightBead/method paper/clustering/zscored/supervoxels 2000/Red/'     
    with open(path + name_export +'.pkl', 'wb') as f:
    
       pickle.dump(dic, f)
   


#%% Plot distribution of correlation coefficient

###### Plot for the GCaMP channel ###############
'''
# sort the array
all_coeffs_sorted = np.copy(all_coeffs)
sorted_indices = np.argsort(all_coeffs_sorted)
all_coeffs_sorted = all_coeffs_sorted[sorted_indices]
'''

if scope == 'LB':
    col = 'g.'
else:
    col = 'm.'    

plt.figure()
plt.plot(np.random.normal(0.5,0.005, size = len(all_coeffs)),all_coeffs,col,alpha = 0.3)
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


### Plot for the Tdtomato channel###############
# Get correlation coeffs between ROIs from the Tdtomato channel
audio_correlated_red, coeffs_red, all_coeffs_red,sorted_indices_red = f.crosscorr_sort(dffs_all_red[:,:], continuous_stim, cutoff_corr ,Hz,0.0)

'''
all_coeffs_sorted_red = np.copy(all_coeffs_red)
sorted_indices_red = np.argsort(all_coeffs_sorted_red)
all_coeffs_sorted_red = all_coeffs_sorted_red[sorted_indices_red]
'''

plt.figure()
plt.plot(np.random.normal(0.5,0.005, size = len(all_coeffs_red)),all_coeffs_red,col,alpha = 0.3)
plt.axhline(coeffs_red[0], linestyle ='--', color = 'k')
plt.xticks([])
plt.yticks(color = 'k')
plt.ylabel('Correlation coefficient',fontsize =24)
plt.ylim(-0.2,0.85)
plt.xlim(0.45, 0.55)
plt.yticks(fontsize =22)
plt.tight_layout()


#%% Plot correlation coefficient vs depth
all_coeffs_sorted = np.copy(all_coeffs)
sorted_indices_coeff = np.argsort(all_coeffs_sorted)
all_coeffs_sorted = all_coeffs_sorted[sorted_indices_coeff]
all_depths_sorted = depth_rois_all[sorted_indices_coeff]

#plt.figure(figsize = (10,10))
#plt.scatter(all_depths_sorted, all_coeffs_sorted)

## Only top 0.5
depths_top05 = depth_rois_all[sorted_indices]
depths_top05[np.argsort(depths_top05)]

#### Sorting by individual brain
depths_top05 = depth_rois_all[trial6]
depths_top05[np.argsort(depths_top05)]

from collections import defaultdict, deque
pos_map = defaultdict(deque)
for i,v in enumerate(sorted_indices):
    pos_map[v].append(i)
    
out = np.empty(len(trial6),dtype = int)
for i, v in enumerate(trial6):
    out[i] = pos_map[v].popleft()    
#######

## filter 04162025 out
sorted_indices_a = []
idx_filter_a=[]
for i,roi in enumerate(sorted_indices):
    if (roi<108000) or (roi>162000):
        sorted_indices_a.append(roi)
        idx_filter_a.append(i)
len(sorted_indices), len(sorted_indices_a), len(idx_filter_a)

#filter out s13,14
sorted_indices_s = []
idx_filter_s=[]
for i,roi in enumerate(sorted_indices):
    if (roi<=24000):
        idx_filter_s.append(i)
        sorted_indices_s.append(roi)
    if (roi>=28000) and (roi<=78000):
        idx_filter_s.append(i)
        sorted_indices_s.append(roi)        
    if (roi>=82000) and (roi<=132000):
        idx_filter_s.append(i)
        sorted_indices_s.append(roi)
    if  (roi>=136000) and (roi<=186000):
        idx_filter_s.append(i)
        sorted_indices_s.append(roi)
    if  (roi>=190000) and (roi<=240000):
        idx_filter_s.append(i)
        sorted_indices_s.append(roi)        
    if (roi>=244000) and (roi<=294000):
        idx_filter_s.append(i)
        sorted_indices_s.append(roi)        
    if  (roi>=298000):
        idx_filter_s.append(i)
        sorted_indices_s.append(roi) 
        
#filter out s13,14 and 04162025
sorted_indices_b = []
idx_filter_b=[]
for i,roi in enumerate(sorted_indices):
    if (roi<=24000):
        idx_filter_b.append(i)
        sorted_indices_b.append(roi)
    if (roi>=28000) and (roi<=78000):
        idx_filter_b.append(i)
        sorted_indices_b.append(roi)        
    if (roi>=82000) and (roi<=108000):
        idx_filter_b.append(i)
        sorted_indices_b.append(roi)
    if  (roi>=162000) and (roi<=186000):
        idx_filter_b.append(i)
        sorted_indices_b.append(roi)
    if  (roi>=190000) and (roi<=240000):
        idx_filter_b.append(i)
        sorted_indices_b.append(roi)        
    if (roi>=244000) and (roi<=294000):
        idx_filter_b.append(i)
        sorted_indices_b.append(roi)        
    if  (roi>=298000):
        idx_filter_b.append(i)
        sorted_indices_b.append(roi)         
        
len(sorted_indices),len(sorted_indices_a), len(idx_filter_a), len(sorted_indices_s), len(idx_filter_s),len(sorted_indices_b), len(idx_filter_b)

depths_top05_a = depth_rois_all[sorted_indices_a]
depths_top05_s = depth_rois_all[sorted_indices_s]
depths_top05_b = depth_rois_all[sorted_indices_b]

plt.figure(figsize = (10,10))
#plt.scatter(depths_top05, coeffs, color = 'g',alpha = 0.2)
plt.scatter(depths_top05_b, coeffs[idx_filter_b], color = 'g',alpha = 0.2)
plt.xlabel('Depth (um)', fontsize = 16)
plt.ylabel('Corr coeff', fontsize = 16)
plt.yticks(fontsize = 16)
plt.xticks(fontsize = 16)
plt.xlim(33.5,286.5)
plt.ylim(0,0.63)

np.unique(depths_top05)
depths_top05_b.shape,coeffs[idx_filter_b].shape
np.min(depths_top05)
# =============================
### Violin plot
# =============================
bin_width=30
#bin_width=np.array([14,13,12,11,12,11,11,11,10,10,11,10,10,10,10,10,10,10,9,8,12])
bin_width=np.array([10,14,12,12,12,11,11,10,10,11,11,10,10,10,10,10,10,10,8,8,9])
jitter_sd_frac=0.05
depths_top05 = depths_top05_b
d = all_depths_sorted #depths_top05 #all_depths_sorted
c = all_coeffs_sorted #[idx_filter_b] #all_coeffs_sorted
c= coeffs
# Determine depth range, snap to bin edges
depth_min = np.floor(d.min() / bin_width) * bin_width
if depth_min<0:
    depth_min = 0
depth_max = np.ceil(d.max() / bin_width) * bin_width

depth_min=54
depth_max=275
# Bin edges and centers
bin_edges = np.arange(depth_min, depth_max + bin_width, bin_width, dtype=float)
bin_centers = bin_edges[:-1] + bin_width / 2.0
#Manually define
bin_edges = np.array([60,74,86,98,110,121,132,143,153,163,174,185,195,205,215,225,235,245,255,263,271,280])
bin_centers = np.array([67,80,92,104,116,126,137,148,158,168,179,190,200,210,220,230,240,250,259,267,275])
# Assign each ROI to a bin index
bin_idx = np.digitize(d, bin_edges, right=False) - 1
data_per_bin = []
centers_kept = []
counts_kept = []
for i in range(len(bin_edges) - 1):
    vals = c[bin_idx == i]  
    data_per_bin.append(vals)
    centers_kept.append(bin_centers[i])
    counts_kept.append(vals.size)

centers_kept = np.asarray(centers_kept)

### If color code coeffs
vmin = np.nanmin(c)
vmax = np.nanmax(c)

### If color code depth
cmap="Greens"
norm = colors.Normalize(vmin=depth_min, vmax=depth_max)
sm = cm.ScalarMappable(norm=norm, cmap=cmap)
bin_colors = sm.to_rgba(centers_kept)   # RGBA color per kept bin

fig, ax = plt.subplots(figsize=(18, 5))
parts = ax.violinplot(
    data_per_bin,
    positions=centers_kept,
    widths=bin_width * 0.85,
    showmeans=True,
    showmedians=False,
    showextrema=False,
 )

# Color each violin body by its bin depth
for body, col in zip(parts['bodies'], bin_colors):
    body.set_facecolor(col); body.set_edgecolor(col); body.set_alpha(0.7)
if 'cmedians' in parts:
    parts['cmedians'].set_color("black") 
if 'cmeans' in parts:    
    parts['cmeans'].set_color("black")
    
    
rng = np.random.default_rng(0)
t = 0
for x0, vals, col in zip(centers_kept, data_per_bin, bin_colors):
    x = rng.normal(loc=x0, scale=bin_width[t] * jitter_sd_frac, size=vals.size)
    #ax.scatter(x, vals, s=8, alpha=0.35, color = 'b')
    #sc = ax.scatter(x, vals, s=10, alpha=0.7, zorder=3,c=vals, cmap=cmap, vmin=vmin, vmax=vmax)
    #ax.scatter(x, vals, s=10, alpha=0.85, zorder=3, c=[col])        
    ax.scatter(x, vals,s=12, c=[col], edgecolors=['k'],alpha=0.85, zorder=3,linewidths=0.1)
    t+=1

ax.set_xlabel("Depth (µm)", fontsize = 16)
ax.set_ylabel("Correlation coefficient", fontsize = 16)
ax.set_xticks(centers_kept)
ax.set_xticklabels([f"{int(x - bin_width[i]/2)}–{int(x + bin_width[i]/2)}" for i,x in enumerate(centers_kept)], rotation=45, ha="right", fontsize=16)
ax.set_xticklabels([f"{int(bin_edges[i])}–{int(bin_edges[i+1])}" for i,x in enumerate(centers_kept)], rotation=45, ha="right", fontsize=16)

ax.tick_params(axis='y', labelsize=16)
#ax.grid(True, axis='y', linestyle=':', alpha=0.5)

ylim = ax.get_ylim()
y_text = ylim[0] - 0.02 * (ylim[1] - ylim[0])
for x0, n in zip(centers_kept, counts_kept):
    ax.text(x0, y_text, f"n={n}", ha="center", va="bottom", fontsize=12)
ax.set_ylim(y_text,)

fig.tight_layout()

path = 'C:/Users/wayan.CHRISTAPNI/Princeton Dropbox/Wayan Gauthey/Princeton/Lightbead/Method paper/Figures/Panels/Figure 2/14012025/Revision round 1/Depth/Correct values/exluding 04162025/excluding s13 and s14/'
if save:
    plt.savefig(path + 'violin_coeffs_depth_noa2_nos13_14.pdf', transparent = True)



# =============================
# HM
# =============================


### Sort ROIs by depth and by coeff if tie
order = np.lexsort((-coeffs, depths_top05))
order = np.lexsort((-coeffs[idx_filter_b], depths_top05_b))
dffs_sorted_depth    = dffs_all[sorted_indices_b,:][order]
depths_sorted  = depths_top05[order]
coeffs_sorted  = coeffs[order]


all_depths_sorted.shape, coeffs.shape

sorted_indices.shape,depths_top05.shape

sorted_indices_depth_top05 = sorted_indices[np.argsort(depths_top05)]

to_plot_LB = zscore(dffs_sorted_depth[:,:7300],axis = 1)
to_plot_LB = zscore(dffs_all[sorted_indices_depth_top05,:7300],axis = 1)
to_plot_LB = np.flip(zscore(dffs_all[trial6],axis = 1),axis = 0)
to_plot_LB = temp2


cmap_base = 'viridis' #gnuplot
vmin, vmax = -0.4, 1.1  # -0.8, 1
cmap = f.truncate_colormap(cmap_base, vmin, vmax)

plt.figure(figsize = (4.7,5.3)) #(4,5)
im = plt.imshow(to_plot_LB, aspect = 'auto', vmin = -1, vmax = 1,cmap = cmap, extent = [0.035,258,0,1700])   
#im = plt.imshow(to_plot_LB, aspect = 'auto', cmap = 'viridis', extent = [0.035,258,0,1700])   
plt.tight_layout()
#plt.colorbar(im)
plt.yticks([])
plt.xticks(fontsize = '16', color = 'w')
plt.fill_between(time_audio,y1=pulse_song+1725, y2=pulse_song +1745,where =pulse_song>0,color='r',alpha=1)
#plt.xlabel('Time (s)', fontsize = '16')
plt.tight_layout()

path = 'C:/Users/wayan.CHRISTAPNI/Princeton Dropbox/Wayan Gauthey/Princeton/Lightbead/Method paper/Figures/Panels/Figure 2/14012025/Revision round 1/Depth/Correct values/exluding 04162025/excluding s13 and s14/'
if save:
    plt.savefig(path + 'HM_sorted_both.pdf', transparent = True)

#### For individual brains

depths_top05 = depth_rois_all[trial6]

from collections import defaultdict, deque
pos_map = defaultdict(deque)
for i,v in enumerate(sorted_indices):
    pos_map[v].append(i)
    
out = np.empty(len(trial6),dtype = int)
for i, v in enumerate(trial6):
    out[i] = pos_map[v].popleft()    


order = np.lexsort((-coeffs[out], depths_top05))
dffs_sorted_depth = dffs_all[trial6,:][order]


order.shape,dffs_sorted_depth.shape

to_plot_LB = zscore(dffs_sorted_depth,axis = 1)

cmap_base = 'viridis' #gnuplot
vmin, vmax = -0.4, 1.1  # -0.8, 1
cmap = f.truncate_colormap(cmap_base, vmin, vmax)

plt.figure(figsize = (4.7,5.3)) #(4,5)
im = plt.imshow(to_plot_LB, aspect = 'auto', vmin = -1, vmax = 1,cmap = cmap, extent = [0.035,258,0,400])   
plt.tight_layout()
#plt.colorbar(im)
plt.yticks([])
plt.xticks(fontsize = '16')
#plt.fill_between(time_audio_LB,y1=pulse_song+1725, y2=pulse_song +1745,where =pulse_song>0,color='r',alpha=1)
plt.xlabel('Time (s)', fontsize = '16')
plt.tight_layout()


### Here we build a line plot of depth to go along the heatmap
## get count of each depth
unique_elements, counts = np.unique(depths_top05_b, return_counts=True) 

for i in range(unique_elements.shape[0]):
    if i == 0:
        temp = np.ones((counts[i]))*unique_elements[i]
    else:
        temp = np.hstack((temp,np.ones((counts[i]))*unique_elements[i]))


plt.figure(figsize = (3,8))
plt.plot(temp,-np.arange(0,1488), color = 'k') 
plt.yticks([]) 
plt.ylim(-1488,0)  
plt.xlim(0,280)  
plt.xticks(color = 'w')   

path = 'C:/Users/wayan.CHRISTAPNI/Princeton Dropbox/Wayan Gauthey/Princeton/Lightbead/Method paper/Figures/Panels/Figure 2/14012025/Revision round 1/Depth/Correct values/exluding 04162025/excluding s13 and s14/'
if save:
    plt.savefig(path + 'depth_for_HM.pdf', transparent = True)
