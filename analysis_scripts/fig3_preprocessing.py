# -*- coding: utf-8 -*-
"""
This script extracts the stimulus correlated ROIs and store them in pkl files for later analyses in Figure 3
It plots panel B of Fig3, and the associated supplemental figure S7b and S8
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

# Get correlation coeffs between ROIs from the Tdtomato channel
audio_correlated_red, coeffs_red, all_coeffs_red,sorted_indices_red = f.crosscorr_sort(dffs_all_red[:,:], continuous_stim, cutoff_corr ,Hz,0.0)

       
#%% Export audio correlated ROIs

if export == True:
    dic = {'audio_correlated': dffs_all[audio_correlated,:]}
     
     #TODO CHANGE THIS TO THE DESIRED FOLDER
    path = 'D:/Wayan/LightBead/method paper/clustering/zscored/supervoxels 2000/Red/'     
    with open(path + name_export +'.pkl', 'wb') as f:
    
       pickle.dump(dic, f)
   


#%% Plot distribution of correlation coefficient

###### Plot for the GCaMP channel ###############
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



### Plot for the Tdtomato channel in S7B ###############
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
        

depths_top05_b = depth_rois_all[sorted_indices_b]


# =============================
### Violin plot
# =============================
bin_width=np.array([10,14,12,12,12,11,11,10,10,11,11,10,10,10,10,10,10,10,8,8,9])
jitter_sd_frac=0.05
c= coeffs[idx_filter_b]
d = depths_top05_b

depth_min=54
depth_max=275
# Bin edges and centers
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
bin_colors = sm.to_rgba(centers_kept)   

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
    ax.scatter(x, vals,s=12, c=[col], edgecolors=['k'],alpha=0.85, zorder=3,linewidths=0.1)
    t+=1

ax.set_xlabel("Depth (µm)", fontsize = 16)
ax.set_ylabel("Correlation coefficient", fontsize = 16)
ax.set_xticks(centers_kept)
ax.set_xticklabels([f"{int(x - bin_width[i]/2)}–{int(x + bin_width[i]/2)}" for i,x in enumerate(centers_kept)], rotation=45, ha="right", fontsize=16)
ax.set_xticklabels([f"{int(bin_edges[i])}–{int(bin_edges[i+1])}" for i,x in enumerate(centers_kept)], rotation=45, ha="right", fontsize=16)
ax.tick_params(axis='y', labelsize=16)

ylim = ax.get_ylim()
y_text = ylim[0] - 0.02 * (ylim[1] - ylim[0])
for x0, n in zip(centers_kept, counts_kept):
    ax.text(x0, y_text, f"n={n}", ha="center", va="bottom", fontsize=12)
ax.set_ylim(y_text,)

fig.tight_layout()



# =============================
# HM
# =============================
### Sort ROIs by depth and by coeff if tie
order = np.lexsort((-coeffs[idx_filter_b], depths_top05_b))
dffs_sorted_depth    = dffs_all[sorted_indices_b,:][order]

to_plot_LB = zscore(dffs_sorted_depth[:,:7300],axis = 1)

cmap_base = 'viridis' 
vmin, vmax = -0.4, 1.1 
cmap = f.truncate_colormap(cmap_base, vmin, vmax)

plt.figure(figsize = (4.7,5.3)) 
im = plt.imshow(to_plot_LB, aspect = 'auto', vmin = -1, vmax = 1,cmap = cmap, extent = [0.035,258,0,1700])   
plt.tight_layout()
plt.colorbar(im)
plt.yticks([])
plt.xticks(fontsize = '16', color = 'w')
plt.fill_between(time_audio,y1=pulse_song+1725, y2=pulse_song +1745,where =pulse_song>0,color='r',alpha=1)
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
plt.xticks(color = 'k', fontsize = 16)  
plt.xlabel('Depth', fontsize = 18) 

