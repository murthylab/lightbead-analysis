# -*- coding: utf-8 -*-
"""
Created on Mon Jun 30 15:44:13 2025

@author: wayan
"""

#%% Imports

import pandas as pd 
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import Functions as f
from scipy.stats import zscore
from scipy.io import loadmat
from scipy.signal import convolve
import pickle
from pandas import DataFrame



matplotlib.rcParams['axes.spines.right'] = False
matplotlib.rcParams['axes.spines.top'] = False

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


#%% Load data

#############################################################################
# To reproduce figures: 
# GO THROUGH THE TODO AND FOLLOW THE INSTRUCTIONS THERE
#############################################################################

#TODO CHANGE THIS TO THE DESIRED FOLDER containing the audio correlated data
path = 'D:/Wayan/LightBead/method paper/clustering/zscored/supervoxels 2000/' 
data_LB = pd.read_pickle(path + 'dffs_audio_LB_corr_top05_all'+'.pkl') 
data_2p = pd.read_pickle(path + 'dffs_audio_2p_corr_top05_all'+'.pkl') 

dffs_LB = data_LB['audio_correlated']
dffs_2p = data_2p['audio_correlated']

Hz_LB= 28.2893
fr_LB = 1/Hz_LB
time_activity_LB = np.arange(fr_LB,(dffs_LB.shape[1]+fr_LB)*fr_LB,fr_LB) 

Hz_2p = 2.20337115787
fr_2p = 1/Hz_2p
time_activity_2p  = np.arange(fr_2p,(dffs_2p.shape[1]+fr_2p)*fr_2p,fr_2p) 

#TODO CHANGE THIS TO THE DESIRED FOLDER containng the aligned data for LB
path_dico_LB = 'D:/Wayan/LightBead/method paper/dico data/supervoxels_500/'
list_dic = ['GCaMP6f_04032024_a2_r1.pkl']
fname = ['04032024_GCamp6f_a2_r1_w3_n1000_labels.h5']
data = pd.read_pickle(path_dico_LB + list_dic[0])
time_audio_LB = data['time_audio_aligned']
pulse_song = data['pulse_song']
sine_song = data['sine_song']
       
#TODO CHANGE THIS TO THE DESIRED FOLDER containing for 2p
path_dico = 'D:/Wayan/LightBead/method paper/dico data/zscored/RigE/supervoxels_1000/'
list_dic = ['GCaMP6f_12132024_a2_r2.pkl']
fname = ['06212024_6f_a1_r8_n500_labels.h5']
data = pd.read_pickle(path_dico + list_dic[0])
time_audio_2p = data['time_audio_aligned']

dropbox_dir = "C:/Users/wayan.CHRISTAPNI/Documents/Labs/Murthy/LightBead/Code/audio stimuli/Method paper/"
audio_stim = loadmat(dropbox_dir + 'highspeed_pulse_2_WG_paper_forplotting.mat')
time_audio_2p = audio_stim['stim_time'][0]
pulse_song = audio_stim['pulse_song'][0]


start_block_seconds_LB = np.array([5,25,45,65,84,103,123,143,163,183,202.99894,222.99788,242.99788])
end_block_seconds_LB = np.array([15,35,55,75,94,113,133,153,173,192.99894,212.99788,232.99788,252.99788])

start_block_seconds_2p = np.array([5,25,45,65,84,103,123,143,163,183,202.99894,222.99788,242.99788])
end_block_seconds_2p = np.array([15,35,55,75,94,113,133,153,173,192.99894,212.99788,232.99788,252.99788])


#%% 

################################################################
# Plot distribution of peaks and fourier spectrum panels
################################################################

# For LB
peaks_freq_LB = f.peaks_fourier_ROI_combine(zscore(dffs_LB,axis = 1),start_block_seconds_LB,end_block_seconds_LB,0,28.2893,time_activity_LB,'LB',282,1, None)
f.plot_distribution_peaks_fourier_ROIs(peaks_freq_LB,'LB', 'g',None)


# For 2p
peaks_freq_2p = f.peaks_fourier_ROI_combine(zscore(dffs_2p,axis = 1),start_block_seconds_2p,end_block_seconds_2p,2,2.2,time_activity_2p,'2p',282,1, None)
f.plot_distribution_peaks_fourier_ROIs(peaks_freq_2p,'2p', 'm',None)




#%% 

################################################################
# Plot heatmaps and ROI vs thresold panels
################################################################


list_dic = ['GCaMP6f_04032024_a2_r1.pkl' ,'GCaMP6f_04032024_a2_r5.pkl'] 
path_dico_LB = 'D:/Wayan/LightBead/method paper/dico data/zscored/supervoxels_2000/'

# Threshold to extract audio ROIs
cutoff_corr = [6,6.7]
         
min_dim = 7977

# Create kernel
tau_rise = 0.050  # 50 ms rise time
tau_decay = 0.140  # 140 ms decay time
dt = fr_LB  
kernel_duration = 1.0  
kernel_t = np.arange(0, kernel_duration, dt)
kernel = (1 - np.exp(-kernel_t / tau_rise)) * np.exp(-kernel_t / tau_decay)
kernel /= np.max(kernel)  

# Values for ROIs vs threshold panels
threshold_test = np.arange(0.0,1,0.005)

cutoff_05 = 0.5
line = [0.0287,0.043]



for i, dic in enumerate(list_dic):
    data = pd.read_pickle(path_dico_LB + dic)
    print('Run:', dic)
    
    ####################################################
    ### Extract auditory correlated ROIs and plot heatmap
    ####################################################
    
    dffs = data['dffs_aligned'][:,:min_dim] 
    time_audio = data['time_audio_aligned']
    pulse_song = data['pulse_song']

    time_activity= np.arange(fr_LB,(dffs.shape[1]+fr_LB)*fr_LB,fr_LB)  

    stim = f.create_stim(dffs, start_block_seconds_LB,end_block_seconds_LB,Hz_LB, t_i2c=0)
    #Convolve stimulus with kernel
    continuous_stim = convolve(stim, kernel, mode='full')[:len(stim)] 
    time_filter = np.arange(0,len(stim))/Hz_LB 
    conv = np.convolve (stim, kernel, mode = 'same')
    conv = conv/np.max(conv) 
    
    #Extract audio correlated ROIs
    audio_correlated, coeffs, all_coeffs, sort_i = f.crosscorr_sort(zscore(dffs, axis = 1), conv, cutoff_corr[i] ,Hz_LB,0.0)

    cut_off_audio = int(265*Hz_LB)
    to_plot = np.flip(zscore(dffs[audio_correlated,:cut_off_audio], axis = 1),axis = 0)
    
    plt.figure(figsize = (5,5))
    im = plt.imshow(to_plot, aspect = 'auto', vmin = -1.0, vmax = 1.0,cmap = 'viridis',extent = [0,265,0,3617])
    plt.xticks(fontsize = 18)
    plt.yticks([])
    plt.fill_between(time_audio,y1=pulse_song+3676, y2=pulse_song+3712,where =pulse_song>0,color='r',alpha=1)
    plt.tight_layout()
    plt.xlabel('Time (s)',fontsize = 18)
    plt.ylabel('ROIs',fontsize = 18)
    plt.tight_layout()
    
    ####################################################
    ### Plot ROIs vs threshold
    ####################################################
    n_roi, c = f.crosscorr_sort_corr(zscore(dffs, axis = 1), conv, threshold_test,cutoff_05 ,Hz_LB)

    plt.figure(figsize = (15,10))
    plt.plot(threshold_test,np.array(n_roi)/1000,color = 'k', lw = 3.5)
    plt.xlabel('Correlation coefficient',fontsize = 22)
    plt.ylabel('# ROIs',fontsize = 22)
    plt.xticks(fontsize = 22)
    plt.yticks(fontsize = 22)
    plt.tight_layout()
    plt.axvline(x=line[i], color = 'r', linestyle = '--', alpha = 1,lw = 2.5)
    plt.axvline(x=c[0], color = 'k', linestyle = '--', alpha = 1,lw = 2.5)
    
    # Plot zoom in panel
    plt.figure(figsize = (15,3))
    plt.plot(threshold_test,np.array(n_roi)/1000, color = 'k',lw = 3.5)
    plt.xlabel('Correlation coefficient',fontsize = 22)
    plt.ylabel('# ROIs',fontsize = 22)
    plt.xlim(0,0.2)
    if i == 0:
        plt.ylim(0,7)
    else:   
        plt.ylim(0,15)
    plt.axvline(x=line[i], color = 'r', linestyle = '--', alpha = 1,lw = 2.5)
    plt.axvline(x=c[0], color = 'k', linestyle = '--', alpha = 1,lw = 2.5)
    plt.xticks(fontsize = 22)
    plt.yticks(fontsize = 22)
    plt.locator_params(axis='y', nbins=3)
    plt.tight_layout()   
 
    

    
#%% 

################################################################
# Extract ROIs by comparing to a null distribution
################################################################

## Load data
list_dic = ['GCaMP6f_04032024_a2_r1.pkl' ,'GCaMP6f_04032024_a2_r5.pkl','GCaMP6f_04192024_a1_r2.pkl'] 
path_dico_LB = 'D:/Wayan/LightBead/method paper/dico data/zscored/supervoxels_2000/'
min_dim = 7977

n_shuffle = 1000

for i, dic in enumerate(list_dic[0:1]):

    data = pd.read_pickle(path_dico_LB + dic)
    print('Run:', dic)
    
    dffs = data['dffs_aligned'][:,:min_dim] 
    time_audio = data['time_audio_aligned']
    pulse_song = data['pulse_song']
    
    time_activity= np.arange(fr_LB,(dffs.shape[1]+fr_LB)*fr_LB,fr_LB)  
    
    
    #1) z score dffs and the stimulus
    dffs_z = zscore(dffs,axis = 1)
    
    stim = f.create_stim(dffs_z, start_block_seconds_LB,end_block_seconds_LB,Hz_LB, t_i2c=0)
    # Create kernel
    tau_rise = 0.050  # 50 ms rise time
    tau_decay = 0.140  # 140 ms decay time
    dt = fr_LB  
    kernel_duration = 1.0  
    kernel_t = np.arange(0, kernel_duration, dt)
    kernel = (1 - np.exp(-kernel_t / tau_rise)) * np.exp(-kernel_t / tau_decay)
    kernel /= np.max(kernel)  
    #Convolve stimulus with kernel
    continuous_stim = convolve(stim, kernel, mode='full')[:len(stim)] 
    time_filter = np.arange(0,len(stim))/Hz_LB 
    conv = np.convolve (stim, kernel, mode = 'same')
    #conv = conv/np.max(conv)
    conv_z = zscore(conv) 
    
    #2) compute correlation coefficient of each ROI with the stimulus
    audio_correlated, coeffs, all_coeffs, sort_i = f.crosscorr_sort(dffs_z, conv_z, 100 ,Hz_LB,0.0)
  
    
    # 3) shuffle the stimulus using circular shift N times and get correlation coefficient for each shuffle
    r_null, shifts = circular_shift_null_corr_prestandardized(dffs_z, conv_z,n_shuffles=40,exclude_lags=0,seed=0,batch_size=256,dtype=np.float32)
    
    # Build allowed shifts: ON 10s / OFF 10s => period=20s; pick E=4–5s
    T = conv_z.size
    allowed = allowed_circ_shifts(T, Hz_LB, period_sec=20.0, E_sec=7.0)
    # Sample the shifts you want from the allowed set:
    rng = np.random.default_rng(0)
    my_shifts = rng.choice(allowed, size=10000, replace=True)
    # Compute null correlations: 
    r_null, shifts_used = circular_shift_null_corr_prestandardized(dffs_z, conv_z,n_shuffles=10000, shifts=my_shifts,batch_size=256,dtype=np.float32)
    
    r_null, perms = block_permute_null_corr_prestandardized(dffs_z, conv_z,fs=Hz_LB,n_shuffles=300,block_sec=10.0,jitter_within_block=int(0.0*Hz_LB),seed=0,batch_size=128,dtype=np.float32,forbid_identity_perm=False)
    ## Exclude lag around xs
    #r_null, shifts_used = circular_shift_null_corr_prestandardized(dffs_z, conv_z,n_shuffles=200,exclude_lags=int(5.0 * Hz_LB),seed=0,batch_size=256)
   
    '''#3) Randomly shuffle each ROI N times and get correlation coefficient for each shuffle
    coeffs_shuffle_all = np.zeros((dffs_z.shape[0],n_shuffle))
    
    for i in range(n_shuffle):
        # shuffle all rows
        d_shuffled = shuffle_rows_numpy(dffs_z, seed=i, key_dtype=np.float32)
        # extract correlation coefficient
        audio_correlated_shuffle, coeffs_shuffle, all_coeffs_shuffle, sort_i_shuffle = f.crosscorr_sort(d_shuffled, conv_z, 100 ,Hz_LB,0.0)
        # store coefficients
        coeffs_shuffle_all[:,i] = all_coeffs_shuffle
    '''
 
    #4)  Compute p values
    pvals = permutation_pvals_two_sided(all_coeffs, coeffs_shuffle_all)
    pvals = permutation_pvals_two_sided(all_coeffs, r_null)
    pvals  = permutation_pvals_one_sided(all_coeffs, r_null, alternative="greater")
    
    ## Plot hist of pvalues
    plt.figure()
    plt.hist(pvals, bins = 200)
    
    plt.figure()
    plt.hist(r_null, bins = 200)
    
    np.sum(pvals<0.05)

    '''#4) Randomly shuffle the stimulus N times and get correlation coefficient for each shuffle
    coeffs_shuffle_all_stim = np.zeros((n_shuffle,dffs_z.shape[0]))
    for i in range(20):
        # shuffle the stimulus
        s_cop =np.copy(conv_z)
        s_shuffled = random.shuffle(s_cop)
        # extract correlation coefficient
        audio_correlated_shuffle_stim, coeffs_shuffle_stim, all_coeffs_shuffle_stim, sort_i_shuffle_stim = f.crosscorr_sort(dffs_z, s_cop, 100 ,Hz_LB,0.0)
        # store coefficients
        coeffs_shuffle_all_stim[i,:] = all_coeffs_shuffle_stim
    
    #4)  Compute p values
    pvals_stim = permutation_pvals_two_sided(all_coeffs_shuffle_stim, coeffs_shuffle_all_stim)
    '''


import Functions as f

## Plot the HM
path = 'D:/Wayan/LightBead/method paper/dico data/zscored/Suppl corr/' 
data_LB = pd.read_pickle(path + 'GCaMP6f_04032024_a2_r5_pval_6s'+'.pkl') 
dff_z = data_LB['dffs_auditory_z']
#Sort the ROIs
audio_correlated2, coeffs2, all_coeffs2, sort_i2 = f.crosscorr_sort(dffs_z[np.where(pvals<0.05)[0],:], conv_z, 100 ,Hz_LB,0.0)

to_plot_LB = np.flip(dffs_z[np.where(pvals<0.05)[0],:][audio_correlated2,:7300],axis = 0)
to_plot_LB = np.flip(dffs_z[sig_mask,:7300],axis = 0)

cmap_base = 'viridis' #gnuplot
vmin, vmax = -0.4,1.0  # for first submission values were (-0.4,1.1)       -0.8, 1
cmap = f.truncate_colormap(cmap_base, vmin, vmax)

plt.figure(figsize = (4.7,5.3)) #(4,5)
im = plt.imshow(to_plot_LB, aspect = 'auto', vmin = -1, vmax = 1,cmap = cmap, extent = [0.035,258,0,1700])   
#im = plt.imshow(to_plot_LB, aspect = 'auto', vmin = -1, vmax = 1,cmap = 'viridis', extent = [0.035,258,0,1700])   
plt.tight_layout()
#plt.colorbar(im)
plt.yticks([])
plt.xticks(color = 'w')
plt.fill_between(time_audio_LB,y1=pulse_song+1725, y2=pulse_song +1745,where =pulse_song>0,color='r',alpha=1)
#plt.xlabel('Time (s)')
plt.tight_layout()

path = 'C:/Users/wayan.CHRISTAPNI/Princeton Dropbox/Wayan Gauthey/Princeton/Lightbead/Method paper/Figures/Panels/Figure 2/14012025/Revision round 1/suppl corr/'
if save:
    plt.savefig(path + 'HM_04032024_a2_r1_7s.pdf', transparent = True)


### Save correlated ROIs
auditory_roi = np.where(pvals<0.05)
dffs_auditory_z = dffs_z[np.where(pvals<0.05)[0],:][audio_correlated2,:7300]
dffs_auditory = dffs[np.where(pvals<0.05)[0],:][audio_correlated2,:7300]


dic = {'auditory_roi': auditory_roi,'dffs_auditory_z': dffs_auditory_z,'dffs_auditory': dffs_auditory, 'pvals':pvals, 'r_null':r_null, 'n_rois': np.sum(pvals<0.05)}

path_export ='D:/Wayan/LightBead/method paper/dico data/zscored/Suppl corr/'  
fly_dic = 'GCaMP6f_04032024_a2_r1_pval_7s'
with open(path_export+fly_dic +'.pkl', 'wb') as f:
   pickle.dump(dic, f)
   
df = DataFrame(auditory_roi[0], columns = ['ROI'])
df.to_csv('D:/Wayan/LightBead/method paper/dico data/zscored/Suppl corr/audio_roi_04032024_6f_a2_r1_pval_7s.csv')   
   
   
plt.figure()
plt.plot(time_activity_LB[:7300], dffs_z[np.where(pvals_bh<0.05)[0],:][audio_correlated2[-500],:7300])
plt.fill_between(time_audio_LB,y1=pulse_song+np.max(dffs_z[np.where(pvals_bh<0.05)[0],:][audio_correlated2[-500],:7300]), y2=pulse_song,where =pulse_song>0,color='r',alpha=1)


   
   
   
   
   


############################### Correct for mutiple comparaison

plt.figure()
plt.plot(zscore(dffs_z,axis = 1)[audio_correlated2[-1],:7300], color = 'k')


#5) Compute FDR
sig_mask, pvals_bh = fdr_bh(pvals, q=0.05)
np.sum(sig_mask)
np.sum(pvals_bh<0.05)
np.min(pvals)

sel, t_star, fdr_at_t, curve = permutation_fdr_on_stat(all_coeffs, coeffs_shuffle_all, q=0.05, two_sided=True, mode="pooled")
np.sum(sel)
sel, t_star, fdr_at_t, curve = permutation_fdr_on_stat(all_coeffs, coeffs_shuffle_all_stim[:20,:], q=0.05, two_sided=True, mode="aligned")
np.sum(sel)

out = plot_permutation_fdr_diagnostics(all_coeffs, coeffs_shuffle_all_stim[:20,:], q=0.05, two_sided=False, mode="aligned")
out = plot_permutation_fdr_diagnostics(all_coeffs, coeffs_shuffle_all, q=0.05, two_sided=False, mode="pooled")
sel_mask = out["sel_mask"]     # boolean array of selected ROIs
t_star   = out["t_star"]       # chosen threshold on |r|
fdr_hat  = out["fdr_at_t_star"]
np.sum(sel_mask)



from typing import Optional, Tuple
def circular_shift_null_corr_with_shifts(Xu, s_u, shifts, batch_size=256, dtype=np.float32):
    """
    Xu: (n_rois, T) unit-L2, mean-zero ROI (or we can normalize inside)
    s_u: (T,) unit-L2, mean-zero stimulus
    shifts: (n_shuffles,) integers in [0, T)
    """
    Xu = np.asarray(Xu, dtype=dtype, order="C")
    s_u = np.asarray(s_u, dtype=dtype).reshape(-1)
    n_rois, T = Xu.shape
    assert s_u.shape[0] == T
    shifts = np.asarray(shifts, dtype=np.int64)

    r_null = np.empty((n_rois, shifts.size), dtype=dtype)
    t = np.arange(T, dtype=np.int64)[:, None]
    for start in range(0, shifts.size, batch_size):
        end = min(start + batch_size, shifts.size)
        k = shifts[start:end]
        idx = (t - k[None, :]) % T              # (T,B)
        S_batch = s_u[idx]                      # (T,B), already unit-L2
        r_null[:, start:end] = Xu @ S_batch     # (n_rois,B)
    return r_null

def allowed_circ_shifts(T, fs, period_sec, E_sec):
    period = int(round(period_sec * fs))
    E = int(round(E_sec * fs))
    allowed = np.ones(T, dtype=bool)
    if period > 0:
        n_mult = int(np.ceil(T / period)) + 1
        for k in range(n_mult):
            center = (k * period) % T
            lo = (center - E) % T
            hi = (center + E) % T
            if lo <= hi:
                allowed[lo:hi+1] = False
            else:
                allowed[:hi+1] = False
                allowed[lo:] = False
    # ensure at least one shift remains
    if not np.any(allowed):
        raise ValueError("Exclusion too wide—no shifts left.")
    return np.flatnonzero(allowed)

def circular_shift_null_corr_prestandardized(
    dffs_z: np.ndarray,           # (n_rois, T) mean-zero (z-scored) ROI traces
    stim_z: np.ndarray,           # (T,)       mean-zero (z-scored) stimulus
    n_shuffles: int,
    exclude_lags: int = 0,        # ignored when 'shifts' is provided
    seed: Optional[int] = None,
    batch_size: int = 256,        # shifts per batch (tune for RAM/BLAS)
    dtype: np.dtype = np.float32,
    shifts: Optional[np.ndarray] = None,  # (n_shuffles,) specific circular shifts to use
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Pearson correlations between each ROI and circularly-shifted stimulus versions.

    If 'shifts' is provided (ints in [0, T)), it is used directly and 'exclude_lags' is ignored.
    Otherwise, draws 'n_shuffles' random shifts, optionally excluding small lags.

    Returns
    -------
    r_null : (n_rois, n_shuffles)  correlations (one column per shift)
    shifts : (n_shuffles,)         the shift used for each column (np.int64)
    """
    # --- inputs ---
    X = np.asarray(dffs_z, dtype=dtype, order="C")
    s = np.asarray(stim_z, dtype=dtype).reshape(-1)

    n_rois, T = X.shape
    if s.shape[0] != T:
        raise ValueError("stim_z length must equal number of columns in dffs_z")

    # --- unit-L2 normalize once (already mean-zero) ---
    Xnorm = np.linalg.norm(X, axis=1, keepdims=True).astype(dtype)
    Xnorm[Xnorm == 0] = 1.0
    Xu = X / Xnorm                         # (n_rois, T)

    s_norm = float(np.linalg.norm(s))
    if s_norm == 0:
        # degenerate stimulus: all correlations are zero
        return np.zeros((n_rois, n_shuffles), dtype=dtype), np.zeros(n_shuffles, dtype=np.int64)
    s_u = s / s_norm                       # (T,)

    # --- decide shifts ---
    if shifts is not None:
        shifts = np.asarray(shifts, dtype=np.int64).ravel()
        if shifts.size != n_shuffles:
            raise ValueError("len(shifts) must equal n_shuffles.")
        # bring into [0, T)
        shifts %= T
    else:
        rng = np.random.default_rng(seed)
        if exclude_lags <= 0:
            shifts = rng.integers(0, T, size=n_shuffles, endpoint=False, dtype=np.int64)
        else:
            # disallow shifts in [-exclude_lags, +exclude_lags] modulo T
            mask = np.ones(T, dtype=bool)
            mask[:exclude_lags+1] = False
            if exclude_lags > 0:
                mask[T-exclude_lags:] = False
            allowed = np.nonzero(mask)[0].astype(np.int64)
            if allowed.size == 0:
                raise ValueError("Exclusion window too large: no shifts remain.")
            shifts = rng.choice(allowed, size=n_shuffles, replace=True)

    # --- allocate output ---
    r_null = np.empty((n_rois, n_shuffles), dtype=dtype)

    # --- batched matmul core ---
    t = np.arange(T, dtype=np.int64)[:, None]   # (T,1)
    for start in range(0, n_shuffles, batch_size):
        end = min(start + batch_size, n_shuffles)
        k = shifts[start:end]                   # (B,)
        idx = (t - k[None, :]) % T              # (T, B)  column b = rolled by k[b]
        S_batch = s_u[idx]                      # (T, B), unit-L2 columns
        r_null[:, start:end] = Xu @ S_batch     # (n_rois, B)

    return r_null, shifts


def _make_block_indices(T: int, block_len: int) -> np.ndarray:
    """
    Return BI of shape (block_len, n_blocks) with absolute indices per block.
    Pads the final block by repeating its last index so each column has block_len rows.
    """
    if block_len <= 0:
        raise ValueError("block_len must be positive")
    n_blocks = int(np.ceil(T / block_len))
    pad = n_blocks * block_len - T
    idx = np.arange(T, dtype=np.int64)
    if pad > 0:
        idx = np.concatenate([idx, np.full(pad, idx[-1], dtype=np.int64)])
    return idx.reshape(n_blocks, block_len).T  # (block_len, n_blocks)

def block_permute_null_corr_prestandardized(
    dffs_z: np.ndarray,          # (n_rois, T), mean-zero (z-scored)
    stim_z: np.ndarray,          # (T,), mean-zero (z-scored)
    fs: float,                   # Hz
    n_shuffles: int,
    block_sec: float = 10.0,     # seconds
    jitter_within_block: int = 0,# ±samples to circularly roll inside each block
    seed: Optional[int] = None,
    batch_size: int = 128,
    dtype: np.dtype = np.float32,
    forbid_identity_perm: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Permute stimulus in contiguous blocks (optionally with within-block jitter) and
    compute Pearson r for each ROI vs each permuted stimulus.
    Returns:
      r_null : (n_rois, n_shuffles)
      perms  : (n_shuffles, n_blocks)
    """
    X = np.asarray(dffs_z, dtype=dtype, order="C")
    s = np.asarray(stim_z,  dtype=dtype).reshape(-1)

    n_rois, T = X.shape
    if s.shape[0] != T:
        raise ValueError("stim_z length must equal number of columns in dffs_z")

    # Unit-L2 normalize (inputs are already mean-zero)
    Xnorm = np.linalg.norm(X, axis=1, keepdims=True).astype(dtype)
    Xnorm[Xnorm == 0] = 1.0
    Xu = X / Xnorm

    s_norm = float(np.linalg.norm(s))
    if s_norm == 0:
        return np.zeros((n_rois, n_shuffles), dtype=dtype), np.empty((n_shuffles, 0), dtype=np.int64)
    s_u = s / s_norm

    # Build block layout
    block_len = int(round(block_sec * fs))
    if block_len <= 0:
        raise ValueError("block_sec too small for the given fs")
    BI = _make_block_indices(T, block_len)      # (block_len, n_blocks)
    block_len_eff, n_blocks = BI.shape

    rng = np.random.default_rng(seed)
    r_null = np.empty((n_rois, n_shuffles), dtype=dtype)
    perms_used = np.empty((n_shuffles, n_blocks), dtype=np.int64)

    # Pre-allocate index matrix for the stimulus in this batch: **T × B** (fixed length)
    for start in range(0, n_shuffles, batch_size):
        end = min(start + batch_size, n_shuffles)
        B = end - start

        # Draw block permutations (avoid identity if requested and possible)
        perms = np.empty((B, n_blocks), dtype=np.int64)
        for b in range(B):
            if forbid_identity_perm and n_blocks == 1:
                # Only one block -> identity is unavoidable; allow it.
                perms[b] = np.array([0], dtype=np.int64)
            else:
                while True:
                    p = rng.permutation(n_blocks)
                    if not forbid_identity_perm or np.any(p != np.arange(n_blocks)):
                        break
                perms[b] = p

        # Per-(shuffle, block) jitters
        if jitter_within_block > 0:
            J = int(jitter_within_block)
            jit = rng.integers(-J, J + 1, size=(B, n_blocks), endpoint=True, dtype=np.int64)
        else:
            jit = np.zeros((B, n_blocks), dtype=np.int64)

        # Assemble absolute indices for each permuted stimulus (column)
        idx_batch = np.empty((T, B), dtype=np.int64)   # <-- FIX: allocate (T, B), not padded length
        for b in range(B):
            cols = []
            for j in range(n_blocks):
                col = BI[:, perms[b, j]]
                if jitter_within_block != 0:
                    col = np.roll(col, jit[b, j], axis=0)  # roll within the block
                cols.append(col)
            idx_b = np.concatenate(cols, axis=0)[:T]   # truncate padding back to T
            idx_batch[:, b] = idx_b                    # shapes now match (T,)

        # Gather permuted stimulus and correlate
        S_batch = s_u[idx_batch]                       # (T, B)
        r_null[:, start:end] = Xu @ S_batch            # (n_rois, B)
        perms_used[start:end] = perms

    return r_null, perms_used



def permutation_pvals_one_sided(r_obs, r_null, alternative="greater"):
    """
    One-sided permutation p-values with finite-sample correction.

    p_i = (1 + #{ r_null >= r_obs_i }) / (N_i + 1)      if alternative == "greater"
    p_i = (1 + #{ r_null <= r_obs_i }) / (N_i + 1)      if alternative == "less"

    Parameters
    ----------
    r_obs : array-like, shape (n_rois,)
        Observed statistics (e.g., correlations) per ROI.
    r_null : array-like, shape (n_rois, n_perm) or (n_perm,)
        Null statistics from permutations. If 2D, each row is that ROI's null.
        If 1D, a pooled null used for all ROIs.
    alternative : {"greater","less"}, default "greater"
        Direction of the one-sided test.

    Returns
    -------
    pvals : ndarray, shape (n_rois,)
        One-sided permutation p-values.
    """
    r_obs = np.asarray(r_obs, dtype=np.float64).reshape(-1)
    r_null = np.asarray(r_null, dtype=np.float64)

    if alternative not in ("greater", "less"):
        raise ValueError("alternative must be 'greater' or 'less'.")

    if r_null.ndim == 1:
        # Pooled null
        valid = ~np.isnan(r_null)
        N = int(valid.sum())
        if N == 0:
            return np.ones_like(r_obs)
        rn = r_null[valid][None, :]  # (1, N)
        if alternative == "greater":
            counts = (rn >= r_obs[:, None]).sum(axis=1)
        else:
            counts = (rn <= r_obs[:, None]).sum(axis=1)
        pvals = (1.0 + counts) / (N + 1.0)
        return pvals

    elif r_null.ndim == 2:
        if r_null.shape[0] != r_obs.shape[0]:
            raise ValueError("For per-ROI nulls, r_null must have shape (n_rois, n_perm).")
        valid = ~np.isnan(r_null)                       # (n_rois, n_perm)
        N = valid.sum(axis=1).astype(np.int64)          # (n_rois,)

        # Broadcast compare with care about NaNs
        if alternative == "greater":
            ge = (r_null >= r_obs[:, None]) & valid
        else:
            ge = (r_null <= r_obs[:, None]) & valid
        counts = ge.sum(axis=1)

        # Avoid division by zero (rows with all-NaN nulls → p=1)
        denom = N + 1.0
        denom[denom == 0] = np.inf
        pvals = (1.0 + counts) / denom
        pvals[np.isinf(denom)] = 1.0
        return pvals

    else:
        raise ValueError("r_null must be 1D (pooled) or 2D (per-ROI).")


from typing import Tuple, Dict, Optional

def permutation_fdr_on_stat(
    r_obs: np.ndarray,
    r_null: np.ndarray,
    q: float = 0.05,
    two_sided: bool = True,
    mode: str = "auto",      # "aligned", "pooled", or "auto"
    min_rois: int = 1,       # require at least this many discoveries
    return_curve: bool = True
) -> Tuple[np.ndarray, float, float, Optional[Dict[str, np.ndarray]]]:
    """
    Permutation-based FDR control directly on the statistic (|r|).

    We pick a threshold t on |r| such that:
        FDR_hat(t) = E_null[# |r_null| >= t] / # { |r_obs| >= t }  <= q
    and select all ROIs with |r_obs| >= t*.

    Parameters
    ----------
    r_obs : (n_rois,)
        Observed correlation coefficients per ROI.
    r_null : (n_perm, n_rois)  OR  (n_rois, n_perm)
        Null correlations from permutations.
        - If shape is (n_perm, n_rois): ALIGNED permutations (recommended).
        - If shape is (n_rois, n_perm): INDEPENDENT/POOLED nulls (per-ROI shuffles).
    q : float, default 0.05
        Target FDR level.
    two_sided : bool, default True
        If True, use |r|. If False, use r (right-tailed).
    mode : {"auto","aligned","pooled"}, default "auto"
        How to estimate expected false positives under the null.
        - "aligned": uses per-permutation counts V_b(t) then averages across permutations.
        - "pooled": uses the pooled null distribution across all ROIs and permutations.
        - "auto": choose "aligned" if r_null looks like (n_perm, n_rois); else "pooled".
    min_rois : int, default 1
        Require at least this many discoveries; otherwise return no selections.
    return_curve : bool, default True
        If True, return diagnostic arrays (threshold grid, R(t), Vbar(t), FDR_hat(t)).

    Returns
    -------
    sel_mask : (n_rois,) bool
        Boolean mask of selected ROIs.
    t_star : float
        Chosen threshold on (|r| if two_sided else r). If no threshold meets FDR, returns np.inf.
    fdr_at_t_star : float
        Estimated FDR at the chosen threshold (np.nan if no discoveries).
    extras : dict or None
        If return_curve=True, contains:
            - 't_grid': thresholds tested (descending)
            - 'R': observed exceedances at each t
            - 'Vbar': expected null exceedances at each t
            - 'FDR_hat': Vbar / R
        Else None.
    """
    r_obs = np.asarray(r_obs).reshape(-1)
    n_rois = r_obs.size

    if two_sided:
        s_obs = np.abs(r_obs)
    else:
        s_obs = r_obs.copy()

    r_null = np.asarray(r_null)
    if r_null.ndim != 2:
        raise ValueError("r_null must be 2D: (n_perm, n_rois) or (n_rois, n_perm).")

    # Decide operating mode
    if mode == "auto":
        if r_null.shape[1] == n_rois:          # (n_perm, n_rois) likely
            mode_eff = "aligned"
        elif r_null.shape[0] == n_rois:        # (n_rois, n_perm) likely
            mode_eff = "pooled"
        else:
            raise ValueError("Cannot infer mode from r_null shape; set mode='aligned' or 'pooled'.")
    else:
        mode_eff = mode
    if mode_eff not in {"aligned", "pooled"}:
        raise ValueError("mode must be 'aligned', 'pooled', or 'auto'.")

    # Build statistic for nulls
    if two_sided:
        if mode_eff == "aligned":
            # Expecting (n_perm, n_rois). If the array is (n_rois, n_perm), transpose.
            if r_null.shape[1] != n_rois and r_null.shape[0] == n_rois:
                r_null = r_null.T  # now (n_perm, n_rois)
            s_null = np.abs(r_null)  # (n_perm, n_rois)
        else:  # pooled
            if r_null.shape[0] == n_rois:
                s_null = np.abs(r_null.reshape(n_rois, -1))   # (n_rois, n_perm)
            else:
                s_null = np.abs(r_null.T)                     # (n_rois, n_perm)
    else:
        if mode_eff == "aligned":
            if r_null.shape[1] != n_rois and r_null.shape[0] == n_rois:
                r_null = r_null.T
            s_null = r_null
        else:
            if r_null.shape[0] == n_rois:
                s_null = r_null.reshape(n_rois, -1)
            else:
                s_null = r_null.T

    # Threshold grid: evaluate on the unique observed stats (descending).
    # This yields the smallest t with FDR_hat(t) <= q (i.e., the largest set of ROIs).
    t_grid = np.unique(s_obs[~np.isnan(s_obs)])
    if t_grid.size == 0:
        # No valid stats
        return np.zeros(n_rois, dtype=bool), np.inf, np.nan, (
            None if not return_curve else {"t_grid": np.array([]), "R": np.array([]), "Vbar": np.array([]), "FDR_hat": np.array([])}
        )
    t_grid.sort()
    t_grid = t_grid[::-1]  # descending

    # Observed exceedances R(t): compute once by sorting s_obs
    s_obs_sorted = np.sort(s_obs)[::-1]
    # For each threshold equals s_obs_sorted[k], R = k+1. We'll map back by searchsorted.
    # Build R(t) for each t in t_grid using descending order positions:
    # For speed, use np.searchsorted on the ASCENDING array with -t to get count of >= t.
    s_obs_sorted_asc = np.sort(s_obs)  # ascending
    R = s_obs_sorted_asc.size - np.searchsorted(s_obs_sorted_asc, t_grid, side='left')

    # Expected null exceedances Vbar(t)
    if mode_eff == "aligned":
        # s_null: (n_perm, n_rois). For each permutation b, count V_b(t) = #(s_null[b] >= t),
        # then average over permutations.
        # We can do this efficiently by sorting each permutation once (ascending),
        # then using searchsorted for all t. A small loop over permutations is fine.
        n_perm = s_null.shape[0]
        Vbar = np.zeros_like(t_grid, dtype=float)
        # Sort ascending per perm
        s_null_sorted_asc = np.sort(s_null, axis=1)  # (n_perm, n_rois)
        # For each perm, compute counts at all t via vectorized searchsorted, then average
        # (loop only over n_perm, searchsorted over all thresholds per loop)
        for b in range(n_perm):
            counts_b = s_null_sorted_asc.shape[1] - np.searchsorted(s_null_sorted_asc[b], t_grid, side='left')
            Vbar += counts_b
        Vbar /= n_perm

    else:  # pooled
        # s_null: (n_rois, n_perm) but independent across ROIs.
        # Approximate E_null[#exceedances] as m * P(|r_null| >= t) using the pooled ECDF.
        pooled = s_null.reshape(-1)
        pooled = pooled[~np.isnan(pooled)]
        pooled.sort()  # ascending
        # For each t, number of pooled values >= t:
        ge_counts = pooled.size - np.searchsorted(pooled, t_grid, side='left')
        # Convert to expected false positives among m ROIs
        p_ge = ge_counts / max(pooled.size, 1)
        Vbar = n_rois * p_ge

    # Estimated FDR curve
    with np.errstate(divide='ignore', invalid='ignore'):
        FDR_hat = Vbar / np.maximum(R, 1)

    # Pick the smallest t (i.e., most inclusive) with FDR_hat <= q and R >= min_rois
    valid = (FDR_hat <= q) & (R >= min_rois)
    if np.any(valid):
        # Choose the *last* True in the DESCENDING t_grid (i.e., the smallest t)
        idx = np.where(valid)[0][-1]
        t_star = float(t_grid[idx])
        fdr_at_t_star = float(FDR_hat[idx])
        sel_mask = s_obs >= t_star
    else:
        t_star = float('inf')
        fdr_at_t_star = float('nan')
        sel_mask = np.zeros(n_rois, dtype=bool)

    extras = None
    if return_curve:
        extras = {"t_grid": t_grid, "R": R.astype(float), "Vbar": Vbar.astype(float), "FDR_hat": FDR_hat.astype(float)}
    return sel_mask, t_star, fdr_at_t_star, extras

def plot_permutation_fdr_diagnostics(r_obs, r_null, q=0.05, two_sided=True, mode="auto", min_rois=1):
    """
    Plots the diagnostic curves for permutation-based FDR on |r|:
      - R(t): #observed exceedances
      - V̄(t): expected false exceedances under null
      - FDR_hat(t) = V̄(t)/R(t), with horizontal line at q
      - Vertical line at chosen threshold t*
    """
    sel, t_star, fdr_star, curve = permutation_fdr_on_stat(
        r_obs, r_null, q=q, two_sided=two_sided, mode=mode, min_rois=min_rois, return_curve=True
    )
    t = curve["t_grid"]; R = curve["R"]; Vbar = curve["Vbar"]; FDR_hat = curve["FDR_hat"]

    # --- Panel 1: R(t) vs V̄(t) ---
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(t, R, label="R(t): observed ≥ t")
    ax[0].plot(t, Vbar, label="V̄(t): expected null ≥ t")
    if np.isfinite(t_star):
        ax[0].axvline(t_star, linestyle="--", label=f"t* = {t_star:.3f}")
    ax[0].set_xlabel("threshold t on |r|" if two_sided else "threshold t on r")
    ax[0].set_ylabel("count")
    ax[0].set_title("Observed vs expected exceedances")
    ax[0].legend()

    # --- Panel 2: FDR_hat(t) ---
    ax[1].plot(t, FDR_hat, label="FDR̂(t)")
    ax[1].axhline(q, linestyle=":", label=f"target q = {q}")
    if np.isfinite(t_star):
        ax[1].axvline(t_star, linestyle="--", label=f"t* = {t_star:.3f} (FDR̂={fdr_star:.3f})")
    ax[1].set_xlabel("threshold t on |r|" if two_sided else "threshold t on r")
    ax[1].set_ylabel("estimated FDR")
    ax[1].set_title("FDR̂(t) curve")
    ax[1].set_ylim(0, max(q*2, np.nanmax(FDR_hat)*1.1) if np.isfinite(np.nanmax(FDR_hat)) else 1)
    ax[1].legend()

    plt.tight_layout()
    return {"sel_mask": sel, "t_star": t_star, "fdr_at_t_star": fdr_star, "curve": curve}




# 6) plot the extracted ROIs

# First we sort them bz coeff
audio_correlated_o, coeffs_o, all_coeffs_o, sort_i_o = f.crosscorr_sort(dffs_z, conv_z, 0.5 ,Hz_LB,0.0)
audio_correlated, coeffs, all_coeffs, sort_i = f.crosscorr_sort(dffs_z[pvals<0.05,:], conv_z, 100 ,Hz_LB,0.0)


to_plot = np.flip(zscore(dffs_z[pvals<0.05,:][audio_correlated,:], axis = 1),axis = 0)

plt.figure(figsize = (5,5))
im = plt.imshow(to_plot, aspect = 'auto', vmin = -1.0, vmax = 1.0,cmap = 'viridis')
plt.xticks(fontsize = 18)
plt.yticks([])
#plt.fill_between(time_audio,y1=pulse_song+3676, y2=pulse_song+3712,where =pulse_song>0,color='r',alpha=1)
plt.tight_layout()
plt.xlabel('Time (s)',fontsize = 18)
plt.ylabel('ROIs',fontsize = 18)
plt.tight_layout()

roiss = np.arange(0,54000)
roiss[pvals<0.05]


count = 0
for i, roi in enumerate(sort_i_o):
    if roi in roiss[pvals<0.05]:
        count +=1

sort_i[-1]
sort_i_o[-1]

plt.figure()
plt.plot(dffs_z[pvals<0.05,:][audio_correlated[-1],:])

def shuffle_rows_numpy(X, seed=0, key_dtype=np.float32):
    rng  = np.random.default_rng(seed)
    keys = rng.random(X.shape, dtype=key_dtype)         # ~ size(X)
    idx  = np.argsort(keys, axis=1, kind='quicksort')   # int64 indices (~2× size(X) if X is float32)
    del keys                                             # free keys before the gather
    Xsh  = np.take_along_axis(X, idx, axis=1)           # new array same dtype/shape as X
    return Xsh



plt.figure(figsize = (15,5))
plt.plot(dffs_z[0,:], color = 'k')

plt.figure(figsize = (15,5))
plt.plot(d_shuffled[0,:], color = 'b')


def permutation_pvals_two_sided(r_obs, r_null):
    """
    Compute two-sided permutation p-values:
      p_i = (1 + #{ |r_null| >= |r_obs_i| }) / (N_i + 1)

    Parameters
    ----------
    r_obs : array-like, shape (n_rois,)
        Observed correlations per ROI.
    r_null : array-like, shape (n_rois, n_perm)  OR  (n_perm,)
        Null correlations from shuffles. If 2D, each row corresponds to that ROI's null.
        If 1D, a pooled null used for all ROIs.

    Returns
    -------
    pvals : ndarray, shape (n_rois,)
        Two-sided permutation p-values for each ROI.
    """
    r_obs = np.asarray(r_obs, dtype=np.float64).reshape(-1)
    abs_obs = np.abs(r_obs)

    r_null = np.asarray(r_null, dtype=np.float64)

    if r_null.ndim == 1:
        # Pooled null, broadcast to all ROIs
        abs_null = np.abs(r_null)[None, :]                     # (1, n_perm)
        valid = ~np.isnan(abs_null)                            # (1, n_perm)
        N = valid.sum(axis=1).astype(np.int64)                 # (1,)
        ge = (abs_null >= abs_obs[:, None]) & valid            # (n_rois, n_perm)
        counts = ge.sum(axis=1)                                # (n_rois,)
        pvals = (1.0 + counts) / (N[0] + 1.0)                  # scalar N for all ROIs
        return pvals

    elif r_null.ndim == 2:
        if r_null.shape[0] != r_obs.shape[0]:
            raise ValueError("For per-ROI nulls, r_null must have shape (n_rois, n_perm).")
        abs_null = np.abs(r_null)                              # (n_rois, n_perm)
        valid = ~np.isnan(abs_null)                            # (n_rois, n_perm)
        N = valid.sum(axis=1).astype(np.int64)                 # (n_rois,)
        # Compare each ROI's |obs| to its own null distribution
        ge = (abs_null >= abs_obs[:, None]) & valid            # (n_rois, n_perm)
        counts = ge.sum(axis=1)                                # (n_rois,)
        pvals = (1.0 + counts) / (N + 1.0)                     # (n_rois,)
        return pvals

    else:
        raise ValueError("r_null must be 1D (pooled) or 2D (per-ROI).")
        
        
def fdr_bh(pvals, q=0.05):
    """
    Benjamini-Hochberg FDR control.
    Returns a boolean mask of discoveries and the BH-adjusted p-values.
    """
    pvals = np.asarray(pvals, dtype=np.float64)
    n = pvals.size
    order = np.argsort(pvals)
    ranked = pvals[order]
    thresh = q * (np.arange(1, n + 1) / n)
    is_sig = ranked <= thresh
    if np.any(is_sig):
        k = np.max(np.where(is_sig)[0])
        cutoff = ranked[k]
        discoveries = pvals <= cutoff
    else:
        discoveries = np.zeros_like(pvals, dtype=bool)

    # Adjusted p-values
    adj = np.empty_like(ranked)
    # monotone step-up
    adj[-1] = ranked[-1] * n / n
    for i in range(n - 2, -1, -1):
        adj[i] = min(adj[i + 1], ranked[i] * n / (i + 1))
    adj_p = np.empty_like(adj)
    adj_p[order] = adj
    return discoveries, adj_p      



# ==================================================
# Reproduce Diego-s method
# ==================================================
from sklearn.linear_model import Ridge
from statsmodels.stats.multitest import multipletests


fs = Hz_LB  # <-- set to your actual imaging Hz (e.g., 28.3 if that’s the case)

result = run_pulse_encoding_pipeline(
    dffs=dffs_z,
    stim=conv_z,
    fs=fs,
    lag_sec=8.0,
    bin_sec=0.1,      # ~100 ms lag spacing; L ≈ 80 columns
    K=15,
    alphas=np.logspace(-3, 2, 20),
    B=1000,           # 1000 shuffles per ROI (increase later if you like)
    chunk_sec=10.0,   # paper’s 10 s chunks
    fdr_q=0.01,
    prescreen_frac=0.2,  # process top 20% by |corr| first; set None to run all
    rng_seed=0
)

sel_idx = result["sel_idx"]         # indices of significant ROIs
filters = result["filters"]         # temporal filters (per ROI)
lags_s = result["lag_samples"] / fs # lags in seconds for plotting




# -------------------- helpers --------------------
def build_lagged_design(stim, fs, lag_sec=8.0, bin_sec=0.1):
    """
    One-channel (pulse) design with nonnegative time lags up to lag_sec, sampled every bin_sec.
    Returns:
      X: (T, L) design; lags: (L,) offsets in samples (0 ... ~lag_sec)
    """
    T = stim.shape[0]
    step = max(1, int(round(bin_sec * fs)))     # samples between adjacent lags
    L = max(1, int(round(lag_sec * fs / step))) # number of lag columns
    lags = np.arange(L) * step                  # nonnegative offsets
    pad = np.zeros(lags[-1], dtype=stim.dtype) if L > 1 else np.zeros(0, dtype=stim.dtype)
    stimp = np.concatenate([pad, stim])
    # X[t, j] = stim[t - lags[j]]  (with left padding)
    X = np.stack([stimp[(lags[-1] - d):(lags[-1] - d + T)] for d in lags], axis=1)
    return X.astype(np.float32), lags

def contiguous_kfold_indices(T, K=15):
    """Return list of test-index arrays for K contiguous folds over length T."""
    fold_sizes = np.full(K, T // K, dtype=int)
    fold_sizes[:T % K] += 1
    idx = np.arange(T)
    out, start = [], 0
    for s in fold_sizes:
        out.append(idx[start:start+s])
        start += s
    return out

def ridge_cv_predict_per_roi(X, Y, alphas, folds, center_X=True):
    """
    For each ROI (row of Y), pick alpha by median CV corr, and return:
      g_pred (n_rois, T), rhos_cv (n_rois, K), alpha_best (n_rois,), coefs (n_rois, P)
    """
    T, P = X.shape
    n_rois = Y.shape[0]
    K = len(folds)
    all_idx = np.arange(T)
    trte = [(np.setdiff1d(all_idx, te, assume_unique=True), te) for te in folds]

    g_pred = np.zeros_like(Y, dtype=np.float32)
    rhos_cv = np.full((n_rois, K), np.nan, dtype=np.float32)
    alpha_best = np.empty(n_rois, dtype=np.float32)
    coefs = np.zeros((n_rois, P), dtype=np.float32)

    # Precompute global X mean to speed up final refit
    Xmu_all = X.mean(axis=0, keepdims=True).astype(np.float32)
    Xc_all = (X - Xmu_all) if center_X else X

    for i in range(n_rois):
        y = Y[i].astype(np.float32)
        corrs_by_alpha = []

        # cross-validate over ridge alphas
        for alpha in alphas:
            fold_corrs = []
            for k, (tr, te) in enumerate(trte):
                Xtr, Xte = X[tr], X[te]
                ytr, yte = y[tr], y[te]

                if center_X:
                    mu = Xtr.mean(axis=0, keepdims=True).astype(np.float32)
                    Xtr_c = Xtr - mu
                    Xte_c = Xte - mu
                else:
                    Xtr_c, Xte_c = Xtr, Xte

                ymu = float(ytr.mean())
                ytr_c = ytr - ymu
                yte_c = yte - ymu

                model = Ridge(alpha=float(alpha), fit_intercept=False)
                model.fit(Xtr_c, ytr_c)
                g = model.predict(Xte_c).astype(np.float32)

                num = float(np.dot(yte_c, g))
                den = float(np.linalg.norm(yte_c) * np.linalg.norm(g) + 1e-12)
                fold_corrs.append(num / den)
            corrs_by_alpha.append(fold_corrs)

        corrs_by_alpha = np.array(corrs_by_alpha, dtype=np.float32)  # (n_alpha, K)
        j_best = int(np.nanargmax(np.median(corrs_by_alpha, axis=1)))
        alpha_star = float(alphas[j_best])
        alpha_best[i] = alpha_star
        rhos_cv[i] = corrs_by_alpha[j_best]

        # Recompute OOS predictions for alpha_star and store them
        for k, (tr, te) in enumerate(trte):
            Xtr, Xte = X[tr], X[te]
            ytr = y[tr]
            if center_X:
                mu = Xtr.mean(axis=0, keepdims=True).astype(np.float32)
                Xtr_c = Xtr - mu
                Xte_c = Xte - mu
            else:
                Xtr_c, Xte_c = Xtr, Xte
            ymu = float(ytr.mean())
            ytr_c = ytr - ymu

            model = Ridge(alpha=alpha_star, fit_intercept=False)
            model.fit(Xtr_c, ytr_c)
            g_pred[i, te] = (model.predict(Xte_c) + ymu).astype(np.float32)

        # Final filter on all data (for inspection)
        y_full_c = (y - y.mean()).astype(np.float32)
        model_full = Ridge(alpha=alpha_star, fit_intercept=False)
        model_full.fit(Xc_all, y_full_c)
        coefs[i] = model_full.coef_.astype(np.float32)

    return g_pred, rhos_cv, alpha_best, coefs

def chunk_shuffle_indices(T, chunk_len):
    """Return a permutation index that shuffles 1D indices in contiguous chunks of length chunk_len."""
    n_chunks = int(np.ceil(T / chunk_len))
    pad = n_chunks * chunk_len - T
    idx = np.arange(T)
    if pad > 0:
        idx = np.concatenate([idx, np.full(pad, idx[-1])])
    chunks = idx.reshape(n_chunks, chunk_len)
    perm = np.random.permutation(n_chunks)
    return chunks[perm].ravel()[:T]

def pvalues_chunk_shuffle(F, G_pred, fs, folds, B=1000, chunk_sec=10.0, percentile=30.0, rng=None):
    """
    Paper-style p-values:
      - Compute fold-wise OOS correlations between F and G_pred (per ROI).
      - T = 30th percentile of those K correlations.
      - Shuffle F in 10 s chunks, B times; for each shuffle take median corr across folds.
      - p = fraction of shuffle medians >= T.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    n_rois, T = F.shape
    K = len(folds)

    # True fold-wise correlations and T (30th percentile)
    rho_cv = np.full((n_rois, K), np.nan, dtype=np.float32)
    for i in range(n_rois):
        for k, te in enumerate(folds):
            y = F[i, te]; g = G_pred[i, te]
            y_c = y - y.mean(); g_c = g - g.mean()
            num = float(np.dot(y_c, g_c))
            den = float(np.linalg.norm(y_c) * np.linalg.norm(g_c) + 1e-12)
            rho_cv[i, k] = num / den
    Tq = np.nanpercentile(rho_cv, percentile, axis=1)

    # Shuffled null
    chunk_len = max(1, int(round(chunk_sec * fs)))
    pvals = np.zeros(n_rois, dtype=np.float64)
    for i in range(n_rois):
        null_meds = np.empty(B, dtype=np.float32)
        for b in range(B):
            sh_idx = chunk_shuffle_indices(T, chunk_len)
            sF = F[i, sh_idx]
            rb = []
            for k, te in enumerate(folds):
                y = sF[te]; g = G_pred[i, te]
                y_c = y - y.mean(); g_c = g - g.mean()
                num = float(np.dot(y_c, g_c))
                den = float(np.linalg.norm(y_c) * np.linalg.norm(g_c) + 1e-12)
                rb.append(num / den)
            null_meds[b] = np.median(rb)
        pvals[i] = (null_meds >= Tq[i]).mean()
    return pvals, rho_cv

# -------------------- main pipeline (Option A: one filter) --------------------
def run_pulse_encoding_pipeline(
    dffs,                 # (n_rois, 7977)
    stim,                 # (7977,)
    fs=28.0,              # imaging Hz (set your actual frame rate)
    lag_sec=8.0,
    bin_sec=0.1,
    K=15,
    alphas=np.logspace(-3, 2, 20),
    B=1000,
    chunk_sec=10.0,
    fdr_q=0.01,
    prescreen_frac=None,  # e.g., 0.2 to process top 20% by |corr(ROI, stim)|
    rng_seed=0
):
    assert dffs.shape[1] == stim.shape[0] == 7977, "Time axis must be 7977 in this dataset."

    # Optional prescreen to save time: keep top ROIs by |corr(ROI, stim)|
    roi_idx = np.arange(dffs.shape[0])
    if prescreen_frac is not None:
        # fast z-scored correlation
        y = dffs - dffs.mean(axis=1, keepdims=True)
        y /= (np.linalg.norm(y, axis=1, keepdims=True) + 1e-12)
        s = stim - stim.mean(); s = s / (np.linalg.norm(s) + 1e-12)
        r_fast = (y @ s.astype(np.float32))
        keep = int(max(1, round(prescreen_frac * dffs.shape[0])))
        roi_idx = np.argsort(np.abs(r_fast))[::-1][:keep]

    F = dffs[roi_idx].astype(np.float32)

    # 1) Lagged design for pulse channel
    X, lags = build_lagged_design(stim.astype(np.float32), fs, lag_sec=lag_sec, bin_sec=bin_sec)

    # 2) Contiguous CV folds
    folds = contiguous_kfold_indices(T=stim.shape[0], K=K)

    # 3) Ridge CV (shared X, per ROI)
    G_pred, rhos_cv, alpha_best, coefs = ridge_cv_predict_per_roi(X, F, alphas, folds, center_X=True)

    # 4) Paper-style p-values (chunk shuffle ROI vs SAME predictions)
    rng = np.random.default_rng(rng_seed)
    pvals, rho_cv = pvalues_chunk_shuffle(F, G_pred, fs, folds, B=B, chunk_sec=chunk_sec, rng=rng)

    # 5) BH-FDR at 0.01
    rej, pvals_bh, _, _ = multipletests(pvals, alpha=fdr_q, method="fdr_bh")
    sel_local = np.where(rej)[0]
    sel_idx = roi_idx[sel_local]  # map back to global ROI indices

    # Package filters (1 channel × L lags)
    L = X.shape[1]
    filters = coefs.reshape(-1, 1, L)  # (n_kept_rois, 1, L)

    out = {
        "sel_idx": sel_idx,               # global ROI indices passing BH 0.01
        "pvals": pvals,
        "pvals_bh": pvals_bh,
        "rhos_cv": rhos_cv,               # (n_kept, K) CV correlations
        "alpha_best": alpha_best,         # (n_kept,)
        "filters": filters,               # (n_kept, 1, L)
        "lag_samples": lags,              # (L,)
        "fs": fs,
        "roi_idx_kept": roi_idx,          # mapping from kept subset to global
    }
    return out