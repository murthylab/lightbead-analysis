#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SIGNAL: df/f
- extract df/f and zscored(df/f) from motion corrected 4D gcamp array
- applies a high pass filter to adjust for long-scale drift such as bleaching etc

4 apr 2022
@author: sama ahmed

todo:

"""

import numpy as np
from scipy.ndimage import gaussian_filter1d
import os
import logging
from typing import Tuple

# --------*--------*--------*--------
# primary
# --------*--------*--------*--------


def getdff(gcamp_path: str, outdir: str, fname: str, shape: Tuple[int, int, int, int], F0window: int, HPF_sigma: int):
    """ get df/f and zscored(df/f) from 4D motion corrected gcamp data
    inputs:
        gcamp_path: path/to/green_moco.mmap
        outdir: path/to/save/dff/and/zscore_dff/
        fname: filename.mmap
        shape: tuple, (x, y, z, t)
    """
    logging.info(f'loading gcamp from {gcamp_path}')
    brain = np.memmap(gcamp_path, dtype='float32', mode='r+', shape=shape)

    # initialize the df/f and z(df/f) memmaps (flush these later)
    dff_path = os.path.join(outdir, fname)
    brain_dff = np.memmap(dff_path, dtype='float32', mode='w+', shape=shape)

    z_dff_path = os.path.join(outdir, f'zscore_{fname}')
    z_brain_dff = np.memmap(z_dff_path, dtype='float32', mode='w+', shape=shape)

    # for each slice in brain -> filter, then dff and zdff
    logging.info('adjusting each pixel for bleaching, then df/f and z(df/f)')
    for ii in range(shape[2]):  # for each slice in z
        logging.info(f'processing slice # {ii}')
        thisSlice = brain[:, :, ii, :]

        hpf = _high_pass_filter(thisSlice[:, :, None, :],hpf_sigma=HPF_sigma)  # needs 4D input

        dff = _dff(hpf,F0_win=F0window)
        brain_dff[:, :, ii, :] = dff[:, :, 0, :]

        zscored_dff = zscore_4D(dff)
        z_brain_dff[:, :, ii, :] = zscored_dff[:, :, 0, :]

    brain_dff.flush()
    z_brain_dff.flush()

# --------*--------*--------*--------
# simple processing of 4D brain data
# --------*--------*--------*--------


def mean_per_volume(brain):
    """returns mean along time axis
    inputs:
        brain: 4D array (x, y, z, y)
    outputs:
        1D array (t, )
    """
    x = [brain[:, :, :, ii].mean() for ii in range(brain.shape[3])]
    return np.array(x)


def zscore_4D(brain):
    """
    input:
        brain: 4D array (x, y, z, t)
    """
    brain_mean = np.mean(brain, axis=3)
    brain_std = np.std(brain, axis=3)
    brain = (brain - brain_mean[:, :, :, None]) / brain_std[:, :, :, None]
    return brain


def smooth_signal(brain):
    return gaussian_filter1d(brain, sigma=2, axis=-1, truncate=1)

# --------*--------*--------*--------
# helpers
# --------*--------*--------*--------


def _high_pass_filter(brain, hpf_sigma):
    logging.info('high pass filtering via memmapping')
    smoothed = gaussian_filter1d(brain, sigma=hpf_sigma, axis=-1, truncate=1)
    logging.info(f'shape of high pass smoothing filter: {smoothed.shape}')
    corrected = brain - smoothed + np.mean(brain, axis=3)[:, :, :, None]
    return corrected


def _dff(brain, F0_win):
    """calculated df/f based on F0 baseline"""
    # find average signal in first 60 seconds (187 vols)
    F0 = np.mean(brain[:, :, :, :F0_win], axis=3)
    dff = (brain - F0[:, :, :, None]) / F0[:, :, :, None]
    return dff
