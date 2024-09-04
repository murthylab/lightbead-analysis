#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PSTH: event-triggered average
- align data to a timepoint and find mean signal in 2D (x, y)

26 apr 2022
@author: sama ahmed

todo:

"""

import numpy as np
import os
import logging
from typing import List, Tuple
from skimage.transform import downscale_local_mean
from scipy.ndimage import gaussian_filter

# --------*--------*--------*--------
# primary
# --------*--------*--------*--------


def time_triggered_signal_from_mmap(fpath: str, shape: Tuple[int, int, int, int], z: int, t: List[int], downfactor: Tuple[int, int] = (2, 2)):
    """
    inputs:
        fpath: filepath to memmap file containing 4D data
        shape: tuple (x, y, z, t)
        z: z-slice number
        t: timepoints, list of ints
        downfactor: tuple, downscale x and y
    output:
        returns a 2D array (x, y) with average over timepoints in t
    """
    logging.info(f'loading memmap file from {fpath}')
    m = np.memmap(fpath, dtype='float32', mode='r', shape=shape)

    # initialize
    sum_array = downscale_local_mean(np.zeros((shape[0], shape[1])), downfactor)

    logging.info('generate average x, y')
    for timepoint in t:

        img = downscale_local_mean(m[:, :, z, timepoint], downfactor)
        sum_array = gaussian_filter(img, sigma=0.5)

    logging.info(f'dividing summed array by {len(t)}')

    return sum_array / len(t)
