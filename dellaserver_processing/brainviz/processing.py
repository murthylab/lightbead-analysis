#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PROCESSING: functions for data pre/processing
16 nov 2021
@author: sama ahmed
"""

import numpy as np
from sklearn.preprocessing import quantile_transform
import logging

LOG = logging.getLogger(__name__)


def min_max_normalize(arr):
    """normalize values to range [0, 1]"""
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))


def quant_normalize(arr, n_quantiles=10000):
    """quantile normalize arr of shape: [n_samples, n_features]"""

    return quantile_transform(arr,
                              n_quantiles=n_quantiles,
                              random_state=0,
                              copy=True)


def reshape_xyz_to_z_by_xy(vol, z: int):
    """reshape 3D volume (xyz) to 2D (z, x*y)"""
    return vol.transpose(2, 0, 1).reshape(z, -1)
