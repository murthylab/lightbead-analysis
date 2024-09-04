#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ROI: functions for processing voxels into ROIs
18 mar 2022
@author: sama ahmed


todo
"""

# import matplotlib.pyplot as plt
# import numpy as np
# import os

import numpy as np

from sklearn.feature_extraction.image import grid_to_graph
from sklearn.cluster import AgglomerativeClustering

import logging

LOG = logging.getLogger(__name__)


def create_2d_clusters(brain_slice, n_clusters: int, mempath: str):
    """
    inputs:
        brain_slice: 3D array (x, y, t)
        n_clusters: # of clusters to generate (e.g. 2000)
        mempath: path to cache the output of the computation of the tree

    usage:
        from brainviz import roi
        n_clusters = 2000
        cluster_model = roi.create_2d_clusters(slice, n_clusters, 'dat/cluster_mem')
        labels = []
        labels.append(cluster_model.labels_)

    this is based on Luke Brezovec's "create_clusters" function
    """

    xdim = brain_slice.shape[0]
    ydim = brain_slice.shape[1]
    tdim = brain_slice.shape[2]

    # enforce that clustered voxels must be neighbors
    connectivity = grid_to_graph(xdim, ydim)

    cluster_model = AgglomerativeClustering(n_clusters=n_clusters, memory=mempath, linkage='ward', connectivity=connectivity)

    super_to_cluster = brain_slice.reshape(-1, tdim)
    cluster_model.fit(super_to_cluster)

    return cluster_model


def create_3d_clusters(brain_volume, n_clusters: int, mempath: str):
    """
    inputs:
        brain_volume: 4D array (x, y, z, t)
        n_clusters: # of clusters to generate (e.g. 2000)
        mempath: path to cache the output of the computation of the tree
    """

    xdim = brain_volume.shape[0]
    ydim = brain_volume.shape[1]
    zdim = brain_volume.shape[2]
    tdim = brain_volume.shape[3]

    # enforce that clustered voxels must be neighbors
    connectivity = grid_to_graph(xdim, ydim, zdim)

    cluster_model = AgglomerativeClustering(n_clusters=n_clusters, memory=mempath, linkage='ward', connectivity=connectivity)

    super_to_cluster = brain_volume.reshape(-1, tdim)
    cluster_model.fit(super_to_cluster)

    return cluster_model


def get_supervoxel_mean(brain_slice, cluster_labels, n_clusters: int):
    """
    inputs:
        brain: 3D array (x, y, t)
        cluster_labels: (x*y) from |create_2d_clusters| (see example code above)
        n_clusters: e.g. 2000
    outputs:
        signals: mean df/f of supervoxel (n_clusters, t)
        cluster_idx: list (len = n_clusters) w/brain_slice idx mapped to a cluster
    """
    tdim = brain_slice.shape[2]
    neural_data = brain_slice.reshape(-1, tdim)

    signals = []
    cluster_idx = []

    for nn in range(n_clusters):
        idx = np.where(cluster_labels == nn)[0]
        mean_signal = np.nanmean(neural_data[idx, :], axis=0)

        signals.append(mean_signal)
        cluster_idx.append(idx)

    return np.asarray(signals), cluster_idx


def get_supervoxel_mean_2D(brain_slice, cluster_labels, n_clusters: int):
    """
    inputs:
        brain: 2D array (x, y)
        cluster_labels: (x*y) from |create_2d_clusters| (see example code above)
        n_clusters: e.g. 2000
    outputs:
        signals: mean df/f of supervoxel (n_clusters)
        cluster_idx: list (len = n_clusters) w/brain_slice idx mapped to a cluster
    """
    x_by_y = brain_slice.shape[0] * brain_slice.shape[1]
    neural_data = brain_slice.reshape(x_by_y)  # make into vector

    signals = []
    cluster_idx = []

    for nn in range(n_clusters):
        idx = np.where(cluster_labels == nn)[0]
        mean_signal = np.nanmean(neural_data[idx])

        signals.append(mean_signal)
        cluster_idx.append(idx)

    return np.asarray(signals), cluster_idx
