#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
VIZ: quickly visualize some data

6 apr 2022
@author: sama ahmed

todo:
[ ]
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

import logging
from typing import Tuple

import sys
sys.path.insert(1, '../../src')  # add src/brainviz package to path
from brainviz import io  # noqa: E402
from flydff import signal  # noqa: E402

# --------*--------*--------*--------
# primary
# --------*--------*--------*--------


def quality_check(outdir: str, flytrial: str, fixed_brain_path: str, greenpath: str, redpath: str, z_dff_path: str, shape: Tuple[int, int, int, int]):
    """check quality of fly functional imaging data
    inputs;
        outdir: path/to/output/directory
        flytrial: e.g. 'YYMMDD_1_103'
        fixed_brain_path: path/to/mean/tdtomato/low/res/volume.nii
        greenpath: path/to/green/channel/motioncorrected/signal.mmap
        redpath: path/to/red/channel/motioncorrected/signal.mmap
        z_dff_path: path/to/zscored_dff.map
    returns:
        max_proj: max intensity projection of tdtomato volume
        mean_green: green fluor signal averaged by volume
        mean_red: red fluor signal averaged by volume
        mean_zdff: z-scored df/f averaged by volume
    """

    # load fixed mean tdtomato, take max projection
    fixed = io.load(fixed_brain_path)
    max_proj = np.max(fixed, -1)

    # load 4D (x,y,z,t) data from memmaps
    logging.info('loading data..')
    green = np.memmap(greenpath, dtype='float32', mode='r', shape=shape)
    red = np.memmap(redpath, dtype='float32', mode='r', shape=shape)
    zdff = np.memmap(z_dff_path, dtype='float32', mode='r', shape=shape)

    # initialize
    mean_green = []
    mean_red = []
    mean_zdff = []

    # take average over time
    logging.info('calculating volume-averaged signal..')
    for ii in range(shape[-1]):
        logging.info(f"volume # {ii}")
        mean_green.append(np.mean(green[:, :, :, ii]))
        mean_red.append(np.mean(red[:, :, :, ii]))
        mean_zdff.append(np.mean(zdff[:, :, :, ii]))

    # plot
    fig = plt.figure(figsize=(8, 8))

    _plot_max_tdtomat(fig, max_proj, flytrial)
    _plot_green_red_correlations(fig, mean_green, mean_red)
    _plot_mean_signal_over_time(fig, mean_green, mean_red)
    _plot_zscored_dff(fig, mean_zdff)

    # save
    figname = f'quality_check_{flytrial}.png'
    logging.info(f'saving: {figname}')
    plt.savefig(os.path.join(outdir, figname),
                transparent=False,
                bbox_inches='tight')

    return (max_proj, mean_green, mean_red, mean_zdff)

# --------*--------*--------*--------
# helpers
# --------*--------*--------*--------


def _plot_max_tdtomat(fig, max_proj, flytrial):
    rect = [0.02, 0.5, 0.5, 0.4]
    ax = fig.add_axes(rect)
    ax.matshow(max_proj.T, interpolation='none', cmap='Reds')

    ax.set_xticks([])
    ax.set_yticks([])
    plt.title(f'fly: {flytrial}')
    plt.xlabel('low res mean tdtomato - max projection')


def _plot_green_red_correlations(fig, mean_green, mean_red):
    rect = [0.62, 0.5, 0.3, 0.4]
    ax = fig.add_axes(rect)

    ax.plot(stats.zscore(mean_green), stats.zscore(mean_red), 'ko', alpha=0.1)

    ax.xaxis.label.set_color('g')
    ax.yaxis.label.set_color('m')
    plt.xlim([-4, 4])
    plt.ylim([-4, 4])
    plt.xticks([-4, 0, 4])
    plt.yticks([-4, 0, 4])
    plt.xlabel('zscore(green fluor)')
    plt.ylabel('zscore(red fluor)')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def _plot_mean_signal_over_time(fig, mean_green, mean_red):
    rect = [0.02, 0.2, 0.96, 0.16]
    ax = fig.add_axes(rect)

    ax.plot(stats.zscore(mean_green), color='g')
    ax.plot(stats.zscore(mean_red), color='m')

    plt.ylim([-4, 4])
    plt.yticks([-4, 0, 4])
    plt.xticks([])
    plt.ylabel('mean zscore(fluor)')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)


def _plot_zscored_dff(fig, mean_zdff):
    rect = [0.02, 0.02, 0.96, 0.16]
    ax = fig.add_axes(rect)

    ax.plot(mean_zdff, color='k')

    plt.ylim([-0.2, 0.2])
    plt.yticks([-0.2, 0, 0.2])
    plt.ylabel('zscored(dF/F)')
    plt.xlabel('volume')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
