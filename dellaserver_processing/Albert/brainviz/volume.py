#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
VOLUME: visualize 3D arrays
11 nov 2021
@author: sama ahmed
"""

import matplotlib.pyplot as plt
import numpy as np
import os

import ants

from skimage.measure import block_reduce
from skimage.restoration import denoise_nl_means, estimate_sigma
import skimage.filters
from scipy.ndimage import gaussian_filter

from brainviz.videowriter import VideoWriter
from brainviz import processing as proc
import brainviz.plotting as plotting

import logging

LOG = logging.getLogger(__name__)


class Volume:
    """brain volume: 3D array
    TODO:
        [ ] have option to generate (or load) mask (otsu thershold)
        [ ] add non-local means option
        [ ] check that max_projection() works
    LATER:
        [ ] patch2self option
    """

    def __init__(self, volume, figdir=None):
        self.volume = volume
        self.figdir = figdir if figdir is not None else os.getcwd()
        self._check_volume_is_3D()
        self._pick_slices_for_quick_plots()

    # -- normalization --

    def normalize(self, volume=None):
        """min-max normalize values in each z-slice to range [0,1]"""
        volume = volume if volume is not None else self.volume
        self.mmn = np.full_like(self.volume, 0)
        for ii in range(self.z):
            self.mmn[:, :, ii] = proc.min_max_normalize(self.volume[:, :, ii])
        return Volume(self.mmn)

    def quantile(self, volume=None, n_quantiles=10000):
        """quantile normalize values in volume"""
        volume = volume if volume is not None else self.volume

        self.qn = np.full_like(volume, 0)

        # reshape -> z-by-(x*y) then transpose before Quantile Normalize
        X = proc.reshape_xyz_to_z_by_xy(volume, self.z)
        qn = proc.quant_normalize(X.T, n_quantiles=n_quantiles)

        # reshape back to original
        self.qn = np.reshape(qn, (self.x, self.y, self.z))
        return Volume(self.qn)

    # -- 3D -> 2D --

    def max_projection(self, volume=None):
        """max intensity projection of 3D volume along z-axis"""
        volume = volume if volume is not None else self.volume
        self.max_proj = np.max(volume, -1)

    # -- masks --

    def apply_otsu(self, volume=None, scale=0.5):
        """apply otsu's thresholding to each slice in the volume"""
        volume = volume if volume is not None else self.volume
        self.otsu = np.full_like(volume, 0)

        for iSlice in np.arange(volume.shape[-1]):
            img = volume[:, :, iSlice]
            binary_mask = self._otsu_threshold(img, scale=scale)

            # apply the binary mask to the slice
            selection = np.zeros_like(img)
            selection[binary_mask] = img[binary_mask]
            self.otsu[:, :, iSlice] = selection

        return Volume(self.otsu)

    def apply_otsumask(self, volume=None, scale=0.5):
        """apply otsu's thresholding to each slice and returns 3D mask"""
        volume = volume if volume is not None else self.volume
        self.otsumask = np.full_like(volume, 0)

        for iSlice in np.arange(volume.shape[-1]):
            img = volume[:, :, iSlice]
            binary_mask = self._otsu_threshold(img, scale=scale)
            self.otsumask[:, :, iSlice] = binary_mask

        return self.otsumask

    # -- denoising --

    def nonlocal_means(self, volume=None, fast_mode=True):
        """apply non-local means denoising to volume"""

        volume = volume if volume is not None else self.volume
        self.nlm = np.full_like(volume, 0)

        # process each slice in the volume on its own
        # (assumes volume.shape is xyz)
        for iSlice in np.arange(volume.shape[-1]):
            sigma_est = np.mean(estimate_sigma(volume[:, :, iSlice]))
            LOG.debug(f'estimated noise standard deviation = {sigma_est}')

            # ensure slice is contiguous (denoise_nl_means depends on this...)
            x = volume[:, :, iSlice].copy(order='C')
            self.nlm[:, :, iSlice] = denoise_nl_means(x, h=0.9 * sigma_est,
                                                      sigma=sigma_est, fast_mode=fast_mode)
        return Volume(self.nlm)

    def adaptive_nonlocal_means(self, volume=None):
        """apply AntsPy's adaptive non-local means denoiser"""
        volume = volume if volume is not None else self.volume

        vol = ants.from_numpy(volume)
        vol_denoised = ants.denoise_image(vol)
        self.anlm = vol_denoised.numpy()

        return Volume(self.anlm)

    def gaussian(self, volume=None, sigma=1):
        """apply gaussian filter to volume"""
        volume = volume if volume is not None else self.volume

        self.gaus = gaussian_filter(volume, sigma)
        return Volume(self.gaus)

    # -- transforms --

    def mirror(self, volume=None):
        """flip volume in X"""
        volume = volume if volume is not None else self.volume
        self.mirror = np.flip(volume, 0)
        return Volume(self.mirror)

    def flipZ(self, volume=None):
        """flip volume in Z"""
        volume = volume if volume is not None else self.volume
        self.vol_z = np.flip(volume, 2)
        return Volume(self.vol_z)

    def flipY(self, volume=None):
        """flip volume in Y"""
        volume = volume if volume is not None else self.volume
        self.vol_y = np.flip(volume, 2)
        return Volume(self.vol_y)

    def downsample(self, block_size=(2, 2, 2), func=np.mean):
        """downsample current volume and keep, and also return as Volume"""
        self.ds = block_reduce(self.volume,
                               block_size=block_size,
                               func=func)
        return Volume(self.ds)

    # -- visualization --

    def make_video(self, fpath: str, volume=None, fps: int = 30):
        """save volume as a video traversing the z-direction"""
        volume = volume if volume is not None else self.volume

        writer = VideoWriter(fpath, volume, fps)
        writer.make_video()

    def plot_some_frames(self,
                         volume=None,
                         frames=None,
                         save=False,
                         figname='some_frames.png'):
        """plot grid of select frames from the volume"""
        fig, ax = plt.subplots(4, 4, figsize=(8, 6), dpi=150)

        # choose volume, frames
        volume = volume if volume is not None else self.volume
        frames = frames if frames is not None else self.zslices

        LOG.info(f"using these frames: {frames}")
        for idx, iAx in enumerate(ax.reshape(-1)):
            try:
                plotting.plot_slice(volume[:, :, frames[idx]],
                                    ax=iAx)
            except IndexError:
                LOG.debug('IndexError: not enough frames to plot')
                iAx.axis('off')

        if save:
            plotting.save_fig(self.figdir, figname)

    def zslice_pixel_vals(self,
                          volume=None,
                          save=False,
                          figname='pixel_distribution.png'):
        """boxplot of pixel distributions for selected z-slices in volume"""

        volume = volume if volume is not None else self.volume

        new_vol = proc.reshape_xyz_to_z_by_xy(volume, volume.shape[-1])
        LOG.info(f"new shape for volume: {new_vol.shape}")

        # boxplots
        plt.boxplot(new_vol[self.zslices, :].T,
                    widths=0.1,
                    showfliers=False)

        plt.xticks(range(1, self.zslices.shape[0] + 1),
                   self.zslices,
                   rotation=45)

        plt.xlabel('z-slice', fontsize=8)
        plt.ylabel('pixel value', fontsize=8)

        plotting.PrettyAxes()

        if save:
            plotting.save_fig(self.figdir, figname)

    # -- helpers --

    def _check_volume_is_3D(self):
        assert self.volume.ndim == 3, "volume must be 3D: [xdim, ydim, zdim]"
        self.x = self.volume.shape[0]
        self.y = self.volume.shape[1]
        self.z = self.volume.shape[2]

    def _pick_slices_for_quick_plots(self):
        """pick 16 equally spaced z-slices in volume, if possible"""
        num = 16 if self.z >= 16 else self.z
        self.zslices = np.round(np.linspace(0, self.z - 1, num=num)).astype(int)

    def _otsu_threshold(self, img, scale=1, sigma=3):
        img2 = gaussian_filter(img, sigma=sigma)  # denoise
        t = skimage.filters.threshold_otsu(img2)  # find threshold
        LOG.debug(f"automatic otsu's threshold: {scale * t}")
        binary_mask = img > scale * t
        return binary_mask
