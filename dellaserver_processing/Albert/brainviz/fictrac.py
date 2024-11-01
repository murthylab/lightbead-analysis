#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FICTRAC: load fictrac data from FlyVR1
16 feb 2022
@author: sama ahmed


todo
[ ] convert speed to mm/s https://en.ans.wiki/287/how-to-convert-angular-speed-rad-slash-s-to-linear-velocity-m-slash-s/
"""

# import matplotlib.pyplot as plt
import numpy as np
# import os

import h5py
from scipy.interpolate import interp1d
from scipy.signal import find_peaks, savgol_filter

import logging

LOG = logging.getLogger(__name__)


class Fictrac:
    """fictrac class
    takes fictrac.h5 file and creates object containing relevant behavior features (fly speed on the ball, heading, etc) along with vectors that synchronize the audio/opto data and 2photon volumes to "fictrac time"
    """

    def __init__(self, fictrac_fpath: str):
        self.fpath = fictrac_fpath

        # hardcoded params for Sama's experiments
        self.Fs = 10000  # audio/opto sampling rate
        self.fps = 50  # video frame rate
        self.ball_rad = 9.0 / 2.0  # ball radius

        self.load_data()
        self._run_initializer()

    def load_data(self):
        """loads fictrac behavioral data and synchronization info"""

        self.f = h5py.File(self.fpath, "r")

        # fictrac outputs, smoothed with Savitzky-Golay filter
        self.dRotLabX = self._smooth_fic(self.f['fictrac/output'][:, 5])
        self.dRotLabY = self._smooth_fic(self.f['fictrac/output'][:, 6])
        self.dRotLabZ = self._smooth_fic(self.f['fictrac/output'][:, 7])
        self.posX = self._smooth_fic(self.f['fictrac/output'][:, 14])
        self.posY = self._smooth_fic(self.f['fictrac/output'][:, 15])
        self.heading = self._smooth_fic(self.f['fictrac/output'][:, 16])
        self.runningDir = self._smooth_fic(self.f['fictrac/output'][:, 17])
        self.speed = self._smooth_fic(self.f['fictrac/output'][:, 18])  # rad/fr

        # video timestamps
        self.timestamps = self.f['fictrac/output'][:, 21]

        # zero the timestamps
        self.ts = self.timestamps - self.timestamps[0]

        # fictrac frames cooresponding to self.ts
        self.fic_x = np.arange(len(self.ts))

        self.opto = self.f['input']['samples'][:, 1]
        self.galvo = self.f['input']['samples'][:, 2]
        self.mic = self.f['input']['samples'][:, 4]

    def get_audio_sample_at_vid_frame(self):
        """synchronize audio and video frames"""

        vid = self.f['fictrac']['daq_synchronization_info'][:, 0]
        aud = self.f['fictrac']['daq_synchronization_info'][:, 1]

        # get audio sample indexes at video frame index (self.ts)
        f = interp1d(vid, aud, bounds_error=False)
        self.asavf = np.round(f(self.fic_x)).astype(int)

    def get_gcamp_volume_number_at_vid_frame(self):
        """synchronize 2Photon imaging volumes with video frames"""
        # todo: [ ] make sure hardcoded height and distance params give consistent results across flies/trials

        # the galvo readings indicate when a new volume is collected. Take a look by plt.plot(galvo) and ensure that the flyback frame has a higher peak than the rest of the galvo readings
        peaks, _ = find_peaks(self.galvo,
                              height=1.1,
                              distance=self.Fs * 0.1)

        # number of expected volumes
        volumes = np.arange(len(peaks))

        # find corresponding imaging volume at each fictrac frame
        f = interp1d(peaks, volumes, bounds_error=False)
        self.vol_vf = np.round(f(self.asavf)).astype(int)
        self.vol_vf[self.vol_vf < 0] = 0  # remove spurious predictions

    def convert_to_vidframe_coordinates(self):
        ''' downsample data from audio sampling rate to video frame rates'''
        self.mic_vf = self.mic[self.asavf]  # microphones at "vf": vid frames
        self.opto_vf = self.opto[self.asavf]  # opto at "vf": vid frames

    # -- helpers --
    def _run_initializer(self):
        self.get_audio_sample_at_vid_frame()
        self.get_gcamp_volume_number_at_vid_frame()
        self.convert_to_vidframe_coordinates()

    def _smooth_fic(self, x):
        """savgol_filter(x, window_length, polyorder)"""
        return savgol_filter(np.asarray(x), 25, 3)
