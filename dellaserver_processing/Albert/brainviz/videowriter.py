#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
VIDEOWRITER: write 3d array (e.g. volume) to black and white mp4 video
16 nov 2021
@author: sama ahmed

Usage: ::
    > writer = VideoWriter('this_dir/video_name.mp4', volume, fps)
    > writer.make_video()

notes:
1) needs ffmpeg and cv2 installed
2) volume is normalized and converted to int8 [0, 255] before making into video
3) also see: github.com/murthylab/sleap/blob/develop/sleap/io/videowriter.py
"""

import numpy as np
import cv2
from brainviz.processing import min_max_normalize
import logging

LOG = logging.getLogger(__name__)


class VideoWriter():

    def __init__(self, filepath: str, volume, fps=30):

        self.filepath = filepath
        self.volume = volume
        self.fps = fps

        self._check_volume_is_3D()

        # FourCC is a 4-byte code used to specify the video codec.
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")

        # initialize the video writer
        self._writer = cv2.VideoWriter(self.filepath,
                                       fourcc,
                                       self.fps,
                                       (self.width, self.height),
                                       False)  # False -> b/w video

    def add_frame(self, img):
        self._writer.write(img)

    def close(self):
        self._writer.release()

    def make_video(self):
        """iterates over frames and outputs mp4"""
        for ii in range(self.frames):
            img = self._convert_frame_to_int8(self.volume[:, :, ii])
            self.add_frame(img)

        self.close()
        LOG.info(f'done making video, check {self.filepath}')

    # -- helpers--
    def _convert_frame_to_int8(self, x):
        """min-max normalize to range [0, 255] and convert to int8"""
        x = 255 * min_max_normalize(x)
        img = x.T.astype(np.uint8)
        return img

    def _check_volume_is_3D(self):
        assert self.volume.ndim == 3, "volume must be 3D: [xdim, ydim, nframes]"
        self.width = self.volume.shape[0]
        self.height = self.volume.shape[1]
        self.frames = self.volume.shape[2]
