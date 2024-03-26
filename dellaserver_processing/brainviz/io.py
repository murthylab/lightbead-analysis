#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
IO: input/output utils for loading, saving image arrays
15 nov 2021
@author: sama ahmed

todo:
[ ] find better home for decode_tiff_page
"""

import os
import glob
import ants
import logging

LOG = logging.getLogger(__name__)


def load(fpath: str):
    """loads volume from filepath and returns N-Dim numpy array"""
    LOG.info(f"loading: {fpath}")
    imgarray = ants.image_read(fpath)
    volume = imgarray.numpy()
    LOG.debug(f"volume shape: {volume.shape}")
    return volume


def save(fpath: str, volume):
    "save volume to fpath"
    ants.image_write(ants.from_numpy(volume), fpath)
    LOG.info(f'saved volume to: {fpath}')


def get_file_names(folder: str):
    """returns file names from dir, sorted alphanumerically
    note: also finds hidden files
    """
    files = sorted(os.listdir(folder))
    return sorted(files, key=len)


def get_file_paths(folder: str):
    """returns file paths (folder/*.*) from dir, sorted alphanumerically
    note: ignore hidden files
    """
    filepaths = sorted(glob.glob(os.path.join(folder, '*')))
    return sorted(filepaths, key=len)


def decode_tiff_page(page_number, num_z: int) -> int:
    """given a tiff stack's page number, decode the channel (assumes 2 channels), frame # in z direction, and volume number"""

    LOG.debug("decoding tiff page, expecting 2 channels in tiff file")
    LOG.debug(f"page: {page_number}, z slices in volume; {num_z}")

    ch = _decode_ch(page_number)
    z_frame = _decode_frame(page_number, num_z)
    vol = _decode_vol(page_number, num_z)

    return ch, z_frame, vol

def decode_tiff_page_greenonly(page_number, num_z: int) -> int:
    """given a tiff stack's page number, decode the channel (assumes 1 channel), frame # in z direction, and volume number"""

    LOG.debug("decoding tiff page, expecting 2 channels in tiff file")
    LOG.debug(f"page: {page_number}, z slices in volume; {num_z}")

    ch = 0
    z_frame = _decode_frame_greenonly(page_number, num_z)
    vol = _decode_vol_greenonly(page_number, num_z)

    return ch, z_frame, vol

# -- helpers --
def _decode_ch(page_number: int) -> int:
    return 0 if page_number % 2 == 0 else 1


def _decode_frame(page_number: int, num_z: int) -> int:
    return (page_number // 2) % num_z

def _decode_frame_greenonly(page_number: int, num_z: int) -> int:
    return (page_number) % num_z


def _decode_vol(page_number: int, num_z: int) -> int:
    return page_number // (num_z * 2)

def _decode_vol_greenonly(page_number: int, num_z: int) -> int:
    return page_number // (num_z)
