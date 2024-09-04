#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LOCALATLAS: code for generating a high-resolution single-fly atlas

11 apr 2022
@author: sama ahmed

todo:

"""

import numpy as np
import logging
import glob
from typing import Tuple

import os
import sys
sys.path.insert(1, '../../src')  # add src/brainviz package to path
from brainviz import io  # noqa: E402

# todo - check memmap shape is (vol, x, y, z)

# --------*--------*--------*--------
# primary
# --------*--------*--------*--------


def get_from_tiffs(tiffdir, vol_path, ch, shape: Tuple[int, int, int, int]):
    list_of_tiffs = _list_tiffs(tiffdir)
    logging.info(f'list of tiffs: {list_of_tiffs}')
    _extract_volumes_from_tiffs(list_of_tiffs, vol_path, ch, shape=shape)


def average_volume(vol_path, output_path, shape: Tuple[int, int, int, int]):
    """ """
    # initialize empty container
    sum_array = np.zeros((shape[0], shape[1], shape[2]))  # initialize
    logging.info(f"empty container has shape: {sum_array.shape}")

    files = glob.glob(vol_path + "/*.nii")
    
    #debug log files, log path
    # logging.info(f"{vol_path}")
    # logging.info(f"{files}")

    counter = 0
    for f in files:
        counter += 1

        logging.info(f"{vol_path}")
        # load brain volume
        logging.info(f"{counter} - loading: {f}")

        vol = io.load(f)
        sum_array = np.add(sum_array, vol)

    logging.info(f"# of volumes: {counter}")
    mean_brain = sum_array / (counter)

    logging.info("saving averaged volume")
    io.save(output_path, mean_brain)
    logging.info("saved")

# --------*--------*--------*--------
# helper functions
# --------*--------*--------*--------


def _list_tiffs(tiffdir):
    """returns list of tiff files in a directory"""
    return sorted(glob.glob(tiffdir + '/*.tif'))


def _initialize_fly_volume(arr, FLYBACKFRAME):
    return np.zeros((arr.shape[0],
                     arr.shape[1],
                     FLYBACKFRAME))


def _extract_volumes_from_tiffs(list_of_tiffs, vol_path: str, ch: int, shape: Tuple[int, int, int, int]):
    """get single volumes from tiffs and store in vol_path folder"""

    if not os.path.exists(vol_path):
        os.mkdir(vol_path)

    FLYBACKFRAME = shape[2]
    NUMZFRAMES = FLYBACKFRAME + 1

    logging.info("extracting volumes from tiff files")
    THISPAGE = ch  # page to process in queue, range [ch, last_page_of_experiment]
    STARTPAGE = ch  # start at chosen channel (gcamp=0, tdtom=1)

    for TIFFIDX, tiff in enumerate(list_of_tiffs):
        tiff_array = io.load(tiff)

        if TIFFIDX == 0:  # initialize the first volumes
            volume = _initialize_fly_volume(tiff_array, FLYBACKFRAME)
            logging.info("initializing the first volume")

        NUMPAGES = tiff_array.shape[2]

        logging.info(f"tiff loaded ---> x: {tiff_array.shape[0]}, y: {tiff_array.shape[1]}, pages: {NUMPAGES}")

        # go thru pages in tiff
        for iPage in np.arange(STARTPAGE, NUMPAGES, 2):  # every 2nd page in the tiff
            logging.info(f"decoding page # {iPage} in tiff --> {THISPAGE}")
            ch, frame, vol = io.decode_tiff_page(THISPAGE, NUMZFRAMES)
            logging.info(f"decoded info ---> ch: {ch}, frame: {frame}, vol: {vol}")

            if frame != FLYBACKFRAME:
                # append frame to current volume
                volume[:, :, frame] = tiff_array[:, :, iPage]
            else:
                # save current volume
                logging.info(f"saving volume {vol} to vol_path")
                io.save(os.path.join(vol_path, f"volume_{vol}.nii"), volume)

                # re-initialize the volume
                volume = _initialize_fly_volume(tiff_array, FLYBACKFRAME)

            # next page to process in queue
            THISPAGE += 2  # skip a page (b/c using only one channel)
