#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
VOLS: prepare volumes
- extract individual volumes from tiffs
- generate an average tdtomato volume, used later for motion correction and alignment to single fly atlas

@author: sama ahmed, albert lin

todo:
[ ] rewrite average_volume() to work with any directory
[ ] refactor directory checks out of vols.py
[ ] refactor so get_from_tiffs() works with any directory and choose channel(s)
[ ] add option to concat_vol to save via memmap
"""

import os
import errno
import numpy as np
import glob
import logging

import sys
sys.path.insert(1, '../../src')  # add src/brainviz package to path
from brainviz import io  # noqa: E402

# --------*--------*--------*--------
# primary
# --------*--------*--------*--------


def get_from_tiffs(tiffdir, outdir):
    """save green volumes from tiffs"""

    _check_dirs(tiffdir, outdir)
    green_dir = _get_green_dirs(outdir)

    list_of_tiffs = _list_tiffs(tiffdir)
    logging.info(f'list of tiffs: {list_of_tiffs}')
    _extract_volumes_from_tiffs(list_of_tiffs, green_dir)


def concat_volumes(indir, outpath, xdim=226, ydim=512, zdim=27):
    """make a 4D (x,y,z,t) dataset from a directory of 3D (x,y,z) volumes"""

    # initialize
    volnames = io.get_file_paths(indir)
    # concat_vol = np.empty(shape=(xdim, ydim, zdim, len(volnames)))
    concat_vol = np.memmap(outpath, dtype='float32', mode='w+', shape=(xdim, ydim, zdim, len(volnames)))

    # load volume and append
    for ii, vol in enumerate(volnames):
        logging.info(f'concatentating: {vol}')
        concat_vol[:, :, :, ii] = io.load(vol)

    logging.info(f'saving to mempath {outpath}')

    concat_vol.flush()

    
    # fp = np.memmap(outpath, dtype='float32', mode='w+', shape=(xdim, ydim, zdim, len(volnames)))
    # fp[:] = concat_vol[:]
    # fp.flush()


def average_volume(outdir, ch: int, nvols: int, fname=None):
    """make an average green (ch=0) volume from volumes in dirs

    returns: str, path to mean brain
    """

    _check_channel_options(ch)

    green_dir = _get_green_dirs(outdir)
    if ch == 0:  # ch=0 is green
        vol_dir = green_dir

    PIXELS_X, PIXELS_Y, NUMZFRAMES, FLYBACKFRAME, COLOR = _load_hardcoded_params()

    fname = fname if fname is not None else f'mean-{COLOR[ch]}.nii'
    output_path = os.path.join(outdir, fname)
    logging.info(f"path to folder containing volumes: {vol_dir}")
    logging.info(f"path to the final mean brain: {output_path}")

    # add each volume to initialized container
    sum_array = np.zeros((PIXELS_X, PIXELS_Y, FLYBACKFRAME))  # initialize
    logging.info(f"empty container has shape: {sum_array.shape}")

    files = io.get_file_paths(vol_dir)

    for counter, f in enumerate(files):

        # load brain volume
        logging.info(f"{counter} - loading: {f}")

        vol = io.load(f)

        logging.info("adding to sum_array")
        sum_array = np.add(sum_array, vol)

        if counter == nvols:
            logging.info(f"# of brains: {counter}")
            break

    logging.info("normalizing summed volume")
    mean_brain = sum_array / counter

    logging.info("saving averaged volume")
    io.save(output_path, mean_brain)
    logging.info("saved")

    return output_path

# --------*--------*--------*--------
# helper functions
# --------*--------*--------*--------


def _load_hardcoded_params():
    xdim = 226
    ydim = 512
    zdim = 28  # number of slices in Z direction

    # how many slice left after removing last frame (ie 'flyback frame')
    FLYBACKFRAME = zdim - 1
    COLOR = ['green']

    return xdim, ydim, zdim, FLYBACKFRAME, COLOR


def _initialize_fly_volume(arr, FLYBACKFRAME):
    return np.zeros((arr.shape[0],
                     arr.shape[1],
                     FLYBACKFRAME))


def _check_channel_options(ch):
    ch_options = [0]
    if ch not in ch_options:
        raise ValueError("Invalid ch. Expected 0 only")


def _check_dirs(tiffdir, outdir):
    """check input and output directories"""
    if not os.path.exists(tiffdir):
        logging.info(f'tiffdir: {tiffdir}')
        raise FileNotFoundError

    if not os.path.exists(outdir):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), outdir)


def _get_green_dirs(outdir):
    """prep directories for green volumes"""
    green_dir = os.path.join(outdir, 'green_ch')
    if not os.path.exists(green_dir):
        os.mkdir(green_dir)

    logging.info(f'gcamp volumes, pre-moco: {green_dir}')

    return green_dir


def _list_tiffs(tiffdir):
    """returns list of tiff files in a directory"""
    return sorted(glob.glob(tiffdir + '/*.tif'))


def _extract_volumes_from_tiffs(list_of_tiffs, green_dir: str):
    """get single volumes from tiffs and store in correct green dir"""

    _, _, NUMZFRAMES, FLYBACKFRAME, COLOR = _load_hardcoded_params()

    logging.info("extracting volumes from tiff files")
    THISPAGE = 0  # page to process in queue, range [0, last_page_of_experiment]
    STARTPAGE = 0  # start at 1st channel (gcamp)
    for TIFFIDX, tiff in enumerate(list_of_tiffs):
        tiff_array = io.load(tiff)

        if TIFFIDX == 0:  # initialize the first volumes
            green_volume = _initialize_fly_volume(tiff_array, FLYBACKFRAME)
            logging.info("initializing the first volume")

        NUMPAGES = tiff_array.shape[2]

        logging.info(f"tiff loaded ---> x: {tiff_array.shape[0]}, y: {tiff_array.shape[1]}, pages: {NUMPAGES}")

        # go thru each page in tiff
        for iPage in np.arange(STARTPAGE, NUMPAGES, 1):
            logging.info(f"decoding page # {THISPAGE}")
            ch, frame, vol = io.decode_tiff_page_greenonly(THISPAGE, NUMZFRAMES)
            logging.info(f"decoded info ---> ch: {ch}, frame: {frame}, vol: {vol}")

            if ch == 0:  # ch=0 is green
                if frame != FLYBACKFRAME:
                    # append frame to current volume
                    green_volume[:, :, frame] = tiff_array[:, :, iPage]
                else:
                    # save current volume
                    logging.info(f"saving volume {vol} to file")
                    fpath = f"{COLOR[ch]}_volume_{vol}.nii"
                    io.save(os.path.join(green_dir, fpath), green_volume)

                    # re-initialize the volume
                    green_volume = _initialize_fly_volume(tiff_array, FLYBACKFRAME)

            # next page to process in queue
            THISPAGE += 1
