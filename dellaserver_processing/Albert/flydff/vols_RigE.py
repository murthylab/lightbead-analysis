#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
VOLS: prepare volumes
- extract individual volumes from tiffs
- generate an average tdtomato volume, used later for motion correction and alignment to single fly atlas

24 mar 2021
@author: sama ahmed, Albert Lin

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


def get_from_tiffs(tiffdir, outdir, zdim, flybacknum, firstchannel):
    """save green and red volumes from tiffs"""

    _check_dirs(tiffdir, outdir)
    green_dir, red_dir = _get_green_red_dirs(outdir)

    list_of_tiffs = _list_tiffs(tiffdir)
    logging.info(f'list of tiffs: {list_of_tiffs}')
    if firstchannel==0:
        _extract_volumes_from_tiffs(list_of_tiffs, zdim, flybacknum, green_dir, red_dir)
    elif firstchannel==1:
        _extract_volumes_from_tiffs_RG(list_of_tiffs, zdim, flybacknum, green_dir, red_dir)

def get_from_tiffs_proj(tiffdir, outdir, zdim, flybacknum, firstchannel):
    """save green and red volumes from tiffs"""

    _check_dirs(tiffdir, outdir)
    green_dir, red_dir = _get_green_red_dirs_proj(outdir)

    list_of_tiffs = _list_tiffs(tiffdir)
    logging.info(f'list of tiffs: {list_of_tiffs}')
    if firstchannel==0:
        _extract_volumes_from_tiffs(list_of_tiffs, zdim, flybacknum, green_dir, red_dir)
    elif firstchannel==1:
        _extract_volumes_from_tiffs_RG(list_of_tiffs, zdim, flybacknum, green_dir, red_dir)


def concat_volumes(indir, outpath, xdim:int, ydim:int, zdim:int):
    """make a 4D (x,y,z,t) dataset from a directory of 3D (x,y,z) volumes"""

    # initialize
    volnames = io.get_file_paths(indir)
    # concat_vol = np.empty(shape=(xdim, ydim, zdim, len(volnames)))
    mmshape = (xdim, ydim, zdim, len(volnames))
    logging.info(f'mmap shape {mmshape}')
    concat_vol = np.memmap(outpath, dtype='float32', mode='w+', shape=mmshape)

    # load volume and append
    for ii, vol in enumerate(volnames):
        logging.info(f'concatentating: {vol}')
        concat_vol[:, :, :, ii] = io.load(vol)

    logging.info(f'saving to mempath {outpath}')

    concat_vol.flush()

    
    # fp = np.memmap(outpath, dtype='float32', mode='w+', shape=(xdim, ydim, zdim, len(volnames)))
    # fp[:] = concat_vol[:]
    # fp.flush()


def average_volume(outdir, ch: int, nvols: int, xdim: int, ydim: int, zdim: int, flybacknum:int, fname=None):
    """make an average green (ch=0) or red (ch=1) volume from volumes in dirs

    returns: str, path to mean brain
    """

    _check_channel_options(ch)

    green_dir, red_dir = _get_green_red_dirs(outdir)
    if ch == 0:  # ch=0 is green, and ch=1 is red
        vol_dir = green_dir
    else:
        vol_dir = red_dir

    #PIXELS_X, PIXELS_Y, NUMZFRAMES, FLYBACKFRAME, COLOR = _load_hardcoded_params()

    PIXELS_X = xdim
    PIXELS_Y = ydim
    NUMZFRAMES = zdim
    FLYBACKFRAME = zdim - flybacknum 
    COLOR = ['green', 'red']

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

def detrendBackground(
    stackPath, shape, spatialThresh: int, temporalThresh: int, useTemplate: bool = False
):
    """For experiments with visual stimuli, background changes
    over time. Assuming this time-varying background is uniform
    across a z-slice, can estimate background contamination by
    extracting fluctuations in non-brain (dark) pixels.
    %% Inputs %%
    - OUR shape: shape: tuple, (x, y, z, t)
    - Max stack (memmap): nT x nZ x xRes x yRes stack
    - spatialThresh (int): pixel value percentile for determining background pixels
    - temporalThresh (int): percentile for de-meaning background pixels
    - useTemplate: use a mean projection to determine dark pixels. useful for sparse lines
    %% Outputs %%
    - detrendedStack (array): detrended stack
    """
    global stack
    stack = np.memmap(stackPath, dtype="float32", mode="r+", shape=shape)

    # import pdb; pdb.set_trace()    
    # Chunk memmap in time and read substack into RAM. Soft disabled for now  by increasing chunk size
    # Otherwise, I/O is way too slow.
    chunkSize = 1000
    nT = shape[3]
    nChunks = nT // chunkSize
    if nT % chunkSize != 0:
        nChunks += 1    
    start = 0
    stop = min(chunkSize, nT)    
    print("Removing background...", flush=True)    
    for chunk in range(nChunks):
        subStack = np.array(stack[:,:,:,start:stop])
        subStack = np.transpose(subStack, (2, 1,0, 3))  # (Z,X,Y,T):(2, 0,1, 3) , (Z,Y,X,T): (2, 1,0, 3)  
        for zSlice in subStack: #this is really in X?
            for row in zSlice:  # detrend each row
                meanRow = np.mean(row, axis=-1)  # t projection for one scan line
                blackLevel = np.percentile(
                    meanRow, spatialThresh
                )  # black level for scan line
                backgroundPixels = np.where(meanRow <= blackLevel)[0]  # which pixels are darkest in row
                backgroundComp = np.mean(
                    row[backgroundPixels], axis=0
                )  # time trace of dark pixels
                backgroundComp -= np.percentile(
                    backgroundComp, temporalThresh
                )  # subtract bottom x percentile
                backgroundComp = backgroundComp.astype(np.float32)  # cast to float32 (originally int16)
                row[:] = row - backgroundComp
                # import pdb; pdb.set_trace()        
        subStack = np.transpose(subStack, (2,1,0,3)) #restore XYZT from ZXYT:(1,2,0,3) restore XYZT from ZYXT: (2,1,0,3)
        stack[:,:,:,start:stop] = subStack[:]
        start = stop
        stop = min(stop + chunkSize, nT)
        stack.flush()    
        del stack

def extract_volumes_from_nii(stackPath, dir: str, shape, color:str):
    """get single volumes from nii and store in correct dir"""

    global stack 
    stack = np.memmap(stackPath, dtype="float32", mode="r+", shape=shape)

    NUMZFRAMES = shape[2]
    nT = shape[3]
    print(stack.shape)

    logging.info("extracting volumes from nii files")
    volume = _initialize_fly_volume(stack[:,:,0,0], NUMZFRAMES)
    logging.info("initializing the first volume")

    for vN in range(nT):
        logging.info(f'volume: {vN}')
        volume = stack[:, :,:, vN]
        fpath = f"{color}_volume_{vN}.nii"
        io.save(os.path.join(dir, fpath), volume)

# --------*--------*--------*--------
# helper functions
# --------*--------*--------*--------


def _load_hardcoded_params(): #depricated example
    xdim = 256
    ydim = 128
    zdim = 50  # number of slices in Z direction (N)

    # how many slice left after removing last frame (ie 'flyback frame')
    FLYBACKFRAME = zdim - 1
    COLOR = ['green', 'red']

    return xdim, ydim, zdim, FLYBACKFRAME, COLOR


def _initialize_fly_volume(arr, FLYBACKFRAME):
    return np.zeros((arr.shape[0],
                     arr.shape[1],
                     FLYBACKFRAME))


def _check_channel_options(ch):
    ch_options = [0, 1]
    if ch not in ch_options:
        raise ValueError("Invalid ch. Expected 0 or 1")


def _check_dirs(tiffdir, outdir):
    """check input and output directories"""
    if not os.path.exists(tiffdir):
        logging.info(f'tiffdir: {tiffdir}')
        raise FileNotFoundError

    if not os.path.exists(outdir):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), outdir)


def _get_green_red_dirs(outdir):
    """prep directories for red and green volumes"""
    green_dir = os.path.join(outdir, 'green_ch')
    if not os.path.exists(green_dir):
        os.mkdir(green_dir)

    red_dir = os.path.join(outdir, 'red_ch')
    if not os.path.exists(red_dir):
        os.mkdir(red_dir)

    logging.info(f'gcamp volumes, pre-moco: {green_dir}')
    logging.info(f'tdtom volumes, pre-moco: {red_dir}')

    return green_dir, red_dir

def _get_green_red_dirs_proj(outdir):
    """prep directories for red and green volumes"""
    green_dir = os.path.join(outdir, 'green_ch_raw')
    if not os.path.exists(green_dir):
        os.mkdir(green_dir)

    red_dir = os.path.join(outdir, 'red_ch_raw')
    if not os.path.exists(red_dir):
        os.mkdir(red_dir)

    logging.info(f'gcamp volumes, pre-moco: {green_dir}')
    logging.info(f'tdtom volumes, pre-moco: {red_dir}')

    return green_dir, red_dir

def _list_tiffs(tiffdir):
    """returns list of tiff files in a directory"""
    return sorted(glob.glob(tiffdir + '/*.tif'))


def _extract_volumes_from_tiffs(list_of_tiffs, zdim: int, flybacknum:int, green_dir: str, red_dir: str):
    """get single volumes from tiffs and store in correct green and red dir"""

    NUMZFRAMES = zdim
    FLYBACKFRAME = zdim - flybacknum 
    COLOR = ['green', 'red']

    logging.info("extracting volumes from tiff files")
    THISPAGE = 0  # page to process in queue, range [0, last_page_of_experiment]
    STARTPAGE = 0  # start at 1st channel (gcamp)
    for TIFFIDX, tiff in enumerate(list_of_tiffs):
        tiff_array = io.load(tiff)

        if TIFFIDX == 0:  # initialize the first volumes
            green_volume = _initialize_fly_volume(tiff_array, FLYBACKFRAME)
            red_volume = _initialize_fly_volume(tiff_array, FLYBACKFRAME)
            logging.info("initializing the first volume")

        NUMPAGES = tiff_array.shape[2]

        logging.info(f"tiff loaded ---> x: {tiff_array.shape[0]}, y: {tiff_array.shape[1]}, pages: {NUMPAGES}")

        # go thru each page in tiff
        for iPage in np.arange(STARTPAGE, NUMPAGES, 1):
            logging.info(f"decoding page # {THISPAGE}")
            ch, frame, vol = io.decode_tiff_page(THISPAGE, NUMZFRAMES)
            logging.info(f"decoded info ---> ch: {ch}, frame: {frame}, vol: {vol}")

            if ch == 0:  # ch=0 is green, and ch=1 is red
                if frame < FLYBACKFRAME:
                    # append frame to current volume
                    green_volume[:, :, frame] = tiff_array[:, :, iPage]
                elif frame == FLYBACKFRAME:
                    # save current volume
                    logging.info(f"saving volume {vol} to file")
                    fpath = f"{COLOR[ch]}_volume_{vol}.nii"
                    io.save(os.path.join(green_dir, fpath), green_volume)

                    # re-initialize the volume
                    green_volume = _initialize_fly_volume(tiff_array, FLYBACKFRAME)
            else:  # second channel
                if frame < FLYBACKFRAME:
                    # append frame to current volume
                    red_volume[:, :, frame] = tiff_array[:, :, iPage]
                elif frame == FLYBACKFRAME:
                    # save current volume
                    logging.info(f"saving volume {vol} to file")
                    fpath = f"{COLOR[ch]}_volume_{vol}.nii"
                    io.save(os.path.join(red_dir, fpath), red_volume)

                    # re-initialize the volume
                    red_volume = _initialize_fly_volume(tiff_array, FLYBACKFRAME)

            # next page to process in queue
            THISPAGE += 1

def _extract_volumes_from_tiffs_RG(list_of_tiffs, zdim: int, flybacknum:int, green_dir: str, red_dir: str):
    """get single volumes from tiffs and store in correct green and red dir"""

    NUMZFRAMES = zdim
    FLYBACKFRAME = zdim - flybacknum 
    COLOR = ['red','green']

    logging.info("extracting volumes from tiff files: red first green second")
    THISPAGE = 0  # page to process in queue, range [0, last_page_of_experiment]
    STARTPAGE = 0  # start at 1st channel (gcamp)
    for TIFFIDX, tiff in enumerate(list_of_tiffs):
        tiff_array = io.load(tiff)

        if TIFFIDX == 0:  # initialize the first volumes
            green_volume = _initialize_fly_volume(tiff_array, FLYBACKFRAME)
            red_volume = _initialize_fly_volume(tiff_array, FLYBACKFRAME)
            logging.info("initializing the first volume")

        NUMPAGES = tiff_array.shape[2]

        logging.info(f"tiff loaded ---> x: {tiff_array.shape[0]}, y: {tiff_array.shape[1]}, pages: {NUMPAGES}")

        # go thru each page in tiff
        for iPage in np.arange(STARTPAGE, NUMPAGES, 1):
            logging.info(f"decoding page # {THISPAGE}")
            ch, frame, vol = io.decode_tiff_page(THISPAGE, NUMZFRAMES)
            logging.info(f"decoded info ---> ch: {ch}, frame: {frame}, vol: {vol}")

            if ch == 0:  # ch=0 is red, and ch=1 is green
                if frame < FLYBACKFRAME:
                    # append frame to current volume
                    red_volume[:, :, frame] = tiff_array[:, :, iPage]
                elif frame == FLYBACKFRAME:
                    # save current volume
                    logging.info(f"saving volume {vol} to file")
                    fpath = f"{COLOR[ch]}_volume_{vol}.nii"
                    io.save(os.path.join(red_dir, fpath), red_volume)

                    # re-initialize the volume
                    red_volume = _initialize_fly_volume(tiff_array, FLYBACKFRAME)
            else:  # second channel
                if frame < FLYBACKFRAME:
                    # append frame to current volume
                    green_volume[:, :, frame] = tiff_array[:, :, iPage]
                elif frame == FLYBACKFRAME:
                    # save current volume
                    logging.info(f"saving volume {vol} to file")
                    fpath = f"{COLOR[ch]}_volume_{vol}.nii"
                    io.save(os.path.join(green_dir, fpath), green_volume)

                    # re-initialize the volume
                    green_volume = _initialize_fly_volume(tiff_array, FLYBACKFRAME)

            # next page to process in queue
            THISPAGE += 1