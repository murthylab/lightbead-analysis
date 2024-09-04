#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MOCO: motion correct volumes
- align volumes to a fixed brain using ANTsPy SyN in the green channel

24 mar 2022
@author: sama ahmed. albert lin

todo:
[ ] refactor: use memmapping instead of saving to a directory
[ ] refactor: move directory checks etc out of moco.py to prepdirs.py
"""

import os
import errno
import logging

import numpy as np
import ants

import sys
sys.path.insert(1, '../../src')  # add src/brainviz package to path
from brainviz import io  # noqa: E402

# --------*--------*--------*--------
# primary functions
# --------*--------*--------*--------


def apply(outdir, fixed_brain_path):
    green_dir = _check_dirs(outdir)
    green_moco_dir = _get_green_moco_dirs(outdir)

    fixed = ants.image_read(fixed_brain_path)

    NUMVOLS = len(io.get_file_paths(green_dir))
    for ii in np.arange(NUMVOLS):
        g_name = f'green_volume_{ii}.nii'

        logging.info(f'motion correcting: {g_name}')
        green = ants.image_read(os.path.join(green_dir, g_name))

        # align green volume to fixed mean brain using SyN
        moco_green = ants.registration(fixed, green, type_of_transform='SyN')

        # save motion corrected green volume
        fpath = f"moco_green_volume_{ii}.nii"
        logging.info(f'saving {fpath}')
        ants.image_write(moco_green["warpedmovout"], os.path.join(green_moco_dir, fpath))


# --------*--------*--------*--------
# helper functions
# --------*--------*--------*--------


def _get_green_moco_dirs(outdir):
    """prep directories for green volumes"""
    green_moco_dir = os.path.join(outdir, 'green_moco_ch')
    if not os.path.exists(green_moco_dir):
        os.mkdir(green_moco_dir)


    logging.info(f'moco gcamp volumes: {green_moco_dir}')

    return green_moco_dir


def _check_dirs(outdir):
    """check pre-moco directories"""
    green_dir = os.path.join(outdir, 'green_ch')

    if not green_dir:
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), green_dir)


    return green_dir
