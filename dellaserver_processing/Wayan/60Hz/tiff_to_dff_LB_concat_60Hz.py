def main():
    import os
    import math
    import sys
    sys.path.insert(1, '../../src')  # add src package to path
    baseDir = os.path.dirname(os.getcwd())
    sys.path.append(baseDir)

    import argparse
    import logging

    from flydff import vols_LB_60Hz, moco_LB, signal, viz #moco_g, 
    from brainviz import io

    from time import time

    # --------*--------*--------*--------
    # set logging level
    # --------*--------*--------*--------

    lvl = logging.INFO  # logging.INFO or logging.DEBUG

    logging.basicConfig(filename='log/tiff_to_dff.log',
                        format='%(message)s',
                        level=lvl)

    logging.getLogger().addHandler(logging.StreamHandler())

    LOG = logging.getLogger(__name__)

    LOG.info("---start---")
    ts = time()  # start timer

    # --------*--------*--------*--------
    # parse args
    # --------*--------*--------*--------

    parser = argparse.ArgumentParser(description='extract motion corrected df/f from tiffs')

    parser.add_argument('--tiffdir', dest='tiffdir', type=str, help="path to folder containing all tiffs from scanimage for this flytrial")

    parser.add_argument('--outdir', dest='outdir', type=str, help="path to folder that will contain saved outputs (should be empty folder)")

    parser.add_argument('--flytrial', dest='flytrial', type=str, help="name of flytrial")

    parser.add_argument('--batch_index', dest='batch_index', type=int, help="Index of the image batch. Start index should be 1.")
    parser.add_argument('--total_batches', dest='total_batches', type=int, help="Total number of image batches.")

    args = parser.parse_args()

    # --------*--------*--------*--------
    # debug ?
    # --------*--------*--------*--------

    doDebug = False  # if False, run whole pipeline from start to finish
    if doDebug:
        doGetVols = 0
        doMakeMean = 0
        doMoCo = 0  # TODO _ SAVE MOCO VOLUMES DIRECTLY TO MEMMAP
        doConcatMoCo = 0
        doGetDFF = 1
        doPlot = 1
    else:
        doGetVols = True
        doMakeMean = True
        doMoCo = True
        doConcatMoCo = True
        doGetDFF = True
        doPlot = True

    # --------*--------*--------*--------
    # hyper params
    # --------*--------*--------*--------

    XDIM = 226
    YDIM = 216
    ZDIM = 27
    MOCO_CHANNEL = 1; # 0 for green, 1 for red

    # --------*--------*--------*--------
    # prepare directories
    # --------*--------*--------*--------

    # paths to extracted volumes, pre and post-moco
    green_dir, red_dir = vols_LB_60Hz._get_green_red_dirs(args.outdir)
    green_moco_dir, red_moco_dir = moco_LB._get_green_red_moco_dirs(args.outdir)

    # paths to concatenated, 4D, motion corrected gcamp and tdtomato arrays
    greenpath = os.path.join(args.outdir, f'{args.flytrial}_moco_green.mmap')
    redpath = os.path.join(args.outdir, f'{args.flytrial}_moco_red.mmap')

    # path to low-res mean tdtomato 3D volume
    fixed_brain = f'{args.flytrial}_mean_tom.nii'
    fixed_brain_path = os.path.join(args.outdir, fixed_brain)

    # path to low-res mean gcamp 3D volume
    fixed_brain_green = f'{args.flytrial}_mean_G.nii'
    fixed_brain_path_green = os.path.join(args.outdir, fixed_brain_green)

    # path to df/f and zscored(df/f)
    dff_name = f'{args.flytrial}_dff.mmap'
    dffpath = os.path.join(args.outdir, dff_name)
    zdffpath = os.path.join(args.outdir, f'zscore_{dff_name}')


    # --------*--------*--------*--------
    # run pipeline
    # --------*--------*--------*--------        


    # concatenate motion correction volumes
    if doConcatMoCo:
        vols_LB_60Hz.concat_volumes(green_moco_dir, greenpath, xdim=XDIM, ydim=YDIM, zdim=ZDIM)
        vols_LB_60Hz.concat_volumes(red_moco_dir, redpath, xdim=XDIM, ydim=YDIM, zdim=ZDIM)

    # extract df/f and zscored(df/f) --> save
    if doGetDFF:
        NUMVOLS = len(io.get_file_paths(green_moco_dir))
        LOG.info(f'number of volumes in green_moco_dir: {NUMVOLS}')
        signal.getdff(greenpath, args.outdir, dff_name, shape=(XDIM, YDIM, ZDIM, NUMVOLS))

    # plot mean green and red, annotate with args.flytrial
    if doPlot:
        NUMVOLS = len(io.get_file_paths(green_moco_dir))

        max_proj, mean_green, mean_red, mean_zdff = viz.quality_check(args.outdir, args.flytrial, fixed_brain_path, greenpath, redpath, zdffpath, shape=(XDIM, YDIM, ZDIM, NUMVOLS))

    # --------*--------*--------*--------
    # --------*--------*--------*--------

    te = time()  # stop timer
    LOG.info(f"took {te - ts} seconds")
    LOG.info("---done---")
    LOG.info("")
    # END TEST CODE HERE


if __name__ == '__main__':
    main()
