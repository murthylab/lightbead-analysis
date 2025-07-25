def main():
    import os
    import math
    import sys
    sys.path.insert(1, '../../src')  # add src package to path
    baseDir = os.path.dirname(os.getcwd())
    sys.path.append(baseDir)

    import argparse
    import logging

    from flydff import vols_RigE, moco_RigE, signal, viz #moco_g, 
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

    parser.add_argument('--xdim', dest='XDIM', type=int)
    parser.add_argument('--ydim', dest='YDIM', type=int)
    parser.add_argument('--zdim', dest='ZDIM', type=int)
    parser.add_argument('--flybacknum', dest='FLYBACKNUM', type=int)
    parser.add_argument('--mocochannel', dest='MOCO_CHANNEL', type=int)
    
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

    XDIM = args.XDIM
    YDIM = args.YDIM
    ZDIM = args.ZDIM # (N)
    FLYBACKNUM = args.FLYBACKNUM
    MOCO_CHANNEL = args.MOCO_CHANNEL; # 0 for green, 1 for red

    # --------*--------*--------*--------
    # prepare directories
    # --------*--------*--------*--------

    # paths to extracted volumes, pre and post-moco
    green_dir, red_dir = vols_RigE._get_green_red_dirs(args.outdir)
    green_moco_dir, red_moco_dir = moco_RigE._get_green_red_moco_dirs(args.outdir)

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

    # Calculate starting and ending image for this batch
    batch_index = args.batch_index - 1
    total_batches = args.total_batches
    NUMVOLS = len(io.get_file_paths(green_dir))
    batch_size = math.floor(NUMVOLS / total_batches)
    start_image = batch_index * batch_size
    end_image = start_image + batch_size

    if end_image > NUMVOLS:
        end_image = NUMVOLS

    LOG.info(f"batch_index={batch_index}, batch_size={batch_size}, start_image={start_image}, end_image={end_image}")

    # --------*--------*--------*--------
    # run pipeline
    # --------*--------*--------*--------        

    # motion correct
    if doMoCo:
        if MOCO_CHANNEL == 1:
            moco_RigE.apply(args.outdir, fixed_brain_path, start_image=start_image, end_image=end_image)
        #elif MOCO_CHANNEL == 0:
           # moco_g.apply(args.outdir, fixed_brain_path_green)

    # --------*--------*--------*--------
    # --------*--------*--------*--------

    te = time()  # stop timer
    LOG.info(f"took {te - ts} seconds")
    LOG.info("---done---")
    LOG.info("")
    # END TEST CODE HERE


if __name__ == '__main__':
    main()
