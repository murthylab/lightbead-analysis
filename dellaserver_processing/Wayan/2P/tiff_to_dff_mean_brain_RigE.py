def main():
    import os
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

    XDIM = 256
    YDIM = 128
    ZDIM = 49
    MOCO_CHANNEL = 1; # 0 for green, 1 for red

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

    # --------*--------*--------*--------
    # run pipeline
    # --------*--------*--------*--------

    # extract individual green and red dvolumes from tiffs
    if doGetVols:
        vols_RigE.get_from_tiffs(args.tiffdir, args.outdir)

    # save mean-tdtomato volume. Use later for moco and aligning to atlas
    if doMakeMean:
        vols_RigE.average_volume(args.outdir, ch=1, nvols=100, fname=fixed_brain)
        vols_RigE.average_volume(args.outdir, ch=0, nvols=100, fname=fixed_brain_green)

    te = time()  # stop timer
    LOG.info(f"took {te - ts} seconds")
    LOG.info("---done---")
    LOG.info("")
    # END TEST CODE HERE


if __name__ == '__main__':
    main()
