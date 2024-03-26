def main():
    """
        1 extract volumes from tiff
        2 make average brain
        3 motion correct using SyN -- parallelized (see paramoco.py)
        4 make average moco brain
        5 save as .nii
    """
    import sys
    import os
    sys.path.insert(1, '../../src')  # add src package to path
    baseDir = os.path.dirname(os.getcwd())
    sys.path.append(baseDir)

    import argparse
    import logging

    from brainviz import localatlas

    from time import time

    # --------*--------*--------*--------
    # set logging level
    # --------*--------*--------*--------

    lvl = logging.INFO  # logging.INFO or logging.DEBUG

    logging.basicConfig(filename='log/tiff_to_local_atlas.log',
                        format='%(message)s',
                        level=lvl)

    logging.getLogger().addHandler(logging.StreamHandler())

    LOG = logging.getLogger(__name__)

    LOG.info("---start---")
    ts = time()  # start timer

    # --------*--------*--------*--------
    # parse args
    # --------*--------*--------*--------

    parser = argparse.ArgumentParser(description='generate local atlas from high resolution structural scans stored in tiffs')

    parser.add_argument('--tiffdir', dest='tiffdir', type=str, help="path to folder containing all tiffs from scanimage for this fly")

    parser.add_argument('--outdir', dest='outdir', type=str, help="path to folder that will contain saved outputs (should be empty folder)")

    parser.add_argument('--flyname', dest='flyname', type=str, help="name of fly, e.g.'220210' ")

    parser.add_argument('--doneMoco', dest='doneMoco', type=int, help="0 if extracting volumes from tiffs, 1 if moco vols are ready")

    parser.add_argument('--shape', dest='shape', nargs='+', type=int, help="x y z n: make sure z excludes flyback frame.")

    args = parser.parse_args()

    # --------*--------*--------*--------
    # prepare directories
    # --------*--------*--------*--------

    # path to pre and post-moco atlas (x, y, z)
    pre_moco_path = os.path.join(args.outdir, 'local_mean_0.nii')
    atlas_path = os.path.join(args.outdir, f'{args.flyname}_local_atlas.nii')

    # create folders hold the n-volumes (x,y,z,vols)
    vol_path = os.path.join(args.outdir, 'vols')
    moco_path = os.path.join(args.outdir, 'moco_vols')

    if not os.path.exists(vol_path):
        os.mkdir(vol_path)

    if not os.path.exists(moco_path):
        os.mkdir(moco_path)

    # --------*--------*--------*--------
    # hyperparams
    # --------*--------*--------*--------

    # resolution = (0.49, 0.49, 1)
    ch = 1  # 0 = gcamp, 1 = tdtom
    # shape = (1024, 700, 249, 100)
    shape = tuple(args.shape)

    # --------*--------*--------*--------
    # run pipeline
    # --------*--------*--------*--------

    if args.doneMoco == 1:  # moco vols are ready
        localatlas.average_volume(moco_path, atlas_path, shape)
    else:  # need to extract vols from tiff and make 1st pass average brain
        localatlas.get_from_tiffs(args.tiffdir, vol_path, ch, shape)
        localatlas.average_volume(vol_path, pre_moco_path, shape)

    # --------*--------*--------*--------
    # --------*--------*--------*--------

    te = time()  # stop timer
    LOG.info(f"took {te - ts} seconds")
    LOG.info("---done---")
    LOG.info("")
    # END TEST CODE HERE


if __name__ == '__main__':
    main()
