#!/usr/bin/python

def main():
    """
    run motion correction ("moco") via ANTsPy by aligning a "moving" volume to a "fixed brain". Uses SLURM arrays to moco in parallel.
    """
    import sys
    import os
    sys.path.insert(1, '../../src')  # add src package to path
    baseDir = os.path.dirname(os.getcwd())
    sys.path.append(baseDir)

    import argparse
    import logging

    import ants
    import glob

    from time import time

    # --------*--------*--------*--------
    # set logging level
    # --------*--------*--------*--------

    lvl = logging.INFO  # logging.INFO or logging.DEBUG

    logging.basicConfig(filename='log/paramoco.log',
                        format='%(message)s',
                        level=lvl)

    logging.getLogger().addHandler(logging.StreamHandler())

    LOG = logging.getLogger(__name__)

    logging.info("---start---")
    ts = time()  # start timer

    # --------*--------*--------*--------
    # parse args
    # --------*--------*--------*--------

    parser = argparse.ArgumentParser(description='align moving brain to reference brain')

    parser.add_argument('--mov', dest='mov', type=str, help="path to folder containing all moving brains for this alignment iteration")

    parser.add_argument('--mov_gr', dest='mov_gr', type=str, help="path to folder containing all green channel moving brains for this alignment iteration")

    parser.add_argument('--ref', dest='ref', type=str, help="path to reference brain")

    parser.add_argument('--transform', dest='transform', type=int, help="0=affine, 1=SyN")

    parser.add_argument('--ARRAYNUMBER', dest='ARRAYNUMBER', type=int, help="given by $SLURM_ARRAY_TASK_ID and refers to one file in the moving brain folder")

    parser.add_argument('--outdir', dest='outdir', type=str, help="path to output folder that will contain the newly aligned volumes")

    args = parser.parse_args()

    # --------*--------*--------*--------
    # params
    # --------*--------*--------*--------

    TRANSFORM = ['Affine', 'SyN']

    # basic checks
    logging.info(f'array number: {args.ARRAYNUMBER}')
    logging.info(f'moving brains folder: {args.mov}')
    logging.info(f'reference brain: {args.ref}')
    logging.info(f'transformation: {TRANSFORM[args.transform]}')
    logging.info(f'output folder: {args.outdir}')
    logging.info('---')

    # --------*--------*--------*--------
    # moco
    # --------*--------*--------*--------

    # load reference
    reference_brain = ants.image_read(args.ref)

    # load moving brain
    vols_regex = args.mov + "/*volume*.nii"
    list_of_volumes = sorted(filter(os.path.isfile, glob.glob(vols_regex)))

    mov_path = list_of_volumes[args.ARRAYNUMBER]
    logging.info(f"moving brain: {mov_path}")

    logging.info('loading...')
    moving_brain = ants.image_read(mov_path)

    # load green moving brain
    vols_regex_gr = args.mov_gr + "/*volume*.nii"
    list_of_volumes_gr = sorted(filter(os.path.isfile, glob.glob(vols_regex_gr)))

    mov_path_gr = list_of_volumes_gr[args.ARRAYNUMBER]
    logging.info(f"moving brain green: {mov_path_gr}")

    logging.info('loading...')
    moving_brain_gr = ants.image_read(mov_path_gr)

    # align moving to reference_brain
    logging.info('aligning moving brain to reference...')
    areg = ants.registration(reference_brain, moving_brain, TRANSFORM[args.transform])

    aligned_brain = areg['warpedmovout']
    # transforms of ants registration
    aligned_transforms = areg['fwdtransforms']
    # apply transforms to green image
    aligned_brain_gr = ants.apply_transforms(reference_brain, moving_brain_gr, aligned_transforms)

    # --------*--------*--------*--------
    # save moco brain
    # --------*--------*--------*--------

    # create folders hold the n-volumes (x,y,z,vols)
    moco_path = os.path.join(args.outdir, 'moco_red_vols')
    moco_path_gr = os.path.join(args.outdir, 'moco_green_vols')

    # save the aligned volume out
    logging.info('saving...')
    file = mov_path.rsplit('/', 1)[-1]  # get file name after path
    fname = os.path.join(moco_path, "mc_" + file)

    ants.image_write(aligned_brain, fname)

    logging.info('saving...')
    file = mov_path_gr.rsplit('/', 1)[-1]  # get file name after path
    fname = os.path.join(moco_path_gr, "mc_" + file)

    ants.image_write(aligned_brain_gr, fname)

    # --------*--------*--------*--------
    # --------*--------*--------*--------

    te = time()  # stop timer
    logging.info(f"took {te - ts} seconds")
    logging.info("---done---")
    logging.info("")
    # END TEST CODE HERE


if __name__ == '__main__':
    main()
