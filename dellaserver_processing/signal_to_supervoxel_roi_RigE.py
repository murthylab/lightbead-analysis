def main():
    import os
    import sys
    sys.path.insert(1, '../../src')  # add src package to path
    baseDir = os.path.dirname(os.getcwd())
    sys.path.append(baseDir)

    import numpy as np
    import h5py

    import argparse
    import logging

    from brainviz import roi

    from time import time
    
    from scipy.stats import zscore

    # --------*--------*--------*--------
    # set logging level
    # --------*--------*--------*--------

    lvl = logging.INFO  # logging.INFO or logging.DEBUG

    logging.basicConfig(filename='log/signal_to_supervox.log',
                        format='%(message)s',
                        level=lvl)

    logging.getLogger().addHandler(logging.StreamHandler())

    LOG = logging.getLogger(__name__)

    LOG.info("---start---")
    ts = time()  # start timer

    # --------*--------*--------*--------
    # parse args
    # --------*--------*--------*--------

    parser = argparse.ArgumentParser(description='extract zscored df/f at supervoxel resolution from 4D motion-corrected gcamp signal')

    parser.add_argument('--mempath', dest='mempath', type=str, help="path to mmap file with 4D data")

    parser.add_argument('--mempath_ext', dest='mempath_ext', type=str, help="path to mmap file to extract data from")

    parser.add_argument('--numvols', dest='numvols', type=int, help="number of volumes, used for shape of mmap -> (x,y,z,NUMVOLS)")

    parser.add_argument('--outdir', dest='outdir', type=str, help="path to folder that will contain saved outputs (should be empty folder)")

    parser.add_argument('--n_clusters', dest='n_clusters', type=int, help="# of supervoxels to extract from each z-slice (e.g. 20, 200, or 2000)")

    parser.add_argument('--flytrial', dest='flytrial', type=str, help="name of flytrial")

    parser.add_argument('--xdim', dest='XDIM', type=int)
    parser.add_argument('--ydim', dest='YDIM', type=int)
    parser.add_argument('--zdim', dest='ZDIM', type=int)
    parser.add_argument('--flybacknum', dest='FLYBACKNUM', type=int)

    parser.add_argument('--f0window', dest='F0WINDOW', type=int)
    parser.add_argument('--sigmaoff', dest='SIGMA_OFFSET', type=float)

    args = parser.parse_args()

    # --------*--------*--------*--------
    # helper functions
    # --------*--------*--------*--------

    def _zscore(x):
        """
        input:
            x: 2D array (n, t)
        """
        x_mean = np.mean(x, axis=-1)
        x_std = np.std(x, axis=-1)
        x = (x - x_mean[:, None]) / x_std[:, None]
        return x

    def _dff(brain, F0_win, sigmaoff):
        """calculated zscored(df/f) based on F0 baseline"""
        # find average signal in first 10 seconds (~20 vols)
        #F0 = np.mean(brain[: , :F0_win], axis=-1)
        F0 = np.mean(brain[:, :F0_win], axis=-1) - (sigmaoff*np.std(brain[:, :F0_win], axis=-1))
        dff = (brain - F0[:, None]) / F0[:, None]   
        
        return dff
        #return _zscore(dff)

    def _save_to_mmap(z_dff, args, iSlice):
        fpath = os.path.join(args.outdir, f"{args.flytrial}_n{args.n_clusters}_t{args.numvols}_slice{iSlice}.mmap")

        fp = np.memmap(fpath, mode='w+', dtype='float32', shape=(args.n_clusters, args.numvols))

        # store the array in the mmap variable
        fp[:] = z_dff[:]
        fp.flush()

    # --------*--------*--------*--------
    # hyper params
    # --------*--------*--------*--------

    XDIM = args.XDIM
    YDIM = args.YDIM
    ZDIM = args.ZDIM # (N)
    FLYBACKNUM = args.FLYBACKNUM

    shape = (XDIM, YDIM, ZDIM-FLYBACKNUM, args.numvols)

    # --------*--------*--------*--------
    # generate cluster labels
    # --------*--------*--------*--------

    brain = np.memmap(args.mempath, dtype='float32', mode='r', shape=shape)
    brain_ext = np.memmap(args.mempath_ext, dtype='float32', mode='r', shape=shape)

    labels = []  # a list of size ZDIM, contains each pixel's cluster identity
    for iSlice in range(ZDIM-FLYBACKNUM):

        # brain is sub-sampled to speed up clustering
        logging.info(f"generating {args.n_clusters} clusters for slice # {iSlice}")
        cluster_model = roi.create_2d_clusters(brain[:, :, iSlice, 0:-1:5], args.n_clusters, 'tmp/cluster_mem')

        labels.append(cluster_model.labels_)

    # save the labels and args
    label_path = os.path.join(args.outdir, f"{args.flytrial}_n{args.n_clusters}_labels.h5")

    hf = h5py.File(label_path, 'w')
    hf.create_dataset('labels', data=labels)
    hf.close()

    # --------*--------*--------*--------
    # get zscore df/f for each cluster in each slice
    # --------*--------*--------*--------

    for iSlice in range(ZDIM-FLYBACKNUM):
        mean_signal = np.empty(shape=(args.numvols, args.n_clusters))

        logging.info(f"operating on slice # {iSlice}")

        for iVol in range(args.numvols):
            mean_supervox, _ = roi.get_supervoxel_mean_2D(brain_ext[:, :, iSlice, iVol], labels[iSlice], args.n_clusters)

            mean_signal[iVol] = mean_supervox

        logging.info("calculating zscored(df/f)")
        z_dff = _dff(mean_signal.T, args.F0WINDOW, args.SIGMA_OFFSET)  # QUESTION: do I need zscored?

        logging.info("saving to mmap")
        _save_to_mmap(z_dff, args, iSlice)

        # try saving as h5 as well
        label_path_z = os.path.join(args.outdir, f"{args.flytrial}_n{args.n_clusters}_t{args.numvols}_slice{iSlice}.h5")
        hf = h5py.File(label_path_z, 'w')
        hf.create_dataset('dff', data=z_dff)
        hf.close()

    # --------*--------*--------*--------
    # --------*--------*--------*--------

    te = time()  # stop timer
    LOG.info(f"took {te - ts} seconds")
    LOG.info("---done---")
    LOG.info("")
    # END TEST CODE HERE


if __name__ == '__main__':
    main()
