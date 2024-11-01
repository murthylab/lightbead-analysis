#!/bin/bash

#SBATCH --job-name=102524_run1-signal-to-supervoxel-ROI
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=15:05:00
#SBATCH --mem=32000
#SBATCH --mail-user=albertl@princeton.edu
#SBATCH --mail-type=ALL
#SBATCH --output='slurm_log/%j-102524_run1-signal-to-supervoxel-ROI-n1000.log'

module load anaconda3/2021.11
source /scratch/gpfs/albertl/atlas_dir/bin/activate

# if works smoothly, change to scratch dirs
MEMPATH="/scratch/gpfs/albertl/rigE_data/102524_run1/zscore_102524_run1_dff.mmap"  #mmap that clusters are computed from: zdff are z-scored dffs by voxel
MEMPATH_EXTRACT="/scratch/gpfs/albertl/rigE_data/102524_run1/102524_run1_moco_green.mmap" #mmap that values are pulled from: dff is computed by voxel, green_moco are motion corrected raw values
# clustering operation will currently return z-scored dff of the inputs
OUTDIR="/scratch/gpfs/albertl/rigE_data/102524_run1/supervoxels"
FLYTRIAL="102524_run1"
XDIM=256
YDIM=128
ZDIM=50 # (N, flyback)
FLYBACKNUM=3
NUMVOLS=600
#clustering parameters TODO: write two modes: one for F avg over voxels to compute df/f, one for averaging across zdff
N_CLUSTERS=500  # 20, 200, 2000, 20000
F0WINDOW=100 #initial frames for computing F0 for supervoxels
SIGMA_OFFSET=0.8416 #F_0 = mean - offset*sigma(F)


python3 signal_to_supervoxel_roi_RigE.py --mempath $MEMPATH --mempath_ext $MEMPATH_EXTRACT \
--xdim $XDIM --ydim $YDIM --zdim $ZDIM --flybacknum $FLYBACKNUM \
--numvols $NUMVOLS --outdir $OUTDIR --n_clusters $N_CLUSTERS --flytrial $FLYTRIAL \
--f0window $F0WINDOW --sigmaoff $SIGMA_OFFSET