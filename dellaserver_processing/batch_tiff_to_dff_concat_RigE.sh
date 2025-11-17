#!/bin/bash

#SBATCH --job-name=concatenate
#SBATCH --cpus-per-task=1
#SBATCH --time=1:05:00
#SBATCH --mem=64000
#SBATCH --mail-user=albertl@princeton.edu
#SBATCH --mail-type=ALL
#SBATCH --output='slurm_log/%j-tiff-to-dff-102524_run1.log'

module load anaconda3/2021.11
source /scratch/gpfs/albertl/atlas_dir/bin/activate

TIFFDIR="/scratch/gpfs/albertl/rigE_data/102524_run1/raw"
OUTDIR="/scratch/gpfs/albertl/rigE_data/102524_run1"
FLYTRIAL="102524_run1"  # CHANGE LOG NAME
XDIM=256
YDIM=128
ZDIM=50 # (N, flyback)
FLYBACKNUM=3
MOCO_CHANNEL=1 # 0 for green, 1 for red
#raw motion corrected fluorescence in the green_moco mmap file
# computes both dff and zscored dff per voxel
F0WINDOW=100 #initial frames for computing F0 for voxels
HPFSIGMA=375 # high pass filter on voxels, applies to dff and zdff but not green_moco original default sigma is 374
# still to do: pull out hardcoded signal

python3 tiff_to_dff_concat_RigE.py --tiffdir $TIFFDIR --outdir $OUTDIR --flytrial $FLYTRIAL \
--xdim $XDIM --ydim $YDIM --zdim $ZDIM --flybacknum $FLYBACKNUM --mocochannel $MOCO_CHANNEL \
--f0window $F0WINDOW --hpfsigma $HPFSIGMA