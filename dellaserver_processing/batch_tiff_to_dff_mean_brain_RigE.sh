#!/bin/bash

#SBATCH --job-name=mean_brain
#SBATCH --cpus-per-task=1
#SBATCH --time=4:00:00
#SBATCH --mem=64000
#SBATCH --mail-user=albertl@princeton.edu
#SBATCH --mail-type=ALL
#SBATCH --output='slurm_log/%j-tiff-to-dff-121324_run1.log'

module load anaconda3/2021.11
source /scratch/gpfs/albertl/atlas_dir/bin/activate

TIFFDIR="/scratch/gpfs/albertl/rigE_data/121324_run1/raw"
OUTDIR="/scratch/gpfs/albertl/rigE_data/121324_run1"
FLYTRIAL="121324_run1"  # CHANGE LOG NAME
XDIM=256
YDIM=128
ZDIM=50 # (N, flyback)
FLYBACKNUM=3
FIRST_CHANNEL=1 # flag 0 if channel order is GR, 1 if channel order is RG
MOCO_CHANNEL=1 # 0 for green, 1 for red
NUMVOLS=900 #need to count 0, so is last plus one: 192 for 10 files test, 600 for all
PROJECTOR=1 #boolean: projector subtraction 0 = no, 1 = yes
S_THRESH=10 #spatial threshold for projector correction (if proj is false, not used) pixel value percentile for black level in the scan line
T_THRESH=50 #temporal threshold for projector correction (if proj is false, not used) percentile for de-meaning background pixels

python3 tiff_to_dff_mean_brain_RigE.py --tiffdir $TIFFDIR --outdir $OUTDIR --flytrial $FLYTRIAL \
--xdim $XDIM --ydim $YDIM --zdim $ZDIM --flybacknum $FLYBACKNUM --firstchannel $FIRST_CHANNEL --mocochannel $MOCO_CHANNEL \
--numvols $NUMVOLS --projstim $PROJECTOR --spatialthresh $S_THRESH --temporalthresh $T_THRESH
