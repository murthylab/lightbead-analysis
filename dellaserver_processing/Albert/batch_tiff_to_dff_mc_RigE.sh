#!/bin/bash

#SBATCH --job-name=motion_correction
#SBATCH --cpus-per-task=4
#SBATCH --time=0:30:00
#SBATCH --mem=128000
#SBATCH --mail-user=albertl@princeton.edu
#SBATCH --mail-type=ALL
#SBATCH --output='slurm_log/%A-%a-tiff-to-dff-102524_run1.log'
#SBATCH --array=1-100 #for now this must divide cleanly through the number of volumes

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
TOTAL_BATCHES=${SLURM_ARRAY_TASK_COUNT}
BATCH=${SLURM_ARRAY_TASK_ID}

python3 tiff_to_dff_mc_RigE.py --tiffdir $TIFFDIR --outdir $OUTDIR --flytrial $FLYTRIAL \
--xdim $XDIM --ydim $YDIM --zdim $ZDIM --flybacknum $FLYBACKNUM --mocochannel $MOCO_CHANNEL \
--batch_index=${BATCH} --total_batches=${TOTAL_BATCHES}
