#!/bin/bash

#SBATCH --job-name=motion_correction
#SBATCH --cpus-per-task=4
#SBATCH --time=1:05:00
#SBATCH --mem=128000
#SBATCH --mail-user=wg0346@princeton.edu
#SBATCH --mail-type=ALL
#SBATCH --output='slurm_log/%A-%a-tiff-to-dff-07082024_8m_a1_r4.log'
#SBATCH --array=1-300

module load anaconda3/2021.11
conda activate ants_env

TIFFDIR="/scratch/gpfs/wg0346/auditory_data/07082024_8m_a1_r4/raw"
OUTDIR="/scratch/gpfs/wg0346/auditory_data/07082024_8m_a1_r4"
FLYTRIAL="07082024_8m_a1_r4"  # CHANGE LOG NAME
TOTAL_BATCHES=${SLURM_ARRAY_TASK_COUNT}
BATCH=${SLURM_ARRAY_TASK_ID}

python3 tiff_to_dff_LB_mc.py --tiffdir $TIFFDIR --outdir $OUTDIR --flytrial $FLYTRIAL --batch_index=${BATCH} --total_batches=${TOTAL_BATCHES}
