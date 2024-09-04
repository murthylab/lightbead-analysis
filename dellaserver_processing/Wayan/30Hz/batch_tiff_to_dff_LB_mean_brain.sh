#!/bin/bash

#SBATCH --job-name=mean_brain
#SBATCH --cpus-per-task=4
#SBATCH --time=2:05:00
#SBATCH --mem=64000
#SBATCH --mail-user=wg0346@princeton.edu
#SBATCH --mail-type=ALL
#SBATCH --output='slurm_log/%j-tiff-to-dff-07082024_8m_a1_r4.log'

module load anaconda3/2021.11
conda activate ants_env

TIFFDIR="/scratch/gpfs/wg0346/auditory_data/07082024_8m_a1_r4/raw"
OUTDIR="/scratch/gpfs/wg0346/auditory_data/07082024_8m_a1_r4"
FLYTRIAL="07082024_8m_a1_r4"  # CHANGE LOG NAME

python3 tiff_to_dff_LB_mean_brain.py --tiffdir $TIFFDIR --outdir $OUTDIR --flytrial $FLYTRIAL # within tiff_to_dff you can change green/red channel, type of alignment
