#!/bin/bash

#SBATCH --job-name=concatenate
#SBATCH --cpus-per-task=4
#SBATCH --time=32:05:00
#SBATCH --mem=64000
#SBATCH --mail-user=wg0346@princeton.edu
#SBATCH --mail-type=ALL
#SBATCH --output='slurm_log/%j-tiff-to-dff-06192024_8m_a1_r1.log'

module load anaconda3/2021.11
conda activate ants_env

TIFFDIR="/scratch/gpfs/wg0346/auditory_data/06192024_8m_a1_r1/raw"
OUTDIR="/scratch/gpfs/wg0346/auditory_data/06192024_8m_a1_r1"
FLYTRIAL="06192024_8m_a1_r1"  # CHANGE LOG NAME

python3 tiff_to_dff_LB_concat_60Hz.py --tiffdir $TIFFDIR --outdir $OUTDIR --flytrial $FLYTRIAL # within tiff_to_dff you can change green/red channel, type of alignment
