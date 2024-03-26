#!/bin/bash

#SBATCH --job-name=6s_natstim1_5thA2-tiff-to-dff
#SBATCH --cpus-per-task=4
#SBATCH --time=71:59:00
#SBATCH --mem=64000
#SBATCH --mail-user=albertl@princeton.edu
#SBATCH --mail-type=ALL
#SBATCH --output='slurm_log/%j-tiff-to-dff-6s_natstim1_5thA2.log'

module load anaconda3/2021.11
source /scratch/gpfs/albertl/atlas_dir/bin/activate

TIFFDIR="/scratch/gpfs/albertl/lightbead_data/6s_natstim1_5thA2/raw"
OUTDIR="/scratch/gpfs/albertl/lightbead_data/6s_natstim1_5thA2"
FLYTRIAL="6s_natstim1_5thA2"  # CHANGE LOG NAME

srun python3 tiff_to_dff_lightbead_green.py --tiffdir $TIFFDIR --outdir $OUTDIR --flytrial $FLYTRIAL # within tiff_to_dff you can change resolution (also hardcoded in vols.py), green/red channel, type of alignment
