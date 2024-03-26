#!/bin/bash

#SBATCH --job-name=240306_run_4-tiff-to-dff
#SBATCH --cpus-per-task=4
#SBATCH --time=1:59:00
#SBATCH --mem=64000
#SBATCH --mail-user=albertl@princeton.edu
#SBATCH --mail-type=ALL
#SBATCH --output='slurm_log/%j-tiff-to-dff-240306_run_4.log'

module load anaconda3/2021.11
source /scratch/gpfs/albertl/atlas_dir/bin/activate

TIFFDIR="/scratch/gpfs/albertl/rigE_data/240306_run_4/raw"
OUTDIR="/scratch/gpfs/albertl/rigE_data/240306_run_4"
FLYTRIAL="240306_run_4"  # CHANGE LOG NAME

srun python3 tiff_to_dff.py --tiffdir $TIFFDIR --outdir $OUTDIR --flytrial $FLYTRIAL # within tiff_to_dff you can change resolution (also hardcoded in vols.py), green/red channel, type of alignment
