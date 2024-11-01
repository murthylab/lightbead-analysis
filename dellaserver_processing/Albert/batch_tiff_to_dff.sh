#!/bin/bash

#SBATCH --job-name=wayan_a2_r2_volstest-tiff-to-dff
#SBATCH --cpus-per-task=4
#SBATCH --time=3:59:00
#SBATCH --mem=32000
#SBATCH --mail-user=albertl@princeton.edu
#SBATCH --mail-type=ALL
#SBATCH --output='slurm_log/%j-tiff-to-dff-wayan_a2_r2_volstest.log'

module load anaconda3/2021.11
source /scratch/gpfs/albertl/atlas_dir/bin/activate

TIFFDIR="/scratch/gpfs/albertl/rigE_data/wayan_a2_r2_volstest/raw"
OUTDIR="/scratch/gpfs/albertl/rigE_data/wayan_a2_r2_volstest"
FLYTRIAL="wayan_a2_r2_volstest"  # CHANGE LOG NAME

srun python3 tiff_to_dff.py --tiffdir $TIFFDIR --outdir $OUTDIR --flytrial $FLYTRIAL # within tiff_to_dff you can change resolution (also hardcoded in vols.py), green/red channel, type of alignment
