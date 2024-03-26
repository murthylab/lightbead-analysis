#!/bin/bash

#SBATCH --job-name=20220707_basal_1-tiff-to-local-atlas
#SBATCH --cpus-per-task=4
#SBATCH --time=1:05:00
#SBATCH --mem=64000
#SBATCH --mail-user=albertl@princeton.edu
#SBATCH --mail-type=ALL
#SBATCH --output='slurm_log/%j-20220707_basal_1-tiff-to-local-atlas.log'

module load anaconda3/2021.11
source /scratch/gpfs/albertl/atlas_dir/bin/activate 

# if works smoothly, change to scratch dirs
TIFFDIR="/scratch/gpfs/albertl/atlas_data/20220707_basal_1/raw"
OUTDIR="/scratch/gpfs/albertl/atlas_data/20220707_basal_1"
FLYNAME="20220707_basal_1"
DONEMOCO=0  # 0=False, 1=True (switch this depending on whether moco has been run already or not)
X=512
Y=1024
Z=249 #299
N=50 # normally 100

srun python3 tiff_to_local_atlas.py --tiffdir $TIFFDIR --outdir $OUTDIR --flyname $FLYNAME --doneMoco $DONEMOCO --shape $X $Y $Z $N
