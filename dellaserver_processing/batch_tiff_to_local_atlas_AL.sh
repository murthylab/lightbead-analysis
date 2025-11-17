#!/bin/bash

#SBATCH --job-name=082824_anat1-tiff-to-local-atlas
#SBATCH --cpus-per-task=4
#SBATCH --time=1:05:00
#SBATCH --mem=64000
#SBATCH --mail-user=albertl@princeton.edu
#SBATCH --mail-type=ALL
#SBATCH --output='slurm_log/%j-082824_anat1-tiff-to-local-atlas.log'

module load anaconda3/2021.11
source /scratch/gpfs/albertl/atlas_dir/bin/activate 

# if works smoothly, change to scratch dirs
TIFFDIR="/scratch/gpfs/albertl/rigE_data/082824_anat1/raw"
OUTDIR="/scratch/gpfs/albertl/rigE_data/082824_anat1"
FLYNAME="082824_anat1"
DONEMOCO=1  # 0=False, 1=True (switch this depending on whether moco has been run already or not)
FIRST_CHANNEL=1 # flag 0 if channel order is GR, 1 if channel order is RG
X=1024
Y=512
Z=149 
N=50

srun python3 tiff_to_local_atlas.py --tiffdir $TIFFDIR --outdir $OUTDIR --flyname $FLYNAME --doneMoco $DONEMOCO --firstchannel $FIRST_CHANNEL --shape $X $Y $Z $N
