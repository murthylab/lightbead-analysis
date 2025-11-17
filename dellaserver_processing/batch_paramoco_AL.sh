#!/bin/bash

#SBATCH --job-name=082824_anat1-paramoco
#SBATCH --cpus-per-task=4
#SBATCH --time=01:05:00
#SBATCH --mem=64000
#SBATCH --array=0-49%50
#SBATCH --mail-user=albertl@princeton.edu
#SBATCH --mail-type=ALL
#SBATCH --output='slurm_log/%j-082824_anat1-motion_correction_%a.log'

module load anaconda3/2021.11
source /scratch/gpfs/albertl/atlas_dir/bin/activate 

MOV="/scratch/gpfs/albertl/rigE_data/082824_anat1/red_vols"
MOVGR="/scratch/gpfs/albertl/rigE_data/082824_anat1/green_vols"
OUTDIR="/scratch/gpfs/albertl/rigE_data/082824_anat1"

REF="/scratch/gpfs/albertl/rigE_data/082824_anat1/local_mean_red_0.nii"

TRANSFORM=1 # 0 = Affine (<15minutes), 1 = SyN (<45minutes) (the array number corresponds to the number of volumes--in the future avoid hardcoding this)

srun python paramoco.py --mov $MOV --mov_gr $MOVGR --ref $REF --transform $TRANSFORM --ARRAYNUMBER $SLURM_ARRAY_TASK_ID --outdir $OUTDIR
