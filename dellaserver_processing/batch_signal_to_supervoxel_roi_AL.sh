#!/bin/bash

#SBATCH --job-name=240306_run_3-signal-to-supervoxel-ROI
#SBATCH --cpus-per-task=4
#SBATCH --time=1:05:00
#SBATCH --mem=128000
#SBATCH --mail-user=albertl@princeton.edu
#SBATCH --mail-type=ALL
#SBATCH --output='slurm_log/%j-240306_run_3-signal-to-supervoxel-ROI.log'

module load anaconda3/2021.11
source /scratch/gpfs/albertl/atlas_dir/bin/activate

# if works smoothly, change to scratch dirs
MEMPATH="/scratch/gpfs/albertl/rigE_data/240306_run_3/240306_run_3_moco_green.mmap"
NUMVOLS=450
OUTDIR="/scratch/gpfs/albertl/rigE_data/240306_run_3/supervoxels"
N_CLUSTERS=100  # 20, 200, 2000, 20000
FLYTRIAL="240306_run_3"

srun python3 signal_to_supervoxel_roi.py --mempath $MEMPATH --numvols $NUMVOLS --outdir $OUTDIR --n_clusters $N_CLUSTERS --flytrial $FLYTRIAL # resolution hardcoded
