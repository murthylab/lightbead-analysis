#!/bin/bash

#SBATCH --job-name=a1_r2_wpre-signal-to-supervoxel-ROI
#SBATCH --cpus-per-task=1
#SBATCH --time=4:05:00
#SBATCH --mem=32000
#SBATCH --mail-user=albertl@princeton.edu
#SBATCH --mail-type=ALL
#SBATCH --output='slurm_log/%j-a1_r2_wpre-signal-to-supervoxel-ROI.log'

module load anaconda3/2021.11
source /scratch/gpfs/albertl/atlas_dir/bin/activate

# if works smoothly, change to scratch dirs
MEMPATH="/scratch/gpfs/albertl/rigE_data/a1_r2_wpre/06212024_6f_a1_r2_moco_green.mmap"
NUMVOLS=659
OUTDIR="/scratch/gpfs/albertl/rigE_data/a1_r2_wpre/supervoxels"
N_CLUSTERS=1000  # 20, 200, 2000, 20000
FLYTRIAL="a1_r2_wpre"

srun python3 signal_to_supervoxel_roi.py --mempath $MEMPATH --numvols $NUMVOLS --outdir $OUTDIR --n_clusters $N_CLUSTERS --flytrial $FLYTRIAL # resolution hardcoded
