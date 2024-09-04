#!/bin/bash

#SBATCH --job-name=04032024_GCamp6f_a1_r4-signal-to-supervoxel-ROI
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=15:05:00
#SBATCH --mem=32000
#SBATCH --mail-user=wg0346@princeton.edu
#SBATCH --mail-type=ALL
#SBATCH --output='slurm_log/%j-04032024_GCamp6f_a1_r4-signal-to-supervoxel-ROI-n1000.log'

module load anaconda3/2021.11
source /scratch/gpfs/wg0346/venv
conda activate ants_env

# if works smoothly, change to scratch dirs
MEMPATH="/scratch/gpfs/wg0346/auditory_data/04032024_GCamp6f_a1_r4/04032024_GCamp6f_a1_r4_moco_green.mmap"
NUMVOLS=8454
OUTDIR="/scratch/gpfs/wg0346/auditory_data/04032024_GCamp6f_a1_r4/supervoxels"
N_CLUSTERS=1000  # 20, 200, 2000, 20000
FLYTRIAL="04032024_GCamp6f_a1_r4"

python3 signal_to_supervoxel_roi.py --mempath $MEMPATH --numvols $NUMVOLS --outdir $OUTDIR --n_clusters $N_CLUSTERS --flytrial $FLYTRIAL
