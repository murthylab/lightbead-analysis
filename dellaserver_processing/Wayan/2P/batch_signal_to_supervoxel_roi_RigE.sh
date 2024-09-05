#!/bin/bash

#SBATCH --job-name=06212024_6f_a1_r2-signal-to-supervoxel-ROI
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=15:05:00
#SBATCH --mem=32000
#SBATCH --mail-user=wg0346@princeton.edu
#SBATCH --mail-type=ALL
#SBATCH --output='slurm_log/%j-06212024_6f_a1_r2-signal-to-supervoxel-ROI-n1000.log'

module load anaconda3/2021.11
conda activate ants_env

# if works smoothly, change to scratch dirs
MEMPATH="/scratch/gpfs/wg0346/auditory_data/06212024_6f_a1_r2/06212024_6f_a1_r2_moco_green.mmap"
NUMVOLS=660
OUTDIR="/scratch/gpfs/wg0346/auditory_data/06212024_6f_a1_r2/supervoxels"
N_CLUSTERS=1000  # 20, 200, 2000, 20000
FLYTRIAL="06212024_6f_a1_r2"

python3 signal_to_supervoxel_roi_RigE.py --mempath $MEMPATH --numvols $NUMVOLS --outdir $OUTDIR --n_clusters $N_CLUSTERS --flytrial $FLYTRIAL
