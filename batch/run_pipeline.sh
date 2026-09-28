#!/bin/bash

#SBATCH --job-name=renewable_india
#SBATCH --output=%x.o%j
#SBATCH --ntasks=1
#SBATCH --partition=mem
#SBATCH --cpus-per-task=16
#SBATCH --mem=500G

# Full pipeline (bias correction -> hourly downscaling -> CF -> state series).
# Configuration lives in code/main.py (CONFIG). Run a subset with e.g.
#   sbatch run_pipeline.sh --steps cf cf_states

module purge
module load anaconda3/2023.09-0/none-none
source activate /gpfs/workdir/shared/juicce/envs/xenv

python ../code/main.py "$@"
