#!/bin/bash

#SBATCH --job-name=txt_make_datasets
#SBATCH --output=%x.o%j 
#SBATCH --ntasks=4
#SBATCH --partition=mem                           
#SBATCH --cpus-per-task=4
#SBATCH --mem=500G

# Load necessary modules
module purge
module load anaconda3/2023.09-0/none-none

# Activate anaconda environment
source activate /gpfs/workdir/shared/juicce/envs/xenv

# Run python script
python ../code/bias_correction_qdm.py \
    --era5-dir /gpfs/workdir/shared/juicce/RE_Colin/India/renewable_india/data/raw/era5_daily/ \
    --cmip-dir /gpfs/workdir/shared/juicce/RE_Colin/India/renewable_india/data/raw/CanESM5/ \
    --env-dir /gpfs/workdir/shared/juicce/envs/xenv \
    --out-dir /gpfs/workdir/shared/juicce/RE_Colin/India/renewable_india/data/proc/CanESM5/ \
    --nquantile 25