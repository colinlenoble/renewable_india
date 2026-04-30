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
python ../code/downscale_hourly.py \
    --era5-grib  ../data/raw/era5/era5_india_*.grib \
    --gcm-grid   ../data/proc/cmip6_bc/tas_CanESM5_historical_bc.nc \
    --out-library ../data/proc/era5/diurnal_library_CanESM5.nc \
    --env-dir /gpfs/workdir/shared/juicce/envs/xenv 
# 2 – Apply (per SSP):
python ../code/downscale_hourly.py apply \
    --library  ../data/proc/era5/diurnal_library_CanESM5.nc \
    --bc-dir   ../data/proc/cmip6_bc \
    --gcm      CanESM5 --run r10i1p1f1 \
    --ssps     ssp245 ssp585 \
    --out-dir  ../data/proc/cmip6_hourly
"""

# python ../code/compute_cf.py \
#     --bc-dir      ../data/proc/cmip6_bc \
#     --hourly-dir  ../data/proc/cmip6_hourly \
#     --cmip-dir    ../data/raw/CanESM5 \
#     --shapefile   ../INDIA_STATES.geojson \
#     --out-dir     ../data/proc/cmip6_bc \
#     --gcm         CanESM5 \
#     --run         r10i1p1f1 \
#     --ssps        ssp245 ssp585 \
#     --train-start 1980-01-01 \
#     --train-end   2010-12-31 \
#     --region-col  STNAME_SH