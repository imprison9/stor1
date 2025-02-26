#!/bin/bash

#SBATCH --job-name=feature
#SBATCH -p cpu
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -n 1
#SBATCH --output=slurm_logs/%j.out
#SBATCH --error=slurm_logs/%j.err
sh ./prepare_data.sh