#!/bin/bash

#SBATCH --job-name=criteo_eval
#SBATCH --nodes=20
#SBATCH -p scavenge
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH --time=04:00:00
#SBATCH --mail-user=ncm5@rice.edu
#SBATCH --mail-type=all

srun ./check
