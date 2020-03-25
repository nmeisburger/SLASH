#!/bin/bash

#SBATCH --job-name=criteo_eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=50G
#SBATCH --time=10:00:00
#SBATCH --mail-user=ncm5@rice.edu
#SBATCH --mail-type=all

srun ./check
