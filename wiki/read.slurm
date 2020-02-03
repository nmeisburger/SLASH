#!/bin/bash

#SBATCH --job-name=WikiRead
#SBATCH --nodes=1 --exclusive
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:00:00
#SBATCH --mail-user=ncm5@rice.edu
#SBATCH --mail-type=all
#SBATCH --constraint=opath

srun python read.py