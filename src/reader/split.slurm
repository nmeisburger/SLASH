#!/bin/bash

#SBATCH --job-name=partitioning
#SBATCH --nodes=1
#SBATCH -p scavenge
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=45G
#SBATCH --time=10:00:00
#SBATCH --mail-user=ncm5@rice.edu
#SBATCH --mail-type=all
##SBATCH --constraint="cascadelake&opath"

srun ./splitter