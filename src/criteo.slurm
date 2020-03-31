#!/bin/bash

#SBATCH --job-name=criteo-testing
#SBATCH --nodes=20
##SBATCH -p scavenge
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=50G
#SBATCH --time=04:00:00
#SBATCH --mail-user=ncm5@rice.edu
#SBATCH --mail-type=all
##SBATCH --constraint="cascadelake&opath"

srun ./runme
