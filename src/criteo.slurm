#!/bin/bash

#SBATCH --job-name=criteo-100mil
#SBATCH --nodes=10
##SBATCH -p scavenge
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --mail-user=ncm5@rice.edu
#SBATCH --mail-type=all
##SBATCH --constraint="cascadelake&opath"

srun ./runme
