#!/bin/bash

#SBATCH --job-name=D-FLASH-WIKI
#SBATCH --nodes=6
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=01:00:00
#SBATCH --mail-user=ncm5@rice.edu
#SBATCH --mail-type=all
#SBATCH --constraint="cascadelake&opath"

srun ./runme
