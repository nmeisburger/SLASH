#!/bin/bash

#SBATCH --job-name=backup-D-FLASH5-Billion
#SBATCH --nodes=5
##SBATCH -p scavenge
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=20:00:00
#SBATCH --mail-user=ncm5@rice.edu
#SBATCH --mail-type=all
##SBATCH --constraint="cascadelake&opath"

srun ./runme
