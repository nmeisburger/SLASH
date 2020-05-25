#!/bin/bash

#SBATCH --job-name=criteo-eval
#SBATCH --nodes=20
#SBATCH -p scavenge
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=30G
#SBATCH --time=02:00:00
#SBATCH --mail-user=ncm5@rice.edu
#SBATCH --mail-type=all

source setup.sh 

mpicxx -o eval ../utils/evaluate.cpp

srun ./eval