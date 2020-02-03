#!/bin/bash

#SBATCH --job-name=WikiExtraction
#SBATCH --nodes=1 --exclusive
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --time=06:00:00
#SBATCH --mail-user=ncm5@rice.edu
#SBATCH --mail-type=all
#SBATCH --constraint=opath

srun python WikiExtractor.py --json enwiki-latest-pages-articles.xml.bz2 -o extract