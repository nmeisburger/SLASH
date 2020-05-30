# SLASH
Sketching based distributed locally sensitive hashing algorithm for similarity search with ultra high dimensionality datasets.

## Criteo Nearest Neighbor Instructions

### To get the list of nearest neighbors foreach query
1. Clone this repository on NOTS. 
2. Navigate to src directory `$ cd SLASH/src`.
3. Load MPI compiler and library `$ source scripts/setup.sh`.
4. Make sure that only `#define FULL_CRITEO` is uncommented at the top of `benchmarking.h`.
5. Compile program `$ make clean ; make`.
6. Navigate to slurm script `$ cd scripts`.
7. Change the email at the bottom of `criteo.slurm` to recieve notifactions when the jobs starts and completes.
8. Submit job request `$ sbatch criteo.slurm`.
9. The output will be in the file `Criteo-20`.

### To analyze these results
1. Navigate to the utils directory `$ cd ../utils`
2. Make sure that the TOPK_PATH definition at the top of the file is the path to the `Criteo-20` results file. 
3. Navigate back to the scripts director `$ cd ../scripts`.
4. Run `$ sbatch eval.slurm` to preform the evaluation. The output will be in a slurm.out file in the scripts directory.

## Instructions for General Use
1. Download and unzip either the `criteo_tb`, `webspam_wc_normalized_trigram.svm.bz2`, or `kdd12.bz2` from https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html. 
* Note that the run on criteo the system requires that the dataset is split into a new file for each MPI process. Each MPI process will read from the file given by the value of `BASEFILE` in `benchmarking.h` concatentated with its MPI process id. For example process 0 will read from "criteo_tb0" and process 4 would read from "criteo_tb4". 
* Partitioning the dataset in this way allows for more efficient parallel file IO and better maximizes the parallel file system which is important with a dataset of this size. 
* The code for preforming the partition can be found in `src/utils/split.cpp` and `src/utils/split.slurm`. The config at the top of `split.cpp` determines how many files to create, the base name of each file, the block size (in bytes) to read at a time, and the number of blocks to put in each new file. The program will make sure that each file starts at the begining of a line even if the offset places it somewhere in the middle.
2. In the `benchmarking.h` file:
* Uncomment which dataset you are using. For kdd12 and webspam also uncomment either `FILE_OUTPUT` or `EVAL_SIM` depending on if you want the program to output the results to a file, or evaluate the simarity and print the result. 
* Note that `EVAL_SIM` requires being able to load the dataset into RAM. Also make sure to set the value of `BASEFILE` to be the path to the downloaded dataset. 
* Note that there is a separate config section in `benchmarking.h` for different datasets, make sure to adjust the correct one.
* Each section of `benchmarking.h` sets various hyper parameters for a dataset (i.e. number of hashtables, number of input vectors etc.).
* Note also that for criteo the `NUM_DATA_VECTORS` value in `benchmarking.h` denotes the number of vectors per node in the cluster vs the total number accross all nodes for the other configs for webspam or kdd12.
3. Compile the program. On NOTS or a similar HPC cluster running `$ source src/setup.sh` from the root of the project to load the necessary MPI library and compiler. We compile and ran our system using OpenMPI/3.1.4 and the mpicxx compiler in GCC/8.3.0.
4. Run `$ make clean; make` to compile the program.
5. The *.slurm files contain the slurm scripts for running the system using the slurm job scheduler. You can edit the size of the cluster as well as some other properties here. Make sure to set the job name at the top and the email that slurm will send notifications to.
6. Run `$ sbatch myjob.slurm` for webspam or kdd12 or `$ sbatch criteo.slurm` for criteo.
