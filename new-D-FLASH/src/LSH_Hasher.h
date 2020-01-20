#ifndef _LSH_HASHER
#define _LSH_HASHER

#include <math.h>
#include <mpi.h>
#include <omp.h>
#include <random>

#define INT_MAX -1

#define HASH_OUTPUT_INDEX(num_tables, vector_indx, table)                                          \
    (unsigned int)(vector_indx * num_tables + table)

class LSH_Hasher {
  private:
    unsigned int _K;          // Number of Hashes Per Table
    unsigned int _L;          // Number of Tables
    unsigned int _num_hashes; // Total Number of Hashes
    unsigned int _log_num_hashes;

    unsigned int _range_base_2; // Log base 2 of the number of rows per table

    int _world_size; // Number of Nodes in Cluster
    int _my_rank;    // Rank of Individual Node

    int *_hash_function_params; // K * L params

    int _hash_seed; // Seed for begining of hash

    int _rand_seed1, _rand_seed2;

    void densified_min_hash(unsigned int *hashes, unsigned int *vector, unsigned int len_vector);

    unsigned int random_double_hash(unsigned int binid, int count);

  public:
    LSH_Hasher(unsigned int K, unsigned int L, unsigned int range_base_2, int world_size,
               int _my_rank);

    void hash(unsigned int num_vectors, unsigned int *vector_indices, unsigned int *vector_markers,
              unsigned int *hashes);

    ~LSH_Hasher();
};

#endif