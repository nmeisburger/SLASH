#ifndef _LSH_RESERVOIR
#define _LSH_RESERVOIR

#include <algorithm>
#include <cmath>
#include <mpi.h>
#include <omp.h>
#include <random>

#include "CMS.h"
#include "LSH_Hasher.h"
#include "dataset.h"

#define RESERVOIR_INDEX(table, num_reservoirs, reservoir)                                          \
    (unsigned int)(table * num_reservoirs + reservoir)

#define STORE_LOG_INDEX(table, vector, num_tables) (unsigned int)(vector * num_tables + table)

struct Location {
    unsigned int vector;
    unsigned int reservoir;
    unsigned int reservoir_location;
};

class LSH_Reservoir {
  private:
    unsigned int _L;
    unsigned int _K;
    unsigned int _reservoir_size;
    unsigned int _range_base_2;
    unsigned int _num_reservoirs;

    unsigned int _already_added;

    int _my_rank;
    int _world_size;
    unsigned int _total_vectors_added;
    unsigned int _initial_offset;

    unsigned int **_reservoirs;
    omp_lock_t *_reservoir_locks;

    unsigned int *_reservoir_counters;
    omp_lock_t *_reservoir_counter_locks;

    LSH_Hasher *_hash_family;
    CMS *_top_k_sketch;

    void reservoir_sampling(Location *store_log, unsigned int *hashes, unsigned int num_vectors,
                            unsigned int offset);

    void insert(Location *store_log, unsigned int num_vectors);

  public:
    LSH_Reservoir(unsigned int L, unsigned int K, unsigned int range_base_2,
                  unsigned int reservoir_size, LSH_Hasher *hash_family, CMS *sketch, int my_rank,
                  int world_size, unsigned int initial_offset);

    void add(unsigned int num_vectors, unsigned int *vector_markers, unsigned int *vector_indices,
             unsigned int offset);

    void add_dist(std::string filename, unsigned int read_offset, unsigned int num_vectors,
                  unsigned int dimension);

    void query_dist(std::string filename, unsigned int read_offset, unsigned int num_vectors,
                    unsigned int dimension, int top_k, unsigned int *outputs);

    void extract(unsigned int num_vectors, unsigned int *hashes, unsigned int *result);

    ~LSH_Reservoir();
};

#endif