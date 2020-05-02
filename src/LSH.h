#pragma once

#include "Reservoir.h"
#include <iostream>
#include <mpi.h>
#include <omp.h>
#include <vector>

class LSH {
  private:
    unsigned int L;
    unsigned int reservoir_size;
    unsigned int range_pow;
    unsigned int range;
    int my_rank;
    int world_size;
    Reservoir ***reservoirs;

  public:
    LSH(unsigned int num_tables, unsigned int reservoir_size, unsigned int range_pow, int my_rank,
        int world_size);

    void insert(unsigned int num_items, unsigned int *hashes, unsigned int *items);

    void insert(unsigned int *hashes, unsigned int item);

    void retrieve(unsigned int num_query, unsigned int *hashes, unsigned int *results_buffer);

    void top_k(unsigned int num_query, unsigned int top_k, unsigned int *hashes,
               unsigned int *selection);

    void reset();

    void view();

    void add_random_items(unsigned int num_items, bool verbose);

    ~LSH();
};