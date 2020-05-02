#pragma once

#include <algorithm>
#include <iostream>
#include <omp.h>

#define EMPTY -1

class Reservoir {
  private:
    unsigned int size;
    unsigned int count;
    omp_lock_t *lock;
    unsigned int *reservoir;

  public:
    Reservoir(unsigned int size);

    void add(unsigned int item);

    void retrieve(unsigned int *buffer);

    unsigned int get_size();

    unsigned int get_count();

    void reset();

    void view();

    ~Reservoir();
};
