#include "LSH.h"

LSH::LSH(unsigned int num_tables, unsigned int reservoir_size, unsigned int range_pow, int my_rank,
         int world_size) {
    this->L = num_tables;
    this->reservoir_size = reservoir_size;
    this->range_pow = range_pow;
    this->range = 1 << range_pow;

    this->my_rank = my_rank;
    this->world_size = world_size;

    reservoirs = new Reservoir **[L];
    for (int i = 0; i < L; i++) {
        reservoirs[i] = new Reservoir *[range]();
        for (size_t r = 0; r < range; r++) {
            reservoirs[i][r] = new Reservoir(reservoir_size);
        }
    }
}

void LSH::insert(unsigned int num_items, unsigned int *hashes, unsigned int *items) {
#pragma omp parallel for default(none) shared(num_items, hashes, items)
    for (size_t n = 0; n < num_items; n++) {
        for (size_t table = 0; table < L; table++) {
            reservoirs[table][hashes[n * L + table]]->add(items[n]);
        }
    }
}

void LSH::insert(unsigned int *hashes, unsigned int item) {
    for (size_t table = 0; table < L; table++) {
        reservoirs[table][hashes[table]]->add(item);
    }
}

void LSH::retrieve(unsigned int num_query, unsigned int *hashes, unsigned int *results_buffer) {

#pragma omp parallel for default(none) shared(num_query, hashes, results_buffer)
    for (size_t query = 0; query < num_query; query++) {
        for (size_t table = 0; table < L; table++) {
            size_t loc = query * L + table;
            reservoirs[table][hashes[loc]]->retrieve(results_buffer + loc * reservoir_size);
        }
    }
}

void LSH::top_k(unsigned int num_query, unsigned int top_k, unsigned int *hashes,
                unsigned int *selection) {

    unsigned int *extracted_reservoirs = new unsigned int[num_query * L * reservoir_size];

    this->retrieve(num_query, hashes, extracted_reservoirs);

    unsigned int block = L * reservoir_size;
    for (size_t query = 0; query < num_query; query++) {
        unsigned int *start = extracted_reservoirs + query * block;
        std::sort(start, start + block);
        std::vector<std::pair<unsigned int, unsigned int>> counts;
        unsigned int count = 0;
        unsigned int last = *start;
        for (size_t i = 0; i < block; i++) {
            if (last == start[i]) {
                count++;
            } else {
                if (last != EMPTY) {
                    counts.push_back(std::make_pair(last, count));
                }
                count = 1;
                last = start[i];
            }
        }
        if (last != EMPTY) {
            counts.push_back(std::make_pair(last, count));
        }

        std::sort(counts.begin(), counts.end(),
                  [&counts](std::pair<int, int> a, std::pair<int, int> b) {
                      return a.second > b.second;
                  });

        size_t k;
        for (k = 0; k < std::min(top_k, (unsigned int)counts.size()); k++) {
            selection[query * top_k + k] = counts[k].first;
        }
        for (; k < top_k; k++) {
            selection[query * top_k + k] = EMPTY;
        }
    }

    delete[] extracted_reservoirs;
}

void LSH::reset() {
    for (size_t t = 0; t < L; t++) {
        for (size_t r = 0; r < range; r++) {
            reservoirs[t][r]->reset();
        }
    }
}

void LSH::view() {
    for (size_t t = 0; t < L; t++) {
        printf("LSH Table %lu\n", t);
        for (size_t r = 0; r < range; r++) {
            reservoirs[t][r]->view();
        }
        printf("\n");
    }
}

void LSH::add_random_items(unsigned int num_items, bool verbose) {

    unsigned int *items = new unsigned int[num_items];
    unsigned int *hashes = new unsigned int[num_items * L];

    for (size_t i = 0; i < num_items; i++) {
        items[i] = i;
        if (verbose)
            printf("Item: %lu -> { ", i);
        for (size_t h = 0; h < L; h++) {
            hashes[i * L + h] = rand() % range;
            if (verbose)
                printf("%u ", hashes[i * L + h]);
        }
        if (verbose)
            printf("}\n");
    }

    insert(num_items, items, hashes);

    delete[] items;
    delete[] hashes;
}

LSH::~LSH() {
    for (size_t t = 0; t < L; t++) {
        for (size_t r = 0; r < range; r++) {
            delete reservoirs[t][r];
        }
        delete[] reservoirs[t];
    }
    delete[] reservoirs;
}