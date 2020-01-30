#include "LSH_Reservoir.h"

LSH_Reservoir::LSH_Reservoir(unsigned int L, unsigned int K, unsigned int range_base_2,
                             unsigned int reservoir_size, LSH_Hasher *hash_family, CMS *sketch,
                             int my_rank, int world_size) {
    _L = L;
    _K = K;
    _reservoir_size = reservoir_size;
    _range_base_2 = range_base_2;
    _num_reservoirs = 1 << _range_base_2;

    _already_added = 0;

    _my_rank = my_rank;
    _world_size = world_size;
    _total_vectors_added = 0;

    _reservoirs = new unsigned int *[_num_reservoirs * _L];
    _reservoir_locks = new omp_lock_t[_num_reservoirs * _L];

    _reservoir_counters = new unsigned int[_num_reservoirs * _L]();
    _reservoir_counter_locks = new omp_lock_t[_num_reservoirs * _L];

    _hash_family = hash_family;
    _top_k_sketch = sketch;

    srand(13957130);

    printf("LSH_Reservoir Created in Node %d\n", _my_rank);
}

void LSH_Reservoir::reservoir_sampling(Location *store_log, unsigned int *hashes,
                                       unsigned int num_vectors, unsigned int offset) {
#pragma omp parallel for default(none) shared(store_log, hashes, num_vectors, offset)
    for (size_t vector_indx = 0; vector_indx < num_vectors; vector_indx++) {
        for (size_t table = 0; table < _L; table++) {
            unsigned int hash_value = hashes[HASH_OUTPUT_INDEX(_L, vector_indx, table)];

            omp_set_lock(_reservoir_counter_locks +
                         RESERVOIR_INDEX(table, _num_reservoirs, hash_value));
            unsigned int count =
                _reservoir_counters[RESERVOIR_INDEX(table, _num_reservoirs, hash_value)];

            if (count == 0) {
                _reservoirs[RESERVOIR_INDEX(table, _num_reservoirs, hash_value)] =
                    new unsigned int[_reservoir_size]();
                for (size_t i = 0; i < _reservoir_size; i++) {
                    _reservoirs[RESERVOIR_INDEX(table, _num_reservoirs, hash_value)][i] = INT_MAX;
                }
            }
            unsigned int location = count;
            if (count >= _reservoir_size) {
                location = rand() % (count + 1);
            }
            store_log[STORE_LOG_INDEX(table, vector_indx, _L)].reservoir = hash_value;
            store_log[STORE_LOG_INDEX(table, vector_indx, _L)].vector_indx =
                vector_indx + _already_added + offset;
            store_log[STORE_LOG_INDEX(table, vector_indx, _L)].reservoir_location = location;
            _reservoir_counters[RESERVOIR_INDEX(table, _num_reservoirs, hash_value)]++;
            omp_unset_lock(_reservoir_counter_locks +
                           RESERVOIR_INDEX(table, _num_reservoirs, hash_value));
            // unsigned int xx = STORE_LOG_INDEX(table, vector_indx, _L);
            // printf("Store Log Check: Position %u vector %u\n", xx, store_log[xx].vector_indx);
        }
    }
}

void LSH_Reservoir::insert(Location *store_log, unsigned int num_vectors) {
    unsigned int location, reservoir, current_vector;
#pragma omp parallel for default(none)                                                             \
    shared(store_log, num_vectors) private(location, reservoir, current_vector)
    for (size_t vector_indx = 0; vector_indx < num_vectors; vector_indx++) {
        // printf("Node %d Vector Index Check %d\n", _my_rank,
        //    store_log[vector_indx * _L].vector_indx);
        for (size_t table = 0; table < _L; table++) {
            // unsigned int xx = STORE_LOG_INDEX(table, vector_indx, _L);
            // printf("Store Log Check: Position %u vector %u\n", xx, store_log[xx].vector_indx);
            location = store_log[STORE_LOG_INDEX(table, vector_indx, _L)].reservoir_location;
            if (location < _reservoir_size) {
                current_vector = store_log[STORE_LOG_INDEX(table, vector_indx, _L)].vector_indx;
                reservoir = store_log[STORE_LOG_INDEX(table, vector_indx, _L)].reservoir;
                omp_set_lock(_reservoir_locks + RESERVOIR_INDEX(table, _num_reservoirs, reservoir));
                _reservoirs[RESERVOIR_INDEX(table, _num_reservoirs, reservoir)][location] =
                    current_vector;
                omp_unset_lock(_reservoir_locks +
                               RESERVOIR_INDEX(table, _num_reservoirs, reservoir));
            }
        }
    }
}

void LSH_Reservoir::add(unsigned int num_vectors, unsigned int *vector_markers,
                        unsigned int *vector_indices, unsigned int offset) {
    unsigned int *hashes = new unsigned int[num_vectors * _L];
    _hash_family->hash(num_vectors, vector_indices, vector_markers, hashes);

    Location *store_log = new Location[num_vectors * _L];
    this->reservoir_sampling(store_log, hashes, num_vectors, offset);
    this->insert(store_log, num_vectors);
}

void LSH_Reservoir::extract(unsigned int num_vectors, unsigned int *hashes, unsigned int *results) {
    unsigned int hash_value;
#pragma omp parallel for shared(num_vectors, hashes, results) private(hash_value)
    for (size_t vector_indx = 0; vector_indx < num_vectors; vector_indx++) {
        for (size_t table = 0; table < _L; table++) {
            hash_value = hashes[HASH_OUTPUT_INDEX(_L, vector_indx, table)];
            if (_reservoir_counters[RESERVOIR_INDEX(table, _num_reservoirs, hash_value)] > 0) {
                for (size_t item = 0; item < _reservoir_size; item++) {
                    results[EXTRACTED_INDEX(_reservoir_size, _L, vector_indx, table, item)] =
                        _reservoirs[RESERVOIR_INDEX(table, _num_reservoirs, hash_value)][item];
                }
            }
        }
    }
}

void LSH_Reservoir::print() {
    for (size_t t = 0; t < _L; t++) {
        printf("\nNode %d - Table %lu\n", _my_rank, t);
        for (size_t h = 0; h < _num_reservoirs; h++) {
            if (_reservoir_counters[RESERVOIR_INDEX(t, _num_reservoirs, h)] > 0) {
                printf("[%lu]: ", h);
                for (size_t r = 0; r < _reservoir_size; r++) {
                    unsigned int val = _reservoirs[RESERVOIR_INDEX(t, _num_reservoirs, h)][r];
                    if (val > 0) {
                        printf("%d ", val);
                    }
                }
                printf("\n");
            }
        }
    }
}

void LSH_Reservoir::printCounts() {
    for (size_t t = 0; t < _L; t++) {
        printf("\nNode %d - Table %lu\n", _my_rank, t);
        for (size_t h = 0; h < _num_reservoirs; h++) {
            printf("[%lu]: %d\n", h, _reservoir_counters[RESERVOIR_INDEX(t, _num_reservoirs, h)]);
        }
    }
}

LSH_Reservoir::~LSH_Reservoir() {
    delete[] _reservoirs;
    delete[] _reservoir_locks;
    delete[] _reservoir_counters;
    delete[] _reservoir_counter_locks;

    delete _hash_family;
    printf("LSH Reservoir Deleted\n");
}