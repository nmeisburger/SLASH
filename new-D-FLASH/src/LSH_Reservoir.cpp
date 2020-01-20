#include "LSH_Reservoir.h"

LSH_Reservoir::LSH_Reservoir(unsigned int L, unsigned int K, unsigned int range_base_2,
                             unsigned int reservoir_size, LSH_Hasher *hash_family, CMS *sketch,
                             int my_rank, int world_size, unsigned int initial_offset) {
    _L = L;
    _K = K;
    _reservoir_size = reservoir_size;
    _range_base_2 = range_base_2;
    _num_reservoirs = 1 << _range_base_2;

    _already_added = 0;

    _my_rank = my_rank;
    _world_size = world_size;
    _total_vectors_added = 0;
    _initial_offset = initial_offset;

    _reservoirs = new unsigned int *[_num_reservoirs * _L];
    _reservoir_locks = new omp_lock_t[_num_reservoirs * _L];

    _reservoir_counters = new unsigned int[_num_reservoirs * _L]();
    _reservoir_counter_locks = new omp_lock_t[_num_reservoirs * _L];

    _hash_family = hash_family;
    _top_k_sketch = sketch;

    srand(13957130);
}

void LSH_Reservoir::reservoir_sampling(Location *store_log, unsigned int *hashes,
                                       unsigned int num_vectors, unsigned int offset) {
#pragma omp parallel for default(none) shared(store_log, hashes, num_vectors, offset)
    for (unsigned int vector = 0; vector < num_vectors; vector++) {
        for (unsigned int table = 0; table < _L; table++) {
            unsigned int hash_value = hashes[HASH_OUTPUT_INDEX(_L, vector, table)];

            omp_set_lock(_reservoir_counter_locks +
                         RESERVOIR_INDEX(table, _num_reservoirs, hash_value));
            unsigned int count =
                _reservoir_counters[RESERVOIR_INDEX(table, _num_reservoirs, hash_value)];

            if (count == 0) {
                _reservoirs[RESERVOIR_INDEX(table, _num_reservoirs, hash_value)] =
                    new unsigned int[_reservoir_size];
            }
            unsigned int location = count;
            if (count >= _reservoir_size) {
                location = rand() % (count + 1);
            }
            store_log[STORE_LOG_INDEX(table, vector, _L)].reservoir = hash_value;
            store_log[STORE_LOG_INDEX(table, vector, _L)].vector = vector + _already_added + offset;
            store_log[STORE_LOG_INDEX(table, vector, _L)].reservoir_location = location;
            _reservoir_counters[RESERVOIR_INDEX(table, _num_reservoirs, hash_value)]++;
            omp_unset_lock(_reservoir_counter_locks +
                           RESERVOIR_INDEX(table, _num_reservoirs, hash_value));
        }
    }
}

void LSH_Reservoir::insert(Location *store_log, unsigned int num_vectors) {
    unsigned int location, reservoir, current_vector;
#pragma omp parallel for default(none)                                                             \
    shared(store_log, num_vectors) private(location, reservoir, current_vector)
    for (unsigned int vector = 0; vector < num_vectors; vector++) {
        for (unsigned int table = 0; table < _L; table++) {
            location = store_log[STORE_LOG_INDEX(table, vector, _L)].reservoir_location;
            if (location < _reservoir_size) {
                current_vector = store_log[STORE_LOG_INDEX(table, vector, _L)].vector;
                reservoir = store_log[STORE_LOG_INDEX(table, vector, _L)].reservoir;
                omp_set_lock(_reservoir_locks + RESERVOIR_INDEX(table, _num_reservoirs, reservoir));
                _reservoirs[RESERVOIR_INDEX(table, _num_reservoirs, reservoir)][location] = vector;
                omp_unset_lock(_reservoir_locks +
                               RESERVOIR_INDEX(table, _num_reservoirs, reservoir));
            }
        }
    }
    _already_added += num_vectors;
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
#pragma omp parallel for shared(num_vectors, hashes, results)
    for (unsigned int vector = 0; vector < num_vectors; vector++) {
        for (unsigned int table = 0; table < _L; table++) {
            for (unsigned int item = 0; item < _reservoir_size; item++) {
                if (_reservoir_counters[HASH_OUTPUT_INDEX(_L, vector, table)] > 0) {
                    results[vector * _reservoir_size * _L + table * _reservoir_size + item] =
                        _reservoirs[HASH_OUTPUT_INDEX(_L, vector, table)][item];
                }
            }
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