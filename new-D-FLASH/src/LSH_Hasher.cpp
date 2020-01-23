#include "LSH_Hasher.h"

LSH_Hasher::LSH_Hasher(unsigned int K, unsigned int L, unsigned int range_base_2, int world_size,
                       int my_rank) {
    _K = K;
    _L = L;
    _num_hashes = _K * _L;
    _log_num_hashes = log2(_num_hashes);
    _range_base_2 = range_base_2;
    _world_size = world_size;
    _my_rank = my_rank;

    srand(145297);

    _hash_function_params = new int[_num_hashes];

    if (_my_rank == 0) {
        for (int i = 0; i < _num_hashes; i++) {
            _hash_function_params[i] = rand();
            if (_hash_function_params[i] % 2 == 0) {
                _hash_function_params[i]++;
            }
        }

        _hash_seed = rand();
        if (_hash_seed % 2 == 0) {
            _hash_seed++;
        }

        _rand_seed1 = rand();
        if (_rand_seed1 % 2 == 0) {
            _rand_seed1++;
        }

        _rand_seed2 = rand();
        if (_rand_seed2 % 2 == 0) {
            _rand_seed2++;
        }
    }

    MPI_Bcast(_hash_function_params, _num_hashes, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&_hash_seed, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&_rand_seed1, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&_rand_seed2, 1, MPI_INT, 0, MPI_COMM_WORLD);

    printf("LSH_Hasher Created in Node %d\n", _my_rank);
}

unsigned int LSH_Hasher::random_double_hash(unsigned int binid, int count) {
    unsigned int tohash = ((binid + 1) << 10) + count;
    return ((unsigned int)_rand_seed1 * tohash << 3) >>
           (32 - _log_num_hashes); // _lognumhash needs to be ceiled.
}

void LSH_Hasher::densified_min_hash(unsigned int *output_hashes, unsigned int *vector,
                                    unsigned int len_vector) {
    /* This function computes the minhash and perform densification. */
    unsigned int range = 1 << _range_base_2;
    // binsize is the number of times the range is larger than the total
    // number of hashes we need.
    unsigned int binsize = ceil(range / _num_hashes);
    unsigned int temp_hashes[_num_hashes];

    for (unsigned int i = 0; i < _num_hashes; i++) {
        temp_hashes[i] = INT_MAX;
    }

    for (unsigned int i = 0; i < len_vector; i++) {
        unsigned int h = vector[i];
        h *= _hash_seed;
        h ^= h >> 13;
        h *= 0x85ebca6b;
        unsigned int curhash = ((h * vector[i]) << 5) >> (32 - _range_base_2);
        unsigned int binid = std::min((unsigned int)floor(curhash / binsize), (_num_hashes - 1));
        if (temp_hashes[binid] > curhash)
            temp_hashes[binid] = curhash;
    }
    /* Densification of the hash. */
    for (unsigned int i = 0; i < _num_hashes; i++) {
        unsigned int next = temp_hashes[i];
        if (next == INT_MAX) {
            unsigned int count = 0;
            while (next == INT_MAX) {
                count++;
                unsigned int index = std::min(random_double_hash(i, count), _num_hashes);
                next = temp_hashes[index];
                if (count > 100){  // Densification failure.
                    next = rand() >> (32 - _range_base_2);
                    break;
                }
            }
        }
        output_hashes[i] = next;
    }
}

void LSH_Hasher::hash(unsigned int num_vectors, unsigned int *vector_indices,
                      unsigned int *vector_markers, unsigned int *hashes) {
#pragma omp parallel for default(none) shared(num_vectors, vector_markers, vector_indices, hashes)
    for (unsigned int vector = 0; vector < num_vectors; vector++) {
        unsigned int *min_hashes = new unsigned int[_num_hashes];
        unsigned int size_non_zeros = vector_markers[vector + 1] - vector_markers[vector];

        densified_min_hash(min_hashes, vector_indices + vector_markers[vector], size_non_zeros);
        
        for (unsigned int table = 0; table < _L; table++) {
            unsigned int index = 0;
            for (unsigned int k = 0; k < _K; k++) {
                unsigned int h = min_hashes[_K * table + k];
                h *= _hash_function_params[_K * table + k];
                h ^= h >> 13;
                h ^= _hash_function_params[_K * table + k];
                index += h * min_hashes[_K * table + k];
            }
            index = (index << 2) >> (32 - _range_base_2);
            hashes[HASH_OUTPUT_INDEX(_L, vector, table)] = index;
        }
        delete[] min_hashes;
    }
}

void LSH_Hasher::showLSHConfig() {
    printf("Random Seed 1: %d\n", _rand_seed1);
    printf("Random Seed 2: %d\n", _rand_seed2);
    printf("Hash Seed: %d\n", _hash_seed);

    for (int i = 0; i < _K * _L; i++) {
        printf("Hash Param %d: %d\n", i, _hash_function_params[i]);
    }
}

LSH_Hasher::~LSH_Hasher() { delete[] _hash_function_params; }
