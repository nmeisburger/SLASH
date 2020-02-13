#include "LSH.h"

void LSH::getHashes(unsigned int *hashIndices, unsigned int *probeDataIdx, unsigned int *dataIdx,
                    unsigned int *dataMarker, size_t numInputEntries) {

#pragma omp parallel for
    for (size_t inputIdx = 0; inputIdx < numInputEntries; inputIdx++) {

        unsigned int *hashes = new unsigned int[_numhashes];
        unsigned int sizenonzeros = dataMarker[inputIdx + 1] - dataMarker[inputIdx];

        optimalMinHash(hashes, (unsigned int *)(dataIdx + dataMarker[inputIdx]), sizenonzeros);

        for (size_t tb = 0; tb < _L; tb++) {
            unsigned int index = 0;
            for (size_t k = 0; k < _K; k++) {
                unsigned int h = hashes[_K * tb + k];
                h *= _rand1[_K * tb + k];
                h ^= h >> 13;
                h ^= _rand1[_K * tb + k];
                index += h * hashes[_K * tb + k];
            }
            index = (index << 2) >> (32 - _rangePow);

            hashIndices[hashIndicesOutputIdx(_L, numInputEntries, inputIdx, tb)] = index;
            probeDataIdx[hashIndicesOutputIdx(_L, numInputEntries, inputIdx, tb)] = inputIdx;
        }
        delete[] hashes;
    }
}

void LSH::showLSHConfig() {
    printf("Random Seed 1: %d\n", _randHash[0]);
    printf("Random Seed 2: %d\n", _randHash[1]);
    printf("Hash Seed: %d\n", _randa);

    for (int i = 0; i < _K * _L; i++) {
        printf("Hash Param %d: %d\n", i, _rand1[i]);
    }
}
