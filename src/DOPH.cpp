#include "DOPH.h"

void DOPH::getHashes(unsigned int *hashIndices, unsigned int *dataIds, unsigned int *dataIdx,
                     unsigned int *dataMarker, size_t numInputEntries, unsigned int idOffset) {

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

            hashIndices[inputIdx * _L + tb] = index;
            if (dataIds != NULL) {
                dataIds[inputIdx] = inputIdx + idOffset;
            }
        }
        delete[] hashes;
    }
}

void DOPH::showDOPHConfig() {
    printf("Random Seed 1: %d\n", _randHash[0]);
    printf("Random Seed 2: %d\n", _randHash[1]);
    printf("Hash Seed: %d\n", _randa);

    for (int i = 0; i < _K * _L; i++) {
        printf("Hash Param %d: %d\n", i, _rand1[i]);
    }
}

unsigned int DOPH::getRandDoubleHash(int binid, int count) {
    unsigned int tohash = ((binid + 1) << 10) + count;
    return ((unsigned int)_randHash[0] * tohash << 3) >>
           (32 - _lognumhash); // _lognumhash needs to be ceiled.
}

void DOPH::optimalMinHash(unsigned int *hashArray, unsigned int *nonZeros,
                          unsigned int sizenonzeros) {
    /* This function computes the minhash and perform densification. */
    unsigned int *hashes = new unsigned int[_numhashes];

    unsigned int range = 1 << _rangePow;
    // binsize is the number of times the range is larger than the total number of hashes we need.
    unsigned int binsize = ceil(range / _numhashes);

    for (size_t i = 0; i < _numhashes; i++) {
        hashes[i] = UINT32_MAX;
    }

    for (size_t i = 0; i < sizenonzeros; i++) {
        unsigned int h = nonZeros[i];
        h *= _randa;
        h ^= h >> 13;
        h *= 0x85ebca6b;
        unsigned int curhash =
            ((unsigned int)(((unsigned int)h * nonZeros[i]) << 5) >> (32 - _rangePow));
        unsigned int binid =
            std::min((unsigned int)floor(curhash / binsize), (unsigned int)(_numhashes - 1));
        if (hashes[binid] > curhash)
            hashes[binid] = curhash;
    }
    /* Densification of the hash. */
    for (unsigned int i = 0; i < _numhashes; i++) {
        unsigned int next = hashes[i];
        if (next != UINT32_MAX) {
            hashArray[i] = hashes[i];
            continue;
        }
        unsigned int count = 0;
        while (next == UINT32_MAX) {
            count++;
            unsigned int index = std::min((unsigned)getRandDoubleHash(i, count), _numhashes);
            next = hashes[index];
            if (count > 100) { // Densification failure.
                printf("Densification Failure.\n");
                exit(1);
                break;
            }
        }
        hashArray[i] = next;
    }
    delete[] hashes;
}