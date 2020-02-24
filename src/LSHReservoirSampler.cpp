#include "LSHReservoirSampler.h"

void LSHReservoirSampler::add(unsigned int numInputEntries, unsigned int *dataIdx, float *dataVal,
                              unsigned int *dataMarker, unsigned int dataOffset) {

    if ((unsigned)numInputEntries > _maxSamples) {
        printf("[LSHReservoirSampler::add] Input length %d is too large! \n", numInputEntries);
        return;
    }

    unsigned int *allprobsHash = new unsigned int[_numTables * numInputEntries];
    unsigned int *allprobsIdx = new unsigned int[_numTables * numInputEntries];

    _hashFamily->getHashes(allprobsHash, allprobsIdx, dataIdx, dataMarker, numInputEntries);

    unsigned int *storelog = new unsigned int[_numTables * 4 * numInputEntries]();

    reservoirSampling(allprobsHash, allprobsIdx, storelog, numInputEntries);
    addTable(storelog, numInputEntries, dataOffset);

    delete[] storelog;
    delete[] allprobsHash;
    delete[] allprobsIdx;

    _sequentialIDCounter_kernel += numInputEntries;
}

void LSHReservoirSampler::extractReservoirs(unsigned int numQueryEntries, unsigned int segmentSize,
                                            unsigned int *queue, unsigned int *hashIndices) {

    unsigned int hashIdx, allocIdx;
#pragma omp parallel for private(hashIdx, allocIdx)
    for (size_t tb = 0; tb < _numTables; tb++) {
        for (size_t queryIdx = 0; queryIdx < numQueryEntries; queryIdx++) {
            for (size_t elemIdx = 0; elemIdx < _reservoirSize; elemIdx++) {
                hashIdx = hashIndices[allprobsHashIdx(numQueryEntries, tb, queryIdx)];
                allocIdx = _tablePointers[tablePointersIdx(_numReservoirsHashed, hashIdx, tb,
                                                           _sechash_a, _sechash_b)];
                if (allocIdx != TABLENULL) {
                    queue[queueElemIdx(segmentSize, tb, queryIdx, elemIdx)] =
                        _tableMem[tableMemResIdx(tb, allocIdx, _aggNumReservoirs) + elemIdx];
                }
            }
        }
    }
}

void LSHReservoirSampler::getQueryHash(unsigned int queryPartitionSize,
                                       unsigned int numQueryPartitionHashes,
                                       unsigned int *queryPartitionIndices,
                                       float *queryPartitionVals,
                                       unsigned int *queryPartitionMarkers,
                                       unsigned int *queryHashes) {

    unsigned int *allprobsIdx = new unsigned int[numQueryPartitionHashes];

    _hashFamily->getHashes(queryHashes, allprobsIdx, queryPartitionIndices, queryPartitionMarkers,
                           queryPartitionSize);

    delete[] allprobsIdx;
}

void LSHReservoirSampler::resetSequentialKernalID() { _sequentialIDCounter_kernel = 0; }