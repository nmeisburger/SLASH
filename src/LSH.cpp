#include "LSH.h"

void LSH::add(unsigned int numInputEntries, unsigned int *dataIdx, float *dataVal,
              unsigned int *dataMarker, unsigned int dataOffset) {

    if ((unsigned)numInputEntries > _maxSamples) {
        printf("[LSHReservoirSampler::add] Input length %d is too large! \n", numInputEntries);
        return;
    }

    unsigned long long int storeLogSize = _numTables * numInputEntries;

    unsigned int *allprobsHash = new unsigned int[storeLogSize];
    unsigned int *allprobsIdx = new unsigned int[storeLogSize];

    _hashFamily->getHashes(allprobsHash, allprobsIdx, dataIdx, dataMarker, numInputEntries);

    unsigned int *storelog = new unsigned int[storeLogSize * 4]();

    reservoirSampling(allprobsHash, allprobsIdx, storelog, numInputEntries);
    addTable(storelog, numInputEntries, dataOffset);

    delete[] storelog;
    delete[] allprobsHash;
    delete[] allprobsIdx;

    _sequentialIDCounter_kernel += numInputEntries;
}

void LSH::extractReservoirs(unsigned int numQueryEntries, unsigned int segmentSize,
                            unsigned int *queue, unsigned int *hashIndices) {

    unsigned int hashIdx, allocIdx;
#pragma omp parallel for private(hashIdx, allocIdx)
    for (size_t tb = 0; tb < _numTables; tb++) {
        for (size_t queryIdx = 0; queryIdx < numQueryEntries; queryIdx++) {
            for (size_t elemIdx = 0; elemIdx < _reservoirSize; elemIdx++) {
                hashIdx = hashIndices[allprobsHashIdx(numQueryEntries, tb, queryIdx)];
                allocIdx = _tablePointers[tablePointersIdx(_numReservoirs, hashIdx, tb, _sechash_a,
                                                           _sechash_b)];
                if (allocIdx != TABLENULL) {
                    queue[queueElemIdx(segmentSize, tb, queryIdx, elemIdx)] =
                        _tableMem[tableMemResIdx(tb, allocIdx, _numReservoirs) + elemIdx];
                }
            }
        }
    }
}

void LSH::reservoirSampling(unsigned int *allprobsHash, unsigned int *allprobsIdx,
                            unsigned int *storelog, unsigned int numProbePerTb) {

    unsigned int counter, allocIdx, reservoirRandNum, TB, hashIdx, inputIdx, ct, reservoir_full,
        location;

#pragma omp parallel for private(TB, hashIdx, inputIdx, ct, allocIdx, counter, reservoir_full,     \
                                 reservoirRandNum, location)
    for (size_t probeIdx = 0; probeIdx < numProbePerTb; probeIdx++) {
        for (size_t tb = 0; tb < _numTables; tb++) {

            TB = numProbePerTb * tb;

            hashIdx = allprobsHash[allprobsHashSimpleIdx(numProbePerTb, tb, probeIdx)];
            inputIdx = allprobsIdx[allprobsHashSimpleIdx(numProbePerTb, tb, probeIdx)];
            ct = 0;

            /* Allocate the reservoir if non-existent. */
            omp_set_lock(_tablePointersLock +
                         tablePointersIdx(_numReservoirs, hashIdx, tb, _sechash_a, _sechash_b));
            allocIdx = _tablePointers[tablePointersIdx(_numReservoirs, hashIdx, tb, _sechash_a,
                                                       _sechash_b)];
            if (allocIdx == TABLENULL) {
                allocIdx = _tableMemAllocator[tableMemAllocatorIdx(tb)];
                _tableMemAllocator[tableMemAllocatorIdx(tb)]++;
                _tablePointers[tablePointersIdx(_numReservoirs, hashIdx, tb, _sechash_a,
                                                _sechash_b)] = allocIdx;
            }
            omp_unset_lock(_tablePointersLock +
                           tablePointersIdx(_numReservoirs, hashIdx, tb, _sechash_a, _sechash_b));

            // ATOMIC: Obtain the counter, and increment the counter. (Counter initialized to 0
            // automatically). Counter counts from 0 to currentCount-1.
            omp_set_lock(_tableCountersLock + tableCountersLockIdx(tb, allocIdx, _numReservoirs));

            counter = _tableMem[tableMemCtIdx(tb, allocIdx,
                                              _numReservoirs)]; // Potentially overflowable.
            _tableMem[tableMemCtIdx(tb, allocIdx, _numReservoirs)]++;
            omp_unset_lock(_tableCountersLock + tableCountersLockIdx(tb, allocIdx, _numReservoirs));

            // The counter here is the old counter. Current count is already counter + 1.
            // If current count is larger than _reservoirSize, current item needs to be sampled.
            // reservoir_full = (counter + 1) > _reservoirSize;

            reservoirRandNum = _global_rand[std::min((unsigned int)(_maxReservoirRand - 1),
                                                     counter)]; // Overflow prevention.

            if ((counter + 1) > _reservoirSize) { // Reservoir full.
                location = reservoirRandNum;
            } else {
                location = counter;
            }

            // location = reservoir_full * (reservoirRandNum)+(1 - reservoir_full) * counter;

            storelog[storelogIdIdx(numProbePerTb, probeIdx, tb)] = inputIdx;
            storelog[storelogCounterIdx(numProbePerTb, probeIdx, tb)] = counter;
            storelog[storelogLocationIdx(numProbePerTb, probeIdx, tb)] = location;
            storelog[storelogHashIdxIdx(numProbePerTb, probeIdx, tb)] = hashIdx;
        }
    }
}

void LSH::addTable(unsigned int *storelog, unsigned int numProbePerTb, unsigned int dataOffset) {

    unsigned int id, hashIdx, allocIdx;
    unsigned locCapped;
    //#pragma omp parallel for private(allocIdx, id, hashIdx, locCapped)
    for (size_t probeIdx = 0; probeIdx < numProbePerTb; probeIdx++) {
        for (size_t tb = 0; tb < _numTables; tb++) {

            id = storelog[storelogIdIdx(numProbePerTb, probeIdx, tb)];
            hashIdx = storelog[storelogHashIdxIdx(numProbePerTb, probeIdx, tb)];
            allocIdx = _tablePointers[tablePointersIdx(_numReservoirs, hashIdx, tb, _sechash_a,
                                                       _sechash_b)];
            // If item_i spills out of the reservoir, it is capped to the dummy location at
            // _reservoirSize.
            locCapped = storelog[storelogLocationIdx(numProbePerTb, probeIdx, tb)];

            if (locCapped < _reservoirSize) {
                _tableMem[tableMemResIdx(tb, allocIdx, _numReservoirs) + locCapped] =
                    id + _sequentialIDCounter_kernel + dataOffset;
            }
        }
    }
}

void LSH::getQueryHash(unsigned int queryPartitionSize, unsigned int numQueryPartitionHashes,
                       unsigned int *queryPartitionIndices, float *queryPartitionVals,
                       unsigned int *queryPartitionMarkers, unsigned int *queryHashes) {

    unsigned int *allprobsIdx = new unsigned int[numQueryPartitionHashes];

    _hashFamily->getHashes(queryHashes, allprobsIdx, queryPartitionIndices, queryPartitionMarkers,
                           queryPartitionSize);

    delete[] allprobsIdx;
}

void LSH::resetSequentialKernalID() { _sequentialIDCounter_kernel = 0; }

void LSH::checkTableMemLoad() {
    unsigned int maxx = 0;
    unsigned int minn = _numReservoirs;
    unsigned int tt = 0;
    for (unsigned int i = 0; i < _numTables; i++) {
        if (_tableMemAllocator[i] < minn) {
            minn = _tableMemAllocator[i];
        }
        if (_tableMemAllocator[i] > maxx) {
            maxx = _tableMemAllocator[i];
        }
        tt += _tableMemAllocator[i];
    }

    printf("Node %d Table Mem Usage ranges from %f to %f, average %f\n", _myRank,
           ((float)minn) / (float)_numReservoirs, ((float)maxx) / (float)_numReservoirs,
           ((float)tt) / (float)(_numTables * _numReservoirs));
}

void LSH::showParams() {
    printf("\n");
    printf("<<< LSHR Parameters >>>\n");
    std::cout << "_rangePow " << _rangePow << "\n";
    std::cout << "_numTables " << _numTables << "\n";
    std::cout << "_reservoirSize " << _reservoirSize << "\n";

    std::cout << "_dimension " << _dimension << "\n";
    std::cout << "_maxSamples " << _maxSamples << "\n";
    std::cout << "_numReservoirs " << _numReservoirs << "\n";
    std::cout << "_maxReservoirRand " << _maxReservoirRand << "\n";
    std::cout << "_tableMemMax " << _tableMemMax << "\n";
    std::cout << "_tableMemReservoirMax " << _tableMemReservoirMax << "\n";
    std::cout << "_tablePointerMax " << _tablePointerMax << "\n";
    printf("\n");
}

void LSH::tableContents() {
    unsigned int hashBucketLocation;
    for (int t = 0; t < _numTables; t++) {
        printf("\nNode %d - Table %d\n", _myRank, t);
        for (int b = 0; b < std::min(_numReservoirs, (unsigned)256); b++) {
            hashBucketLocation =
                _tablePointers[tablePointersIdx(_numReservoirs, b, t, _sechash_a, _sechash_b)];
            if (hashBucketLocation == TABLE_NULL) {
                continue;
            }
            printf("[%d]: ", b);
            for (int i = 0; i < _reservoirSize; i++) {
                if (_tableMem[tableMemResIdx(t, hashBucketLocation, _numReservoirs) + i] == 0) {
                    break;
                }
                printf("%d ", _tableMem[tableMemResIdx(t, hashBucketLocation, _numReservoirs) + i]);
            }
            printf("\n");
        }
        printf("\n");
    }
}
