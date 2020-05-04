#ifndef INDEXING_H
#define INDEXING_H

#define TABLENULL -1

#define HASH(x, M, a, b) (unsigned)((unsigned)(a * x + b) >> (32 - M))
#define BINHASH(x, a, b) (unsigned)((unsigned)(a * x + b) >> 31)

/* Datastructure Indexing. */

// _tableCountersLock:
#define tableCountersLockIdx(tb, allocIdx, aggNumReservoirs)                                       \
    (unsigned long long)(tb * aggNumReservoirs + allocIdx)
// _tableMem: Start of the data section of a reservoir.
#define tableMemResIdx(tb, allocIdx, aggNumReservoirs)                                             \
    (unsigned long long)(tb * aggNumReservoirs * (_reservoirSize + 1) +                            \
                         allocIdx * (_reservoirSize + 1) + 1)
// _tableMem: The counter section of a reservoir.
#define tableMemCtIdx(tb, allocIdx, aggNumReservoirs)                                              \
    (unsigned long long)(tb * aggNumReservoirs * (_reservoirSize + 1) +                            \
                         allocIdx * (_reservoirSize + 1))
// _tablePointers & _tablePointersLock: location of a pointer.

#define tablePointersIdx(numReservoirsHashed, hashIdx, tb)                                         \
    (unsigned long long)(tb * numReservoirsHashed + hashIdx)
// _tableMemAllocator: counter for allocating _tableMem, one per table.
#define tableMemAllocatorIdx(tb) (unsigned)(tb)

// queue: a particular element in the aggregated queue.
#define queueElemIdx(segmentSizePow2, tb, queryIdx, elemIdx)                                       \
    (unsigned)(queryIdx * segmentSizePow2 + tb * _reservoirSize + _reservoirSize +                 \
               elemIdx) // The start position of the segment for a
                        // particular query giventhe entire segment.
// topk queue: the start location of that of a query.
#define topkIdx(topk, queryIdx) (unsigned)(queryIdx * topk) // Indexing in the topk queue.
// allprobsHash: the hashIdx of an input-probe, i is its index in numProbePerTb.
#define allprobsHashSimpleIdx(numProbePerTb, tb, i) (unsigned)(numProbePerTb * tb + i)
// allprobsHash: the hashIdx of an input-probe.
#define allprobsHashIdx(numInputEntries, tb, inputIdx) (unsigned)(numInputEntries * tb + inputIdx)
// storelog: the id of an input.
#define storelogIdIdx(numProbsPerTb, probeIdx, tb) (unsigned)(numProbsPerTb * tb * 4 + 4 * probeIdx)
// storelog: the assigned counter of an input.
#define storelogCounterIdx(numProbsPerTb, probeIdx, tb)                                            \
    (unsigned)(numProbsPerTb * tb * 4 + 4 * probeIdx + 1)
// storelog: the location of storage (in-reservoir) of an input.
#define storelogLocationIdx(numProbsPerTb, probeIdx, tb)                                           \
    (unsigned)(numProbsPerTb * tb * 4 + 4 * probeIdx + 2)
// storelog: the hashIdx of an input.
#define storelogHashIdxIdx(numProbsPerTb, probeIdx, tb)                                            \
    (unsigned)(numProbsPerTb * tb * 4 + 4 * probeIdx + 3)

#endif
