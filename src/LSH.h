#ifndef _LSH_H
#define _LSH_H

#include "DOPH.h"
#include "indexing.h"
#include "mathUtils.h"
#include <algorithm>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define TABLE_NULL -1

class LSH {
  private:
    int _myRank, _worldSize;

    DOPH *_hashFamily;
    unsigned int _rangePow, _numTables, _reservoirSize, _dimension, _maxSamples, _maxReservoirRand;

    unsigned int *_tableMem;
    unsigned int **_reservoirs;
    omp_lock_t *_reservoirsLock;

    unsigned int *_global_rand;
    unsigned int _numReservoirs, _sequentialIDCounter_kernel;
    //     unsigned long long int _tableMemMax, _tableMemReservoirMax, _tablePointerMax;

    // Samples reservoirs and determines where to add data vectors.
    //     void reservoirSampling(unsigned int *allprobsHash, unsigned int *allprobsIdx,
    //                            unsigned int *storelog, unsigned int numProbePerTb);

    // Adds data vectors to tables at locations determined by reservoirSampling function.
    // Param dataOffset is used to account for indexing across nodes.
    void addTable(unsigned int *hashes, unsigned int *ids, unsigned int numInsertions,
                  unsigned int dataOffset);

  public:
    LSH(DOPH *hashFam, unsigned int numHashPerFamily, unsigned int numHashFamilies,
        unsigned int reservoirSize, unsigned int dimension, unsigned int maxSamples, int myRank,
        int worldSize);

    /* Adds input vectors (in sparse format) to the hash table.
    Each vector is assigned ascending identification starting 0.
    For numInputEntries > 1, simply concatenate data vectors.

    @param numInputEntries: number of input vectors.
    @param dataIdx: non-zero indice of the sparse format.
    @param dataVal: non-zero values of the sparse format.
    @param dataMarker: marks the start index of each vector in dataIdx and dataVal.
            Has an additional marker at the end to mark the (end+1) index.
    */
    void add(unsigned int numInputEntries, unsigned int *dataIdx, float *dataVal,
             unsigned int *dataMarker, unsigned int dataOffset);

    /* Computes hashes for a partition of the set of query vectors.

    @param queryPartitionSize: the number of query vectors in the partition.
    @param numQueryPartitionHashes: the number of hashes that need to be computed for the partition.
    @param queryPartitionIndices: the indices of the non zero elements of each query vector.
    @param queryPartitionVals: the values of the non zero elements of each query vector.
    @param queryPartitionMarkers: the indices of the beginning and end of each query vector.
            Has an additional marker at the end to mark the (end+1) index.
    @param queryHashes: an array to populate with the query hashes.
    */
    void getQueryHash(unsigned int queryPartitionSize, unsigned int numQueryPartitionHashes,
                      unsigned int *queryPartitionIndices, float *queryPartitionVals,
                      unsigned int *queryPartitionMarkers, unsigned int *queryHashes);

    /* Extractes the contents from the reservoirs of each hash table for some number of hashes.

    @param numQueryEntries: the number of queries to extract reservoirs for.
    @param segmentSize: the size of the block of memory for each query vector.
            Equal to numQueryProbes * numTables * reservoirSize.
    @param queue: array to store the contents of the reservoirs.
    @param hashIndices: the hash indices of the query vectors, corresponding to a reservoir(s) in
    each table.
    */
    void extractReservoirs(unsigned int numQuery, unsigned int *output, unsigned int *hashIndices);

    void resetSequentialKernalID();

    /* Print current parameter settings to the console.
     */
    void showParams();

    /* Check the memory load of the hash table.
     */
    //     void checkTableMemLoad();

    /* Prints contents of each hash table.
     */
    void tableContents();

    /* Destructor. Frees memory allocations and OpenCL environments.
     */
    ~LSH();
};

#endif