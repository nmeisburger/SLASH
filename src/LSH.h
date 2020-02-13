#ifndef _LSH_H
#define _LSH_H

#define _CRT_SECURE_NO_DEPRECATE
#include "mpi.h"
#include "omp.h"
#include <iostream>
#include <random>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define UNIVERSAL_HASH(x, M, a, b) ((unsigned)(a * x + b) >> (32 - M))
#define BINARY_HASH(x, a, b) ((unsigned)(a * x + b) >> 31)

#define hashIndicesOutputIdx(numHashFamilies, numInputs, dataIdx, tb)                              \
    (unsigned long long)(numInputs * tb + dataIdx)
#define hashesOutputIdx(numHashPerFamily, numInputs, dataIdx, tb, hashInFamIdx)                    \
    (unsigned long long)(tb * (numInputs * numHashPerFamily) + dataIdx * numHashPerFamily +        \
                         hashInFamIdx)

class LSH {
  private:
    /* Core parameters. */
    unsigned int _rangePow; // Here _rangePow always means the power2 range of
                            // the hash table.
    unsigned int _numTables;
    unsigned int _numhashes, _lognumhash, _K, _L;
    int *_randHash, _randa, *_rand1;

    // MPI
    int _worldSize, _worldRank;

    // Determines alternative hash to use for densification if a hash bin is
    // empty after DOPH.
    unsigned int getRandDoubleHash(int binid, int count);

    // Computes K * L hashes for a vector using DOPH.
    void optimalMinHash(unsigned int *hashArray, unsigned int *nonZeros, unsigned int sizenonzeros);

  public:
    /** Obtain hash indice given the (sparse) input vector.
    Hash indice refer to the corresponding "row number" in a hash table, in
    the form of unsigned integer. This function will only be valid when an
    CPU implementation exists for that type of hashing. The outputs indexing
    is defined as hashIndicesOutputIdx(numHashFamilies, numProbes,
    numInputs, inputIdx, probeIdx, tb) (unsigned)(numInputs * numProbes * tb
    + inputIdx * numProbes
    + probeIdx).

    @param hashIndices: for storing hash indices for each vector.
    @param probeDataIdx: for storing the index of the vector corresponding
    to each hash.
    @param dataIdx: non-zero indice of the sparse format.
    @param dataMarker: marks the start index of each vector in dataIdx and
    dataVal. Has additional element at the end for the index of the (end +
    1) element.
    @param numInputEntries: number of input vectors.
    @param numProbes: number of probes per input.
    */
    void getHashes(unsigned int *hashIndices, unsigned int *probeDataIdx, unsigned int *dataIdx,
                   unsigned int *dataMarker, size_t numInputEntries);

    void showLSHConfig();

    /** Constructor.

    Construct an LSH class for optimal densified min-hash (for more details
    refer to Anshumali Shrivastava, anshu@rice.edu). This hashing scheme is
    for very sparse and high dimensional data stored in sparse format.

    @param _K_in: the number of hash functions per table.
    @param _L_in: the number of hash tables.
    @param _rangePow_in: log2 of the range (number of rows or reservoirs) of
    each hash table.
    @param worldSize: the total number of nodes.
    @param worldRank: the rank of the specific node calling the constructor.
    */
    LSH(unsigned int _K_in, unsigned int _L_in, unsigned int _rangePow_in, int worldSize,
        int worldRank);

    // Destructor
    ~LSH();
};

#endif
