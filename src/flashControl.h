#ifndef _FLASH_CONTROL_H
#define _FLASH_CONTROL_H

#include "CMS.h"
#include "LSH.h"
#include "benchmarking.h"
#include "dataset.h"
#include "reader.h"
#include <algorithm>
#include <chrono>
#include <iostream>
#include <mpi.h>
#include <string>

struct VectorFrequency {
    unsigned int vector;
    int count;
};

class flashControl {

  private:
    int _myRank, _worldSize;

    LSH *_myReservoir;

    CMS *_mySketch;

    // Reservoir Params
    unsigned int _numTables, _reservoirSize;

    unsigned int _numQueryVectors,
        _dimension; // Total number of data and query vectors across all nodes

    unsigned int *_dataVectorCts; // Number of data vectors allocated to each node
    unsigned int
        *_dataVectorOffsets; // Offset, in number of vectors, for the range of vectors of each node

    int *_queryVectorCts;     // Number of query vectors allocated to each node
    int *_queryVectorOffsets; // Offset, in number of vectors, for the range of vectors of each node
    int *_queryCts;           // Total length of all query vectors allocated to each node
    int *_queryOffsets; // Offset, in total length of query vectors, of the query range of each node

    // For storing the partition of the query vectors allocated to each node
    int _myQueryVectorsCt;  // Number of query vectors allocated to a specific node
    int _myQueryVectorsLen; // Combined length of all of the query vectors allocated to a
                            // specific node
    unsigned int
        *_myQueryIndices; // Location of non-zeros within a node's partition of query vectors
    float *_myQueryVals;  // Values of non-zeros within a node's partition of query vectors
    unsigned int *_myQueryMarkers; // Start and end indexes of query vectors within a node's
                                   // partition of query vectors

    int _myHashCt;     // Number of hashes computed by specific node
    int *_hashCts;     // Number of hashes computed by each node
    int *_hashOffsets; // Offsets of hash array for each node

    unsigned int *_allQueryHashes; // Combined hashes from all nodes

    unsigned int *_queryIndices;
    float *_queryVals;
    unsigned int *_queryMarkers;

    unsigned int *myDataIndices;
    float *myDataVals;
    unsigned int *myDataMarkers;

  public:
    /* Constructor.
        Initializes a FLASH Controller object that manages communication and data partitions between
    nodes. Determines how the many data vectors will be sent to each node and how many query vectors
    each node will hash.

    @param reservoir: a LSHReservoirSampler object.
    @param cms: a count min sketch/topKAPI obect.
    @param myRank: the rank of node calling the constructor.
    @param worldSize: the total number of nodes.
    @param numDataVectors: the total number of dataVectors that will be added across all nodes.
    @param numQueryVectors: the total number of queryVectors that will used across all nodes.
    @param dimension: the dimension of each vector (or max for sparse datasets)
    @param numTables: the number of tables in the instance of LSHReservoirSampler.
    @param reservoirSize: the size of each reservoir in the instance of LSHReservoirSampler.
    */
    flashControl(LSH *reservoir, CMS *cms, int myRank, int worldSize,
                 unsigned int numDataVectors, unsigned int numQueryVectors, unsigned int dimension,
                 unsigned int numTables, unsigned int reservoirSize);

    // Allocates memory in each node and sends each node its partition of the set of data vectors.
    // void allocateData(std::string filename);

    // Allocates memory in each node and sends each node its partition of the set of query vectors
    // for hashing.
    void allocateQuery(std::string filename);

    std::streampos allocateQuery2(std::string filename, std::streampos offset);

    /* Adds a nodes set of data vectors to its LSHReservoirSampler object.

    @param numBatches: the number of batches to break the data into when adding.
    @param batchPrint: after each set of this number of batches the function will print the
    memory usage of each hash table.
    */
    void add(std::string filename, unsigned int n, unsigned int offset, unsigned int numBatches,
             unsigned int batchPrint);

    void addPartitioned(std::string base_filename, unsigned int numDataVectorsPerNode,
                        unsigned int numBatches, unsigned int batchPrint);

    // Computes the hashes of each partition of the query vectors in each node, and then
    // combines each partition of hashes into a single set of hashes in every node.
    void hashQuery();

    void query(std::string filename, std::string outputFileName, unsigned int batches,
               unsigned int topK);

    /* Extracts reservoirs from each node's hash tables, and sends these top k candidates to
       node 0, which preforms frequency counts and selects the top-k.

    @param topK: the number of top elements to select.
    @param outputs: an array to store the selected top-k for each query vector.
    */
    void topKBruteForceAggretation(unsigned int topK, unsigned int *outputs);

    /* Extracts reservoirs from each node's hash tables, stores frequency counts for each node
        in a CMS object, aggregates CMS objects in node 0, and preforms top-k selection there.

    @param topK: the number of top elements to select.
    @param outputs: an array to store the selected top-k for each query vector.
    @param threshold: used for extracting heavy hitters in topKAPI
    */
    void topKCMSAggregationTree(unsigned int topK, unsigned int *outputs, unsigned int threshold);

    void topKCMSAggregationLinear(unsigned int topK, unsigned int *outputs, unsigned int threshold);

    // For debugging: shows the partitions of the data and query allocated to a specific node.
    void showPartitions();

    void printTables();

    void checkDataTransfer();

    void checkQueryHashes();

    // Destructor
    ~flashControl();
};

#endif
