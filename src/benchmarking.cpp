#include "mpi.h"
#include "omp.h"

#include "CMS.h"
#include "LSH.h"
#include "LSHReservoirSampler.h"
#include "benchmarking.h"
#include "dataset.h"
#include "flashControl.h"
#include "indexing.h"
#include "mathUtils.h"
#include <chrono>

#define TOPK_BENCHMARK

void showConfig(std::string dataset, unsigned int numVectors, unsigned int queries, int nodes,
                unsigned int tables, unsigned int rangePow, unsigned int reservoirSize,
                unsigned int hashes, unsigned int cmsHashes, unsigned int cmsBucketSize) {
    std::cout << "\n=================\n== " << dataset << "\n=================\n" << std::endl;

    printf("%u Vectors, %u Queries\n", numVectors, queries);

    printf("Nodes: %u\nTables: %u\nRangePow: %u\nReservoir Size: "
           "%u\nHashes: %u\n",
           nodes, tables, rangePow, reservoirSize, hashes);

    printf("CMS Bucket Size: %u\nCMS Hashes: %u\n\n", cmsBucketSize, cmsHashes);
}

/*
 * WEBSPAM TESTING FUNCTION
 * WEBSPAM TESTING FUNCTION
 * WEBSPAM TESTING FUNCTION
 * WEBSPAM TESTING FUNCTION
 * WEBSPAM TESTING FUNCTION
 * WEBSPAM TESTING FUNCTION
 * WEBSPAM TESTING FUNCTION
 * WEBSPAM TESTING FUNCTION
 * WEBSPAM TESTING FUNCTION
 * WEBSPAM TESTING FUNCTION
 * WEBSPAM TESTING FUNCTION
 * WEBSPAM TESTING FUNCTION
 */

void testing() {
    /* ===============================================================
  MPI Initialization
  */
    int provided;
    MPI_Init_thread(0, 0, MPI_THREAD_FUNNELED, &provided);
    int myRank, worldSize;
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    if (myRank == 0) {
        showConfig("Webspam", NUM_DATA_VECTORS, NUM_QUERY_VECTORS, worldSize, NUM_TABLES, RANGE_POW,
                   RESERVOIR_SIZE, NUM_HASHES, CMS_HASHES, CMS_BUCKET_SIZE);
    }

    /* ===============================================================
  Data Structure Initialization
  */
    LSH *lsh = new LSH(NUM_HASHES, NUM_TABLES, RANGE_POW, worldSize, myRank);

    MPI_Barrier(MPI_COMM_WORLD);

    CMS *cms = new CMS(CMS_HASHES, CMS_BUCKET_SIZE, NUM_QUERY_VECTORS, myRank, worldSize);

    MPI_Barrier(MPI_COMM_WORLD);

    LSHReservoirSampler *reservoir = new LSHReservoirSampler(
        lsh, RANGE_POW, NUM_TABLES, RESERVOIR_SIZE, DIMENSION, RANGE_ROW_U,
        NUM_DATA_VECTORS + NUM_QUERY_VECTORS, ALLOC_FRACTION, myRank, worldSize);

    MPI_Barrier(MPI_COMM_WORLD);

    flashControl *control =
        new flashControl(reservoir, cms, myRank, worldSize, NUM_DATA_VECTORS, NUM_QUERY_VECTORS,
                         DIMENSION, NUM_TABLES, RESERVOIR_SIZE);

    /* ===============================================================
  Partitioning Query Between Nodes
  */

    control->allocateQuery(BASEFILE);

    MPI_Barrier(MPI_COMM_WORLD);

    /* ===============================================================
  Adding Vectors
  */
    std::cout << "Adding Vectors Node " << myRank << "..." << std::endl;
    auto start = std::chrono::system_clock::now();
    control->add(BASEFILE, 5000, 300, NUM_BATCHES, BATCH_PRINT);

    control->add(BASEFILE, 5000, 5300, NUM_BATCHES, BATCH_PRINT);

    control->add(BASEFILE, 5000, 10300, NUM_BATCHES, BATCH_PRINT);

    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Vectors Added Node " << myRank << ": " << elapsed.count() << " Seconds\n"
              << std::endl;

    // For debugging
    // control->printTables();

    MPI_Barrier(MPI_COMM_WORLD);

    /* ===============================================================
  Hashing Query Vectors
  */
    std::cout << "Computing Query Hashes Node " << myRank << "..." << std::endl;
    start = std::chrono::system_clock::now();
    control->hashQuery();
    end = std::chrono::system_clock::now();
    elapsed = end - start;
    std::cout << "Query Hashes Computed Node " << myRank << ": " << elapsed.count() << " Seconds\n"
              << std::endl;

    MPI_Barrier(MPI_COMM_WORLD);

    /* ===============================================================
  Extracting Reservoirs and Preforming Top-K selection
  */
    unsigned int *outputs = new unsigned int[TOPK * NUM_QUERY_VECTORS];
    start = std::chrono::system_clock::now();
    std::cout << "Extracting Top K (CMS) Node " << myRank << "..." << std::endl;
#ifdef CMS_AGGREGATION
    control->topKCMSAggregationTree(TOPK, outputs, 0);
#endif
#ifdef BF_AGGREGATION
    control->topKBruteForceAggretation(TOPK, outputs);
#endif
    end = std::chrono::system_clock::now();
    elapsed = end - start;
    std::cout << "Top K Extracted Node " << myRank << ": " << elapsed.count() << " Seconds\n"
              << std::endl;

    MPI_Barrier(MPI_COMM_WORLD);

    /* ===============================================================
  De-allocating Data Structures in Memory
  */
    delete control;
    delete reservoir;
    delete lsh;
    delete cms;

    /* ===============================================================
  MPI Closing
  */
    MPI_Finalize();

    if (myRank == 0) {
        /* ===============================================================
    Similarity and Accuracy Calculations
    */
        unsigned int totalNumVectors = NUM_DATA_VECTORS + NUM_QUERY_VECTORS;
        unsigned int *sparseIndices = new unsigned int[totalNumVectors * DIMENSION];
        float *sparseVals = new float[totalNumVectors * DIMENSION];
        unsigned int *sparseMarkers = new unsigned int[totalNumVectors + 1];

        readSparse(BASEFILE, 0, totalNumVectors, sparseIndices, sparseVals, sparseMarkers,
                   totalNumVectors * DIMENSION);

        unsigned int nCnt = 10;
        unsigned int nList[nCnt] = {1, 10, 20, 30, 32, 40, 50, 64, 100, TOPK};

        std::cout << "\n\n================================\nTOP K CMS\n" << std::endl;

        similarityMetric(sparseIndices, sparseVals, sparseMarkers, sparseIndices, sparseVals,
                         sparseMarkers, outputs, NUM_QUERY_VECTORS, TOPK, AVAILABLE_TOPK, nList,
                         nCnt);
        std::cout << "Similarity Metric Computed" << std::endl;
        std::cout << "Evaluation Complete" << std::endl;

        /* ===============================================================
    De-allocating Memory
    */
        delete[] sparseIndices;
        delete[] sparseVals;
        delete[] sparseMarkers;
    }
    delete[] outputs;
}

/**
 * Eval With Similarity
 * Eval With Similarity
 * Eval With Similarity
 * Eval With Similarity
 * Eval With Similarity
 * Eval With Similarity
 * Eval With Similarity
 * Eval With Similarity
 * Eval With Similarity
 * Eval With Similarity
 * Eval With Similarity
 */

void evalWithSimilarity() {
    /* ===============================================================
  MPI Initialization
  */
    int provided;
    MPI_Init_thread(0, 0, MPI_THREAD_FUNNELED, &provided);
    int myRank, worldSize;
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    if (myRank == 0) {
        showConfig(BASEFILE, NUM_DATA_VECTORS, NUM_QUERY_VECTORS, worldSize, NUM_TABLES, RANGE_POW,
                   RESERVOIR_SIZE, NUM_HASHES, CMS_HASHES, CMS_BUCKET_SIZE);
    }

    /* ===============================================================
  Data Structure Initialization
  */
    LSH *lsh = new LSH(NUM_HASHES, NUM_TABLES, RANGE_POW, worldSize, myRank);

    MPI_Barrier(MPI_COMM_WORLD);

    CMS *cms = new CMS(CMS_HASHES, CMS_BUCKET_SIZE, NUM_QUERY_VECTORS, myRank, worldSize);

    MPI_Barrier(MPI_COMM_WORLD);

    LSHReservoirSampler *reservoir = new LSHReservoirSampler(
        lsh, RANGE_POW, NUM_TABLES, RESERVOIR_SIZE, DIMENSION, RANGE_ROW_U,
        NUM_DATA_VECTORS + NUM_QUERY_VECTORS, ALLOC_FRACTION, myRank, worldSize);

    MPI_Barrier(MPI_COMM_WORLD);

    flashControl *control =
        new flashControl(reservoir, cms, myRank, worldSize, NUM_DATA_VECTORS, NUM_QUERY_VECTORS,
                         DIMENSION, NUM_TABLES, RESERVOIR_SIZE);

    /* ===============================================================
  Partitioning Query Between Nodes
  */

    control->allocateQuery(BASEFILE);

    MPI_Barrier(MPI_COMM_WORLD);

    /* ===============================================================
  Adding Vectors
  */
    std::cout << "Adding Vectors Node " << myRank << "..." << std::endl;
    auto start = std::chrono::system_clock::now();
    control->add(BASEFILE, NUM_DATA_VECTORS, NUM_QUERY_VECTORS, NUM_BATCHES, BATCH_PRINT);
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Vectors Added Node " << myRank << ": " << elapsed.count() << " Seconds\n"
              << std::endl;

    MPI_Barrier(MPI_COMM_WORLD);

    /* ===============================================================
  Hashing Query Vectors
  */
    std::cout << "Computing Query Hashes Node " << myRank << "..." << std::endl;
    start = std::chrono::system_clock::now();
    control->hashQuery();
    end = std::chrono::system_clock::now();
    elapsed = end - start;
    std::cout << "Query Hashes Computed Node " << myRank << ": " << elapsed.count() << " Seconds\n"
              << std::endl;

    MPI_Barrier(MPI_COMM_WORLD);

    /* ===============================================================
  Extracting Reservoirs and Preforming Top-K selection
  */
    unsigned int *treeOutputs = new unsigned int[TOPK * NUM_QUERY_VECTORS];
    start = std::chrono::system_clock::now();
    std::cout << "Extracting Top K (TREE) Node " << myRank << "..." << std::endl;
    control->topKCMSAggregationTree(TOPK, treeOutputs, 0);
    end = std::chrono::system_clock::now();
    elapsed = end - start;
    std::cout << "Top K (TREE) Extracted Node " << myRank << ": " << elapsed.count() << " Seconds\n"
              << std::endl;

    MPI_Barrier(MPI_COMM_WORLD);

    // ==============================================

    unsigned int *linearOutputs = new unsigned int[TOPK * NUM_QUERY_VECTORS];
    start = std::chrono::system_clock::now();
    std::cout << "Extracting Top K (LINEAR) Node " << myRank << "..." << std::endl;
    control->topKCMSAggregationLinear(TOPK, linearOutputs, 0);
    end = std::chrono::system_clock::now();
    elapsed = end - start;
    std::cout << "Top K (LINEAR) Extracted Node " << myRank << ": " << elapsed.count()
              << " Seconds\n"
              << std::endl;

    MPI_Barrier(MPI_COMM_WORLD);

    // ==============================================

    unsigned int *bruteforceOutputs = new unsigned int[TOPK * NUM_QUERY_VECTORS];
    start = std::chrono::system_clock::now();
    std::cout << "Extracting Top K (BRUTEFORCE) Node " << myRank << "..." << std::endl;
    control->topKBruteForceAggretation(TOPK, bruteforceOutputs);
    end = std::chrono::system_clock::now();
    elapsed = end - start;
    std::cout << "Top K (BRUTEFORCE) Extracted Node " << myRank << ": " << elapsed.count()
              << " Seconds\n"
              << std::endl;

    MPI_Barrier(MPI_COMM_WORLD);

    /* ===============================================================
  De-allocating Data Structures in Memory
  */
    delete control;
    delete reservoir;
    delete lsh;
    delete cms;

    /* ===============================================================
  MPI Closing
  */
    MPI_Finalize();

    if (myRank == 0) {
        /* ===============================================================
    Similarity and Accuracy Calculations
    */
        unsigned int totalNumVectors = NUM_DATA_VECTORS + NUM_QUERY_VECTORS;
        unsigned int *sparseIndices = new unsigned int[totalNumVectors * DIMENSION];
        float *sparseVals = new float[totalNumVectors * DIMENSION];
        unsigned int *sparseMarkers = new unsigned int[totalNumVectors + 1];

        readSparse(BASEFILE, 0, totalNumVectors, sparseIndices, sparseVals, sparseMarkers,
                   totalNumVectors * DIMENSION);

        unsigned int nCnt = 10;
        unsigned int nList[nCnt] = {1, 10, 20, 30, 32, 40, 50, 64, 100, TOPK};

        std::cout << "\n\n================================\nTOP K TREE\n" << std::endl;

        similarityMetric(sparseIndices, sparseVals, sparseMarkers, sparseIndices, sparseVals,
                         sparseMarkers, treeOutputs, NUM_QUERY_VECTORS, TOPK, AVAILABLE_TOPK, nList,
                         nCnt);

        std::cout << "\n\n================================\nTOP K LINEAR\n" << std::endl;

        similarityMetric(sparseIndices, sparseVals, sparseMarkers, sparseIndices, sparseVals,
                         sparseMarkers, linearOutputs, NUM_QUERY_VECTORS, TOPK, AVAILABLE_TOPK,
                         nList, nCnt);

        std::cout << "\n\n================================\nTOP K "
                     "BRUTEFORCE\n"
                  << std::endl;

        similarityMetric(sparseIndices, sparseVals, sparseMarkers, sparseIndices, sparseVals,
                         sparseMarkers, bruteforceOutputs, NUM_QUERY_VECTORS, TOPK, AVAILABLE_TOPK,
                         nList, nCnt);

        std::cout << "Similarity Metric Computed" << std::endl;

        /* ===============================================================
    De-allocating Memory
    */
        delete[] sparseIndices;
        delete[] sparseVals;
        delete[] sparseMarkers;
    }
    delete[] treeOutputs;
    delete[] linearOutputs;
    delete[] bruteforceOutputs;
}

/**
 * Eval with File Output
 * Eval with File Output
 * Eval with File Output
 * Eval with File Output
 * Eval with File Output
 * Eval with File Output
 * Eval with File Output
 * Eval with File Output
 * Eval with File Output
 * Eval with File Output
 */
void evalWithFileOutput() {
    /* ===============================================================
  MPI Initialization
  */
    int provided;
    MPI_Init_thread(0, 0, MPI_THREAD_FUNNELED, &provided);
    int myRank, worldSize;
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    if (myRank == 0) {
        showConfig(BASEFILE, NUM_DATA_VECTORS, NUM_QUERY_VECTORS, worldSize, NUM_TABLES, RANGE_POW,
                   RESERVOIR_SIZE, NUM_HASHES, CMS_HASHES, CMS_BUCKET_SIZE);
    }

    /* ===============================================================
  Data Structure Initialization
  */
    LSH *lsh = new LSH(NUM_HASHES, NUM_TABLES, RANGE_POW, worldSize, myRank);

    MPI_Barrier(MPI_COMM_WORLD);

    CMS *cms = new CMS(CMS_HASHES, CMS_BUCKET_SIZE, NUM_QUERY_VECTORS, myRank, worldSize);

    MPI_Barrier(MPI_COMM_WORLD);

    LSHReservoirSampler *reservoir = new LSHReservoirSampler(
        lsh, RANGE_POW, NUM_TABLES, RESERVOIR_SIZE, DIMENSION, RANGE_ROW_U,
        NUM_DATA_VECTORS + NUM_QUERY_VECTORS, ALLOC_FRACTION, myRank, worldSize);

    MPI_Barrier(MPI_COMM_WORLD);

    flashControl *control =
        new flashControl(reservoir, cms, myRank, worldSize, NUM_DATA_VECTORS, NUM_QUERY_VECTORS,
                         DIMENSION, NUM_TABLES, RESERVOIR_SIZE);

    /* ===============================================================
  Reading Data
  */
    std::cout << "\nReading Data Node " << myRank << "..." << std::endl;
    auto start = std::chrono::system_clock::now();

    // control->allocateData(BASEFILE);
    MPI_Barrier(MPI_COMM_WORLD);

    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Data Read Node " << myRank << ": " << elapsed.count() << " Seconds\n"
              << std::endl;

    /* ===============================================================
  Partitioning Query Between Nodes
  */

    control->allocateQuery(BASEFILE);

    MPI_Barrier(MPI_COMM_WORLD);

    /* ===============================================================
  Adding Vectors
  */
    std::cout << "Adding Vectors Node " << myRank << "..." << std::endl;
    start = std::chrono::system_clock::now();
    control->add(BASEFILE, NUM_DATA_VECTORS, NUM_QUERY_VECTORS, NUM_BATCHES, BATCH_PRINT);
    end = std::chrono::system_clock::now();
    elapsed = end - start;
    std::cout << "Vectors Added Node " << myRank << ": " << elapsed.count() << " Seconds\n"
              << std::endl;

    MPI_Barrier(MPI_COMM_WORLD);

    /* ===============================================================
  Hashing Query Vectors
  */
    std::cout << "Computing Query Hashes Node " << myRank << "..." << std::endl;
    start = std::chrono::system_clock::now();
    control->hashQuery();
    end = std::chrono::system_clock::now();
    elapsed = end - start;
    std::cout << "Query Hashes Computed Node " << myRank << ": " << elapsed.count() << " Seconds\n"
              << std::endl;

    MPI_Barrier(MPI_COMM_WORLD);

    /* ===============================================================
  Extracting Reservoirs and Preforming Top-K selection
  */
    unsigned int *treeOutputs = new unsigned int[TOPK * NUM_QUERY_VECTORS];
    start = std::chrono::system_clock::now();
    std::cout << "Extracting Top K (TREE) Node " << myRank << "..." << std::endl;
    control->topKCMSAggregationTree(TOPK, treeOutputs, 0);
    end = std::chrono::system_clock::now();
    elapsed = end - start;
    std::cout << "Top K (TREE) Extracted Node " << myRank << ": " << elapsed.count() << " Seconds\n"
              << std::endl;

    std::string filenameTree("Tree-Nodes-");
    filenameTree.append(std::to_string(worldSize));
    if (myRank == 0) {
        writeTopK(filenameTree, NUM_QUERY_VECTORS, TOPK, treeOutputs);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // ==============================================

    unsigned int *linearOutputs = new unsigned int[TOPK * NUM_QUERY_VECTORS];
    start = std::chrono::system_clock::now();
    std::cout << "Extracting Top K (LINEAR) Node " << myRank << "..." << std::endl;
    control->topKCMSAggregationLinear(TOPK, linearOutputs, 0);
    end = std::chrono::system_clock::now();
    elapsed = end - start;
    std::cout << "Top K (LINEAR) Extracted Node " << myRank << ": " << elapsed.count()
              << " Seconds\n"
              << std::endl;

    std::string filenameLinear("Linear-Nodes-");
    filenameLinear.append(std::to_string(worldSize));
    if (myRank == 0) {
        writeTopK(filenameLinear, NUM_QUERY_VECTORS, TOPK, linearOutputs);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // ==============================================

    unsigned int *bruteforceOutputs = new unsigned int[TOPK * NUM_QUERY_VECTORS];
    start = std::chrono::system_clock::now();
    std::cout << "Extracting Top K (BRUTEFORCE) Node " << myRank << "..." << std::endl;
    control->topKBruteForceAggretation(TOPK, bruteforceOutputs);
    end = std::chrono::system_clock::now();
    elapsed = end - start;
    std::cout << "Top K (BRUTEFORCE) Extracted Node " << myRank << ": " << elapsed.count()
              << " Seconds\n"
              << std::endl;

    std::string filenameBruteforce("Bruteforce-Nodes-");
    filenameBruteforce.append(std::to_string(worldSize));
    if (myRank == 0) {
        writeTopK(filenameBruteforce, NUM_QUERY_VECTORS, TOPK, bruteforceOutputs);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    /* ===============================================================
  De-allocating Data Structures in Memory
  */
    delete control;
    delete reservoir;
    delete lsh;
    delete cms;
    delete[] treeOutputs;
    delete[] linearOutputs;
    delete[] bruteforceOutputs;
    /* ===============================================================
  MPI Closing
  */
    MPI_Finalize();
}

/*
 * WIKIDUMP
 * WIKIDUMP
 * WIKIDUMP
 * WIKIDUMP
 * WIKIDUMP
 * WIKIDUMP
 * WIKIDUMP
 * WIKIDUMP
 * WIKIDUMP
 */

void wikiDump() {
    /* ===============================================================
  MPI Initialization
  */
    int provided;
    MPI_Init_thread(0, 0, MPI_THREAD_FUNNELED, &provided);
    int myRank, worldSize;
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    if (myRank == 0) {
        showConfig("WikiDump", NUM_DATA_VECTORS, NUM_QUERY_VECTORS, worldSize, NUM_TABLES,
                   RANGE_POW, RESERVOIR_SIZE, NUM_HASHES, CMS_HASHES, CMS_BUCKET_SIZE);
    }

    /* ===============================================================
  Data Structure Initialization
  */
    LSH *lsh = new LSH(NUM_HASHES, NUM_TABLES, RANGE_POW, worldSize, myRank);

    MPI_Barrier(MPI_COMM_WORLD);

    CMS *cms = new CMS(CMS_HASHES, CMS_BUCKET_SIZE, NUM_QUERY_VECTORS, myRank, worldSize);

    MPI_Barrier(MPI_COMM_WORLD);

    LSHReservoirSampler *reservoir = new LSHReservoirSampler(
        lsh, RANGE_POW, NUM_TABLES, RESERVOIR_SIZE, DIMENSION, RANGE_ROW_U,
        NUM_DATA_VECTORS + NUM_QUERY_VECTORS, ALLOC_FRACTION, myRank, worldSize);

    MPI_Barrier(MPI_COMM_WORLD);

    flashControl *control =
        new flashControl(reservoir, cms, myRank, worldSize, NUM_DATA_VECTORS, NUM_QUERY_VECTORS,
                         DIMENSION, NUM_TABLES, RESERVOIR_SIZE);

    /* ===============================================================
  Reading Data
  */
    std::cout << "\nReading Data Node " << myRank << "..." << std::endl;
    auto start = std::chrono::system_clock::now();

    // control->allocateData(BASEFILE);
    MPI_Barrier(MPI_COMM_WORLD);

    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Data Read Node " << myRank << ": " << elapsed.count() << " Seconds\n"
              << std::endl;

    /* ===============================================================
  Partitioning Query Between Nodes
  */

    control->allocateQuery(BASEFILE);

    MPI_Barrier(MPI_COMM_WORLD);

    /* ===============================================================
  Adding Vectors
  */
    std::cout << "Adding Vectors Node " << myRank << "..." << std::endl;
    start = std::chrono::system_clock::now();
    control->add(BASEFILE, NUM_DATA_VECTORS, NUM_QUERY_VECTORS, NUM_BATCHES, BATCH_PRINT);
    end = std::chrono::system_clock::now();
    elapsed = end - start;
    std::cout << "Vectors Added Node " << myRank << ": " << elapsed.count() << " Seconds\n"
              << std::endl;

    MPI_Barrier(MPI_COMM_WORLD);

    /* ===============================================================
  Hashing Query Vectors
  */
    std::cout << "Computing Query Hashes Node " << myRank << "..." << std::endl;
    start = std::chrono::system_clock::now();
    control->hashQuery();
    end = std::chrono::system_clock::now();
    elapsed = end - start;
    std::cout << "Query Hashes Computed Node " << myRank << ": " << elapsed.count() << " Seconds\n"
              << std::endl;

    MPI_Barrier(MPI_COMM_WORLD);

    /* ===============================================================
  Extracting Reservoirs and Preforming Top-K selection
  */
    unsigned int *treeOutputs = new unsigned int[TOPK * NUM_QUERY_VECTORS];
    start = std::chrono::system_clock::now();
    std::cout << "Extracting Top K (TREE) Node " << myRank << "..." << std::endl;
    control->topKCMSAggregationTree(TOPK, treeOutputs, 0);
    end = std::chrono::system_clock::now();
    elapsed = end - start;
    std::cout << "Top K (TREE) Extracted Node " << myRank << ": " << elapsed.count() << " Seconds\n"
              << std::endl;

    std::string filenameTree("WikiDump-");
    filenameTree.append(std::to_string(worldSize));
    if (myRank == 0) {
        writeTopK(filenameTree, NUM_QUERY_VECTORS, TOPK, treeOutputs);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    /* ===============================================================
  De-allocating Data Structures in Memory
  */
    delete control;
    delete reservoir;
    delete lsh;
    delete cms;
    delete[] treeOutputs;

    /* ===============================================================
  MPI Closing
  */
    MPI_Finalize();
}

/*
 * CRITEO
 * CRITEO
 * CRITEO
 * CRITEO
 * CRITEO
 * CRITEO
 * CRITEO
 * CRITEO
 * CRITEO
 */

void criteo() {
    /* ===============================================================
    MPI Initialization
    */
    int provided;
    MPI_Init_thread(0, 0, MPI_THREAD_FUNNELED, &provided);
    int myRank, worldSize;
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    if (myRank == 0) {
        showConfig("Criteo", NUM_DATA_VECTORS, NUM_QUERY_VECTORS, worldSize, NUM_TABLES, RANGE_POW,
                   RESERVOIR_SIZE, NUM_HASHES, CMS_HASHES, CMS_BUCKET_SIZE);
    }

    /* ===============================================================
    Data Structure Initialization
    */
    LSH *lsh = new LSH(NUM_HASHES, NUM_TABLES, RANGE_POW, worldSize, myRank);

    MPI_Barrier(MPI_COMM_WORLD);

    CMS *cms = new CMS(CMS_HASHES, CMS_BUCKET_SIZE, NUM_QUERY_VECTORS, myRank, worldSize);

    MPI_Barrier(MPI_COMM_WORLD);

    LSHReservoirSampler *reservoir = new LSHReservoirSampler(
        lsh, RANGE_POW, NUM_TABLES, RESERVOIR_SIZE, DIMENSION, RANGE_ROW_U,
        NUM_DATA_VECTORS + NUM_QUERY_VECTORS, ALLOC_FRACTION, myRank, worldSize);

    MPI_Barrier(MPI_COMM_WORLD);

    flashControl *control =
        new flashControl(reservoir, cms, myRank, worldSize, NUM_DATA_VECTORS, NUM_QUERY_VECTORS,
                         DIMENSION, NUM_TABLES, RESERVOIR_SIZE);

    if (myRank == 0) {
        reservoir->showParams();
    }

    /* ===============================================================
    Partitioning Query Between Nodes
    */
    control->allocateQuery(BASEFILE);

    MPI_Barrier(MPI_COMM_WORLD);

    /* ===============================================================
    Adding Vectors
    */
    std::cout << "Adding Vectors Node " << myRank << "..." << std::endl;
    auto start = std::chrono::system_clock::now();
    // size_t offset = NUM_QUERY_VECTORS;
    // size_t batchSize = NUM_DATA_VECTORS / 100;
    // if (myRank == 0) {
    //     printf("Batch Size: %lu For 100 Batches", batchSize);
    // }
    // for (size_t i = 0; i < 100; i++) {
    //     control->add(BASEFILE, batchSize, offset + i * batchSize, NUM_BATCHES, BATCH_PRINT);
    //     if (i % 5 == 0) {
    //         MPI_Barrier(MPI_COMM_WORLD);
    //         if (myRank == 0) {
    //             std::cout << "Batch Complete: " << i << std::endl;
    //         }
    //     }
    // }
    // control->add(BASEFILE, 1000000, 0, 10, 10);
    // control->addPartitioned(BASEFILE, NUM_DATA_VECTORS, 1, 1);

    for (unsigned int i = 0; i < 3; i++) {
        std::string thisFile(BASEFILE);
        thisFile.append(std::to_string(i));
        std::cout << "Adding from file " << thisFile << std::endl;
        control->add(thisFile, NUM_DATA_VECTORS, 0, 1, 1);
        MPI_Barrier(MPI_COMM_WORLD);
    }
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Vectors Added Node " << myRank << ": " << elapsed.count() << " Seconds\n"
              << std::endl;

    MPI_Barrier(MPI_COMM_WORLD);

    /* ===============================================================
    Hashing Query Vectors
    */
    std::cout << "Computing Query Hashes Node " << myRank << "..." << std::endl;
    start = std::chrono::system_clock::now();
    control->hashQuery();
    end = std::chrono::system_clock::now();
    elapsed = end - start;
    std::cout << "Query Hashes Computed Node " << myRank << ": " << elapsed.count() << " Seconds\n"
              << std::endl;

    MPI_Barrier(MPI_COMM_WORLD);

    /* ===============================================================
    Extracting Reservoirs and Preforming Top-K selection
    */
    unsigned int *treeOutputs = new unsigned int[TOPK * NUM_QUERY_VECTORS];
    start = std::chrono::system_clock::now();
    std::cout << "Extracting Top K (TREE) Node " << myRank << "..." << std::endl;
    control->topKCMSAggregationTree(TOPK, treeOutputs, 0);
    end = std::chrono::system_clock::now();
    elapsed = end - start;
    std::cout << "Top K (TREE) Extracted Node " << myRank << ": " << elapsed.count() << " Seconds\n"
              << std::endl;

    std::string filenameTree("Criteo-");
    filenameTree.append(std::to_string(worldSize));
    if (myRank == 0) {
        writeTopK(filenameTree, NUM_QUERY_VECTORS, TOPK, treeOutputs);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    /* ===============================================================
    De-allocating Data Structures in Memory
    */
    delete control;
    delete reservoir;
    delete lsh;
    delete cms;
    delete[] treeOutputs;

    /* ===============================================================
    MPI Closing
    */
    MPI_Finalize();
}

void evaluateResults(std::string resultFile) {

    unsigned int totalNumVectors = NUM_DATA_VECTORS + NUM_QUERY_VECTORS;
    unsigned int *outputs = new unsigned int[NUM_QUERY_VECTORS * TOPK];
    readTopK(resultFile, NUM_QUERY_VECTORS, TOPK, outputs);

    unsigned int *sparseIndices = new unsigned int[((long)totalNumVectors * DIMENSION)];
    float *sparseVals = new float[((long)totalNumVectors * DIMENSION)];
    unsigned int *sparseMarkers = new unsigned int[totalNumVectors + 1];

    readSparse(BASEFILE, 0, totalNumVectors, sparseIndices, sparseVals, sparseMarkers,
               totalNumVectors * DIMENSION);

    unsigned int nCnt = 10;
    unsigned int nList[nCnt] = {1, 10, 20, 30, 32, 40, 50, 64, 100, TOPK};

    std::cout << "\n\n================================\nTOP K TREE\n" << std::endl;

    similarityMetric(sparseIndices, sparseVals, sparseMarkers, sparseIndices, sparseVals,
                     sparseMarkers, outputs, NUM_QUERY_VECTORS, TOPK, AVAILABLE_TOPK, nList, nCnt);
}
