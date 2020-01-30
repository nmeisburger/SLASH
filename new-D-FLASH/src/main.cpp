#include "Config.h"
#include "LSH_Hasher.h"
#include "LSH_Reservoir.h"
#include <iostream>

void evaluateResults(std::string resultFile);

int main() {

    int provided;
    MPI_Init_thread(0, 0, MPI_THREAD_FUNNELED, &provided);
    int my_rank, world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    LSH_Hasher *lsh = new LSH_Hasher(NUM_HASHES, NUM_TABLES, RANGE_POW, world_size, my_rank);

    // lsh->showLSHConfig();

    // unsigned int *markers = new unsigned int[NUM_DATA_VECTORS];
    // unsigned int *indices = new unsigned int[DIMENSION * NUM_DATA_VECTORS];
    // float *vals = new float[DIMENSION * NUM_DATA_VECTORS];

    // read_sparse(BASEFILE, NUM_QUERY_VECTORS, NUM_DATA_VECTORS, indices, vals, markers,
    //             DIMENSION * NUM_DATA_VECTORS);

    CMS *cms = new CMS(CMS_HASHES, CMS_BUCKET_SIZE, NUM_QUERY_VECTORS, my_rank, world_size);

    LSH_Reservoir *reservoir = new LSH_Reservoir(NUM_TABLES, NUM_HASHES, RANGE_POW, RESERVOIR_SIZE,
                                                 lsh, cms, my_rank, world_size);

    reservoir->add_dist(BASEFILE, NUM_QUERY_VECTORS, NUM_DATA_VECTORS, DIMENSION);

    printf("Add finished\n");

    unsigned int *outputs = new unsigned int[TOPK * NUM_QUERY_VECTORS];

    reservoir->query_dist(BASEFILE, 0, NUM_QUERY_VECTORS, DIMENSION, TOPK, outputs);

    printf("Query finished\n");

    delete[] outputs;

    MPI_Finalize();

    // if (my_rank == 0) {
    //     writeTopK("test_output", NUM_QUERY_VECTORS, TOPK, outputs);

    //     evaluateResults("test_output");
    // }

    return 0;
}

void evaluateResults(std::string resultFile) {

    unsigned int totalNumVectors = NUM_DATA_VECTORS + NUM_QUERY_VECTORS;
    unsigned int *outputs = new unsigned int[NUM_QUERY_VECTORS * TOPK];
    readTopK(resultFile, NUM_QUERY_VECTORS, TOPK, outputs);

    unsigned int *sparseIndices = new unsigned int[((long)totalNumVectors * DIMENSION)];
    float *sparseVals = new float[((long)totalNumVectors * DIMENSION)];
    unsigned int *sparseMarkers = new unsigned int[totalNumVectors + 1];

    read_sparse(BASEFILE, 0, totalNumVectors, sparseIndices, sparseVals, sparseMarkers,
                totalNumVectors * DIMENSION);

    const int nCnt = 10;
    int nList[nCnt] = {1, 10, 20, 30, 32, 40, 50, 64, 100, TOPK};

    std::cout << "\n\n================================\nTOP K TREE\n" << std::endl;

    similarityMetric(sparseIndices, sparseVals, sparseMarkers, sparseIndices, sparseVals,
                     sparseMarkers, outputs, NUM_QUERY_VECTORS, TOPK, AVAILABLE_TOPK, nList, nCnt);
}