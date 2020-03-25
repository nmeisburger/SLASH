#include <algorithm>
#include <assert.h>
#include <fstream>
#include <iostream>
#include <math.h>
#include <sstream>

// #define NUM_DATA_VECTORS 6
// #define NUM_QUERY_VECTORS 2
// #define TOPK 2
// #define DIMENSION 40
// #define BASEFILE "./criteo_tb0"
// #define RESULT_FILE "./check"

#define NUM_DATA_VECTORS 100000
#define NUM_QUERY_VECTORS 100
#define TOPK 128
#define DIMENSION 15
#define BASEFILE "../../../dataset/kdd12/kdd12"
#define RESULT_FILE "../Tree-Nodes-1"

void readSparse(std::string fileName, size_t offset, size_t n, int *indices, float *values,
                unsigned int *markers, size_t bufferlen) {
    std::cout << "[readSparse]" << std::endl;

    /* Fill all the markers with the maximum index for the data, to prevent
       indexing outside of the range. */
    for (size_t i = 0; i <= n; i++) {
        markers[i] = bufferlen - 1;
    }

    std::ifstream file(fileName);
    std::string str;

    size_t ct = 0;                  // Counting the input vectors.
    size_t totalLen = 0;            // Counting all the elements.
    while (std::getline(file, str)) // Get one vector (one vector per line).
    {
        if (ct < offset) { // If reading with an offset, skip < offset vectors.
            ct++;
            continue;
        }
        // Constructs an istringstream object iss with a copy of str as content.
        std::istringstream iss(str);
        // Removes label.
        std::string sub;
        iss >> sub;
        // Mark the start location.
        markers[ct - offset] = std::min(totalLen, bufferlen - 1);
        int pos;
        float val;
        int curLen = 0; // Counting elements of the current vector.
        do {
            std::string sub;
            iss >> sub;
            pos = sub.find_first_of(":");
            if (pos == std::string::npos) {
                continue;
            }
            val = stof(sub.substr(pos + 1, (str.length() - 1 - pos)));
            pos = stoi(sub.substr(0, pos));

            if (totalLen < bufferlen) {
                indices[totalLen] = pos;
                values[totalLen] = val;
            } else {
                std::cout << "[readSparse] Buffer is too small, data is truncated!\n";
                return;
            }
            curLen++;
            totalLen++;
        } while (iss);

        if (ct % 1000000 == 0) {
            printf("Read %lu\n", ct);
        }

        ct++;
        if (ct == (offset + n)) {
            break;
        }
    }
    markers[ct - offset] = totalLen; // Final length marker.
    std::cout << "[readSparse] Read " << totalLen << " numbers, " << ct - offset << " vectors. "
              << std::endl;
}

void readTopK(std::string filename, int numQueries, int k, unsigned int *topK) {
    std::ifstream file(filename);
    std::string str;
    int total = 0;
    while (std::getline(file, str)) {
        std::istringstream iss(str);
        for (int i = 0; i < k; i++) {
            std::string item;
            iss >> item;
            topK[total] = stoi(item);
            total++;
        }
    }
    printf("Total %d\n", total);
    assert(total == numQueries * k);
    printf("Read top %d vectors for %d Queries\n", k, numQueries);
}

float SparseVecMul(int *indicesA, float *valuesA, size_t sizeA, int *indicesB, float *valuesB,
                   size_t sizeB) {

    float result = 0;
    size_t ctA = 0;
    size_t ctB = 0;
    unsigned int iA, iB;

    /* Maximum iteration: nonzerosA + nonzerosB.*/
    while (ctA < sizeA && ctB < sizeB) {
        iA = indicesA[ctA];
        iB = indicesB[ctB];

        if (iA == iB) {
            result += valuesA[ctA] * valuesB[ctB];
            ctA++;
            ctB++;
        } else if (iA < iB) {
            ctA++;
        } else if (iA > iB) {
            ctB++;
        }
    }
    return result;
}

float cosineDist(int *indiceA, float *valA, size_t nonzerosA, int *indiceB, float *valB,
                 size_t nonzerosB) {

    float up = 0;
    float a = 0;
    float b = 0;
    unsigned int startA, endA, startB, endB;

    up = SparseVecMul(indiceA, valA, nonzerosA, indiceB, valB, nonzerosB);
    a = SparseVecMul(indiceA, valA, nonzerosA, indiceA, valA, nonzerosA);
    b = SparseVecMul(indiceB, valB, nonzerosB, indiceB, valB, nonzerosB);
    a = sqrtf(a);
    b = sqrtf(b);
    if (a == 0 || b == 0) {
        return 0;
    }
    return up / (a * b);
}

void print_vec(int *indices, float *vals, unsigned int len) {
    for (int i = 0; i < len; i++) {
        printf("%d:%f ", indices[i], vals[i]);
    }
    printf("\n");
}

void similarityMetric(int *queries_indice, float *queries_val, unsigned int *queries_marker,
                      int *bases_indice, float *bases_val, unsigned int *bases_marker,
                      unsigned int *queryOutputs, unsigned int numQueries, unsigned int topk,
                      int *nList, int nCnt) {

    float *out_avt = new float[nCnt]();

    std::cout << "[similarityMetric] Averaging output. " << std::endl;
    /* Output average. */
    for (size_t i = 0; i < numQueries; i++) {
        size_t startA, endA;
        startA = queries_marker[i];
        endA = queries_marker[i + 1];
        for (size_t j = 0; j < topk; j++) {
            size_t startB, endB;
            startB = bases_marker[queryOutputs[i * topk + j]];
            endB = bases_marker[queryOutputs[i * topk + j] + 1];

            // printf("Query: \n");
            // print_vec(queries_indice + startA, queries_val + startA, endA - startA);

            // printf("Result: \n");
            // print_vec(bases_indice + startB, bases_val + startB, endB - startB);
            float dist = cosineDist(queries_indice + startA, queries_val + startA, endA - startA,
                                    bases_indice + startB, bases_val + startB, endB - startB);
            // printf("Q: %llu V: %u: %f\n", i, queryOutputs[i * topk + j], dist);

            for (int n = 0; n < nCnt; n++) {
                if (j < nList[n])
                    out_avt[n] += dist;
            }
        }
    }

    /* Print results. */
    printf("\nS@k = s_out(s_true): In top k, average output similarity (average "
           "groundtruth similarity). \n");
    for (unsigned int n = 0; n < nCnt; n++) {
        printf("S@%d = %1.3f \n", nList[n], out_avt[n] / (numQueries * nList[n]));
    }
    for (unsigned int n = 0; n < nCnt; n++)
        printf("%d ", nList[n]);
    printf("\n");
    for (unsigned int n = 0; n < nCnt; n++)
        printf("%1.3f ", out_avt[n] / (numQueries * nList[n]));
    printf("\n");
}

void evaluateResults(std::string resultFile) {

    size_t totalNumVectors = NUM_DATA_VECTORS + NUM_QUERY_VECTORS;
    unsigned int *outputs = new unsigned int[NUM_QUERY_VECTORS * TOPK];
    readTopK(resultFile, NUM_QUERY_VECTORS, TOPK, outputs);

    size_t amountToAllocate = totalNumVectors * (size_t)DIMENSION;

    int *sparseIndices = new int[amountToAllocate];

    float *sparseVals = new float[amountToAllocate];

    unsigned int *sparseMarkers = new unsigned int[totalNumVectors + 1];

    readSparse(BASEFILE, 0, totalNumVectors, sparseIndices, sparseVals, sparseMarkers,
               amountToAllocate);

    // const int nCnt = 10;
    // int nList[nCnt] = {1, 10, 20, 30, 32, 40, 50, 64, 100, TOPK};

    const int nCnt = 2;
    int nList[nCnt] = {1, 2};

    std::cout << "\n\n================================\nTOP K TREE\n" << std::endl;

    similarityMetric(sparseIndices, sparseVals, sparseMarkers, sparseIndices, sparseVals,
                     sparseMarkers, outputs, NUM_QUERY_VECTORS, TOPK, nList, nCnt);
}

int main() {

    evaluateResults(RESULT_FILE);

    return 0;
}