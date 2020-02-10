#include <algorithm>
#include <assert.h>
#include <fstream>
#include <iostream>
#include <math.h>
#include <sstream>

#define NUM_DATA_VECTORS 30000000
#define NUM_QUERY_VECTORS 10000
#define TOPK 128
#define DIMENSION 265
#define BASEFILE "./wiki_hashes"
#define RESULT_FILE "./WikiDump-6"

void readSparse(std::string fileName, size_t offset, size_t n, int *indices, float *values,
                size_t *markers, size_t bufferlen) {
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

void similarityMetric(int *queries_indice, float *queries_val, size_t *queries_marker,
                      int *bases_indice, float *bases_val, size_t *bases_marker,
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
            float dist = cosineDist(queries_indice + startA, queries_val + startA, endA - startA,
                                    bases_indice + startB, bases_val + startB, endB - startB);
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
    int *sparseMarkers = new int[totalNumVectors + 1];

    readSparse(BASEFILE, 0, totalNumVectors, sparseIndices, sparseVals, sparseMarkers,
               amountToAllocate);

    const int nCnt = 10;
    int nList[nCnt] = {1, 10, 20, 30, 32, 40, 50, 64, 100, TOPK};

    std::cout << "\n\n================================\nTOP K TREE\n" << std::endl;

    similarityMetric(sparseIndices, sparseVals, sparseMarkers, sparseIndices, sparseVals,
                     sparseMarkers, outputs, NUM_QUERY_VECTORS, TOPK, nList, nCnt);
}

int main() {

    evaluateResults(RESULT_FILE);

    return 0;
}