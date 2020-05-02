#include <algorithm>
#include <assert.h>
#include <chrono>
#include <fstream>
#include <iostream>
#include <map>
#include <math.h>
#include <set>
#include <sstream>
#include <stdio.h>

float SparseVecMul(unsigned int *indicesA, float *valuesA, unsigned int sizeA,
                   unsigned int *indicesB, float *valuesB, unsigned int sizeB) {

    float result = 0;
    unsigned int ctA = 0;
    unsigned int ctB = 0;
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

float cosineDist(unsigned int *indiceA, float *valA, unsigned int nonzerosA, unsigned int *indiceB,
                 float *valB, unsigned int nonzerosB) {

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

void readSparse(std::string fileName, unsigned int offset, unsigned int n, unsigned int *indices,
                float *values, unsigned int *markers, int *labels, unsigned int bufferlen) {

    // std::cout << "[readSparse]" << std::endl;

    /* Fill all the markers with the maximum index for the data, to prevent
       indexing outside of the range. */
    for (size_t i = 0; i <= n; i++) {
        markers[i] = bufferlen - 1;
    }

    std::ifstream file(fileName);
    std::string str;

    size_t ct = 0;       // Counting the input vectors.
    size_t totalLen = 0; // Counting all the elements.
    while (std::getline(file, str)) {
        if (ct < offset) { // If reading with an offset, skip < offset
                           // vectors.
            ct++;
            continue;
        }
        // Constructs an istringstream object iss with a copy of str as
        // content.
        std::istringstream iss(str);
        // Removes label.
        std::string sub;
        iss >> sub;
        int label = stoi(sub);
        labels[ct] = label;
        // Mark the start location.
        markers[ct - offset] = std::min(totalLen, (size_t)bufferlen - 1);
        int pos;
        float val;
        unsigned int curLen = 0; // Counting elements of the current vector.
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
                std::cout << "[readSparse] Buffer is too "
                             "small, data is truncated!\n";
                return;
            }
            curLen++;
            totalLen++;
        } while (iss);

        ct++;
        if (ct == (offset + n)) {
            break;
        }
    }
    markers[ct - offset] = totalLen; // Final length marker.
    std::cout << "[readSparse] Read " << totalLen << " numbers, " << ct - offset << " vectors. "
              << std::endl;
}

int main() {

    unsigned int *indices = new unsigned int[4500];
    float *values = new float[4500];
    unsigned int *markers = new unsigned int[101];

    int *labels = new int[100];

    readSparse("./sample", 0, 100, indices, values, markers, labels, 4500);

    float overall_avg = 0.0;

    for (int i = 0; i < 100; i++) {

        float avg = 0.0;
        for (int j = 0; j < 100; j++) {
            if (j == i)
                continue;

            unsigned int start1 = markers[i];
            unsigned int stop1 = markers[i + 1];
            unsigned int start2 = markers[j];
            unsigned int stop2 = markers[j + 1];

            float dist = cosineDist(indices + start1, values + start1, stop1 - start1,
                                    indices + start2, values + start2, stop2 - start2);

            // printf("%f\n", dist);
            avg += dist;
        }

        avg /= 99;

        overall_avg += avg;
    }

    overall_avg /= 100;

    printf("Average sim: %f\n", overall_avg);
}