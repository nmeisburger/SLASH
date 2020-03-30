#ifndef _DATASET_H
#define _DATASET_H

#include "mathUtils.h"
#include <algorithm>
#include <assert.h>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace std;

void readSparse(string fileName, unsigned int offset, unsigned int n, unsigned int *indices,
                float *values, unsigned int *markers, unsigned int bufferlen);

std::streampos readSparse2(std::string fileName, std::streampos fileOffset, unsigned int offset,
                           unsigned int n, unsigned int *indices, float *values,
                           unsigned int *markers, unsigned int bufferlen);

void writeTopK(std::string filename, unsigned int numQueries, unsigned int k, unsigned int *topK);

void writeTopK2(std::string filename, unsigned int numQueries, unsigned int k, unsigned int *topK);

void readTopK(std::string filename, unsigned int numQueries, unsigned int k, unsigned int *topK);

void similarityMetric(unsigned int *queries_indice, float *queries_val,
                      unsigned int *queries_marker, unsigned int *bases_indice, float *bases_val,
                      unsigned int *bases_marker, unsigned int *queryOutputs,
                      unsigned int numQueries, unsigned int topk, unsigned int availableTopk,
                      unsigned int *nList, unsigned int nCnt);
#endif