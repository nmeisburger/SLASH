#ifndef _DATASET
#define _DATASET

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

void read_sparse(std::string fileName, unsigned int offset, unsigned int n, unsigned int *indices,
                 float *values, unsigned int *markers, unsigned int bufferlen);

void writeTopK(std::string filename, int numQueries, int k, unsigned int *topK);

void readTopK(std::string filename, int numQueries, int k, unsigned int *topK);

void similarityMetric(unsigned int *queries_indice, float *queries_val,
                      unsigned int *queries_marker, unsigned int *bases_indice, float *bases_val,
                      unsigned int *bases_marker, unsigned int *queryOutputs,
                      unsigned int numQueries, unsigned int topk, unsigned int availableTopk,
                      int *nList, int nCnt);

#endif