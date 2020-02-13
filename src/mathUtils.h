#ifndef _MATH_UTILS
#define _MATH_UTILS

#include <math.h>

int smallestPow2(int x);
void zCentering(float *values, int n);
unsigned int getLog2(unsigned int x);
float cosineDist(float *A, float *B, unsigned int n);
float cosineDist(unsigned int *indiceA, float *valA, unsigned int nonzerosA, unsigned int *indiceB,
                 float *valB, unsigned int nonzerosB);

float SparseVecMul(unsigned int *indicesA, float *valuesA, unsigned int sizeA,
                   unsigned int *indicesB, float *valuesB, unsigned int sizeB);
float SparseVecMul(unsigned int *indicesA, float *valuesA, unsigned int sizeA, float *B);

#endif