#ifndef _BENCHMARKING_H
#define _BENCHMARKING_H

#include <string>

// #define FILE_OUTPUT
// #define EVAL_SIM
// #define WEBSPAM
// #define KDD12
#define CRITEO
// #define FULL_CRITEO
// #define TEST_RUN

#ifdef WEBSPAM

#define NUM_BATCHES 50
#define BATCH_PRINT 10

#define NUM_HASHES 8
#define RANGE_POW 18

#define NUM_TABLES 24
#define RESERVOIR_SIZE 256

#define DIMENSION 4000
#define NUM_DATA_VECTORS 340000
#define NUM_QUERY_VECTORS 10000
// #define NUM_DATA_VECTORS 10000
// #define NUM_QUERY_VECTORS 150
#define MAX_RESERVOIR_RAND 35000
#define TOPK 128
#define AVAILABLE_TOPK 1024

#define CMS_HASHES 4
#define CMS_BUCKET_SIZE 1024

#define BASEFILE "/scratch/ncm5/dataset/webspam/webspam_trigram.svm"

#endif

#ifdef KDD12

#define NUM_BATCHES 50
#define BATCH_PRINT 10

#define NUM_HASHES 4
#define RANGE_POW 18

#define NUM_TABLES 16
#define RESERVOIR_SIZE 256

#define DIMENSION 15
#define NUM_DATA_VECTORS 140000000
#define NUM_QUERY_VECTORS 10000

// #define NUM_DATA_VECTORS 1000000
// #define NUM_QUERY_VECTORS 1000

#define MAX_RESERVOIR_RAND 35000
#define TOPK 128
#define AVAILABLE_TOPK 1024

#define CMS_HASHES 4
#define CMS_BUCKET_SIZE 2048

#define BASEFILE "/scratch/ncm5/dataset/kdd12/kdd12"

#endif

#if defined(CRITEO) || defined(FULL_CRITEO)

#define NUM_BATCHES 50
#define BATCH_PRINT 50

#define NUM_HASHES 4
#define RANGE_POW 19

#define NUM_TABLES 32
#define RESERVOIR_SIZE 256

#define DIMENSION 42
#define NUM_DATA_VECTORS 200000000
#define NUM_QUERY_VECTORS 10000
#define MAX_RESERVOIR_RAND 35000
#define TOPK 128
#define AVAILABLE_TOPK 1024

#define CMS_HASHES 4
#define CMS_BUCKET_SIZE 2048

#define BASEFILE "/scratch/ncm5/dataset/criteo/criteo_partition"

#endif

#ifdef TEST_RUN

#define NUM_BATCHES 1
#define BATCH_PRINT 10

#define NUM_HASHES 4
#define RANGE_POW 7

#define NUM_TABLES 4
#define RESERVOIR_SIZE 128

#define QUERY_PROBES 1
#define HASHING_PROBES 1

#define DIMENSION 15
#define NUM_DATA_VECTORS 140000000
#define NUM_QUERY_VECTORS 1000
#define MAX_RESERVOIR_RAND 35000
#define TOPK 128
#define AVAILABLE_TOPK 1024
#define CMS_HASHES 2
#define CMS_BUCKET_SIZE 128

#define BASEFILE "/scratch/ncm5/dataset/kdd12/kdd12"

#endif

void testing();
void evalWithSimilarity();
void evalWithFileOutput();
void criteo();
void criteoTesting();
void showConfig(std::string dataset, int numVectors, int queries, int nodes, int tables,
                int rangePow, int reservoirSize, int hashes, int cmsHashes, int cmsBucketSize,
                bool cms, bool tree);
void evaluateResults(std::string resultFile);

#endif
