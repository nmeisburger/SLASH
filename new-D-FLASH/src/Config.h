// #define WEBSPAM

#ifdef WEBSPAM

#define NUM_HASHES 8
#define RANGE_POW 18

#define NUM_TABLES 24
#define RESERVOIR_SIZE 256

#define DIMENSION 4000
#define FULL_DIMENSION 16609143
// #define NUM_DATA_VECTORS 340000
// #define NUM_QUERY_VECTORS 10000
#define NUM_DATA_VECTORS 10000
#define NUM_QUERY_VECTORS 300
#define TOPK 128
#define AVAILABLE_TOPK 128

#define CMS_HASHES 4
#define CMS_BUCKET_SIZE 1024

#define BASEFILE "../../../dataset/webspam/webspam_trigram.svm"

#endif

#define KDD12

#ifdef KDD12

#define NUM_HASHES 8
#define RANGE_POW 18

#define NUM_TABLES 24
#define RESERVOIR_SIZE 256

#define DIMENSION 15
#define FULL_DIMENSION 54686452
#define NUM_DATA_VECTORS 140000000
#define NUM_QUERY_VECTORS 10000
#define TOPK 128
#define AVAILABLE_TOPK 128

#define CMS_HASHES 4
#define CMS_BUCKET_SIZE 2048

#define BASEFILE "../../../dataset/kdd12/kdd12"

#endif
