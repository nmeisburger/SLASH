#define WEBSPAM

#ifdef WEBSPAM

#define NUM_HASHES 8
#define RANGE_POW 18

#define NUM_TABLES 12
#define RESERVOIR_SIZE 128

#define DIMENSION 4000
#define FULL_DIMENSION 16609143
// #define NUM_DATA_VECTORS 340000
// #define NUM_QUERY_VECTORS 10000
#define NUM_DATA_VECTORS 1000
#define NUM_QUERY_VECTORS 30
#define TOPK 128
#define AVAILABLE_TOPK 128

#define CMS_HASHES 4
#define CMS_BUCKET_SIZE 1024

#define BASEFILE "../../../dataset/webspam/webspam_trigram.svm"

#endif
