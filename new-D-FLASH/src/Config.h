#define WEBSPAM

#ifdef WEBSPAM

#define NUM_HASHES 4
#define RANGE_POW 7

#define NUM_TABLES 4
#define RESERVOIR_SIZE 128

#define DIMENSION 4000
#define FULL_DIMENSION 16609143
// #define NUM_DATA_VECTORS 340000
// #define NUM_QUERY_VECTORS 10000
#define NUM_DATA_VECTORS 1000
#define NUM_QUERY_VECTORS 10
#define TOPK 8
#define AVAILABLE_TOPK 128

// #define CMS_HASHES                  8
#define CMS_HASHES 2
// #define CMS_BUCKET_SIZE             2048
#define CMS_BUCKET_SIZE 128

#define BASEFILE "../../../dataset/webspam/webspam_trigram.svm"

#endif
