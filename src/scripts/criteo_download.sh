curl -o criteo_tb.svm.tar.gz https://s3-us-west-2.amazonaws.com/criteo-public-svm-data/criteo_tb.svm.tar.gz

tar -zxvf criteo_tb.svm.tar.gz

split -l 200000000 -d criteo_tb criteo_partition --verbose