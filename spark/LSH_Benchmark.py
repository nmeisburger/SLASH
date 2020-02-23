# http://spark.apache.org/docs/2.2.0/api/python/pyspark.ml.html?highlight=minhash%20lsh
# https://databricks.com/blog/2017/05/09/detecting-abuse-scale-locality-sensitive-hashing-uber-engineering.html
# https://docs.rice.edu/confluence/pages/viewpage.action?pageId=56493483

import time
from pyspark.sql.functions import col
from pyspark.ml.feature import MinHashLSH
from pyspark.ml.linalg import Vectors
from pyspark import SparkConf, SparkContext  # Skip on AWS
from pyspark.sql import SQLContext

conf = SparkConf().setAppName("KDD12 Benchmark")  # Skip on AWS
sc = SparkContext(conf=conf)  # Skip on AWS

sqlCtx = SQLContext(sc)

# KDD-12
DIM = 54686452
NUM_DATA_VECTORS = 90000
NUM_QUERY_VECTORS = 10000
TOPK = 128
DATA_FILE = "s3://flash-kdd12/kdd12-data"
QUERY_FILE = "s3://flash-kdd12/kdd12-query"

# Webspam
DIM = 16609143
NUM_DATA_VECTORS = 9900
NUM_QUERY_VECTORS = 100
DATA_FILE = "s3://flash-kdd12/webspam-data"
QUERY_FILE = "s3://flash-kdd12/webspam-query"
TOPK = 64


def extract_sparse_vector(raw_vector):
    contents = raw_vector.split(" ")
    id = contents[0]
    indices = []
    values = []
    if len(contents) > 1:
        for i in contents[1:]:
            k, v = i.split(":")
            indices.append(int(k))
            values.append(float(v))
    return (id, Vectors.sparse(DIM, indices, values))


raw_data_vectors = sc.textFile(DATA_FILE)
data_vectors = raw_data_vectors.map(
    lambda x: extract_sparse_vector(x)).collect()

data_frame = sqlCtx.createDataFrame(data_vectors, ["id", "features"])
lsh = MinHashLSH(inputCol="features", outputCol="hashes", seed=12049)
lsh_model = lsh.fit(data_frame)
lsh_model.transform(data_frame).head()

raw_query_vectors = sc.textFile(QUERY_FILE)
query_vectors = raw_query_vectors.map(
    lambda x: extract_sparse_vector(x)).collect()


results = []

start = time.time()
query_vectors.map(lambda query: lsh_model.approxNearestNeighbors(
    data_frame, query[1], TOPK)).collect()
end = time.time()
print("Queries Complete: " + str(end - start))


def run():
    start = time.time()
    for id, vector in query_vectors:
        topk = lsh_model.approxNearestNeighbors(
            data_frame, vector, TOPK).collect()
        results.append((vector, topk))
    end = time.time()
    print("Queries Complete: " + str(end - start))


# Similarity Evaluation

def avgSimilarity(query, topk):
    query_mag = float(query.dot(query))
    sim = 0.0
    count = 0
    for output in topk:
        sparse_vector = output[1]
        dot = float(sparse_vector.dot(query))
        vec_mag = float(sparse_vector.dot(sparse_vector))
        denom = (query_mag * vec_mag)
        if (denom > 0):
            sim += (dot / denom)
            count += 1
    if count == 0:
        return 0
    return sim / count


query_results = sc.parallelize(results)
avg_sims = query_results.map(lambda x: avgSimilarity(x[0], x[1]))
sim = avg_sims.reduce(lambda a, b: a + b) / NUM_QUERY_VECTORS
