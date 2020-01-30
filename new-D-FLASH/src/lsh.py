# http://spark.apache.org/docs/2.2.0/api/python/pyspark.ml.html?highlight=minhash%20lsh
# https://databricks.com/blog/2017/05/09/detecting-abuse-scale-locality-sensitive-hashing-uber-engineering.html
# https://docs.rice.edu/confluence/pages/viewpage.action?pageId=56493483

import pyspark as sc
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import MinHashLSH
from pyspark.sql.functions import col
import time

DIM = 10
TOPK = 128


def extract_sparse_vector(raw_vector):
    contents = raw_vector.split(" ")
    id = contents[0]
    indices = []
    values = []
    for i in contents[1:]:
        k, v = i.split(":")
        indices.append(int(k))
        values.append(float(v))
    return (id, Vectors.sparse(DIM, indices, values))


raw_vectors = sc.textFile("kdd12_data")
print("Data Loaded")
vectors = raw_vectors.map(lambda x: extract_sparse_vector(x)).collect()
print("Data Parsed")

data_frame = sc.createDataFrame(vectors, ["id", "features"])
print("Data Frame Created")
lsh = MinHashLSH(inputCol="features", outputCol="hashes", seed=12049)
print("LSH Created")
lsh_model = lsh.fit(data_frame)
print("LSH Model Fit")

raw_query_vectors = sc.textFile("kdd12_query")
print("Queries Loaded")
query_vectors = raw_vectors.map(lambda x: extract_sparse_vector(x)).collect()
print("Queries Parsed")


start = time.time()
for id, vector in query_vectors:
    lsh_model.approxNearestNeighbors(data_frame, x[1], TOPK).collect()
end = time.time()
print("Queries Complete: " + str(end - start))
