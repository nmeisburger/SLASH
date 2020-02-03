import sys

num_query = 10000
num_data = 100000

filename = "../../dataset/kdd12/kdd12"

query_filename = "kdd12-query"
data_filename = "kdd12-data"

file = open(filename)
lines = file.readlines()

print("Lines Read")

query = open(query_filename, 'w')
query.writelines(lines[:num_query])
query.close()

print("Query Written")

data = open(data_filename, 'w')
data.writelines(lines[num_query:num_data])
data.close()

print("Data Written")

file.close()

print("Complete")
