import sys

num_query = 100
num_data = 10000

filename = "../../dataset/webspam/webspam_trigram.svm"

query_filename = "webspam-query"
data_filename = "webspam-data"

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
