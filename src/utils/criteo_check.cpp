#include <algorithm>
#include <assert.h>
#include <chrono>
#include <fstream>
#include <iostream>
#include <map>
#include <math.h>
#include <set>
#include <sstream>

using namespace std;

#define BASEFILE "../../../dataset/criteo/criteo_tb"
#define RESULT_FILE "../../results/criteo/Criteo-20"
#define NUM_FILES 20
#define FILE_SIZE 200000000
#define TOPK 128
#define NUM_QUERY 10000

#define BUFFER_SIZE 500000000
#define VEC_LEN 4000

float SparseVecMul(unsigned int *indicesA, float *valuesA, unsigned int sizeA,
                   unsigned int *indicesB, float *valuesB, unsigned int sizeB) {

    float result = 0;
    unsigned int ctA = 0;
    unsigned int ctB = 0;
    unsigned int iA, iB;

    /* Maximum iteration: nonzerosA + nonzerosB.*/
    while (ctA < sizeA && ctB < sizeB) {
        iA = indicesA[ctA];
        iB = indicesB[ctB];

        if (iA == iB) {
            result += valuesA[ctA] * valuesB[ctB];
            ctA++;
            ctB++;
        } else if (iA < iB) {
            ctA++;
        } else if (iA > iB) {
            ctB++;
        }
    }
    return result;
}

float cosineDist(unsigned int *indiceA, float *valA, unsigned int nonzerosA, unsigned int *indiceB,
                 float *valB, unsigned int nonzerosB) {

    float up = 0;
    float a = 0;
    float b = 0;
    unsigned int startA, endA, startB, endB;

    up = SparseVecMul(indiceA, valA, nonzerosA, indiceB, valB, nonzerosB);
    a = SparseVecMul(indiceA, valA, nonzerosA, indiceA, valA, nonzerosA);
    b = SparseVecMul(indiceB, valB, nonzerosB, indiceB, valB, nonzerosB);
    a = sqrtf(a);
    b = sqrtf(b);
    if (a == 0 || b == 0) {
        return 0;
    }
    return up / (a * b);
}

struct sparse_vector {
    unsigned int *indices;
    float *values;
    unsigned int len;
    int label;
};

sparse_vector *make_sparse_vector(int label) {
    sparse_vector *vec = new sparse_vector();
    vec->label = label;
    vec->indices = new unsigned int[VEC_LEN];
    vec->values = new float[VEC_LEN];
    vec->len = 0;
    return vec;
}

void print_sparse_vector(sparse_vector *vec) {
    for (size_t i = 0; i < vec->len; i++) {
        printf("%u:%f ", vec->indices[i], vec->values[i]);
    }
    printf("\n");
}

void print_vec(unsigned int *indices, float *vals, unsigned int len) {
    for (int i = 0; i < len; i++) {
        printf("%u:%f ", indices[i], vals[i]);
    }
    printf("\n");
}

void readSparse(std::string fileName, unsigned int offset, unsigned int n, unsigned int *indices,
                float *values, unsigned int *markers, int *labels, unsigned int bufferlen) {

    // std::cout << "[readSparse]" << std::endl;

    /* Fill all the markers with the maximum index for the data, to prevent
       indexing outside of the range. */
    for (size_t i = 0; i <= n; i++) {
        markers[i] = bufferlen - 1;
    }

    std::ifstream file(fileName);
    std::string str;

    size_t ct = 0;                  // Counting the input vectors.
    size_t totalLen = 0;            // Counting all the elements.
    while (std::getline(file, str)) // Get one vector (one vector per line).
    {
        if (ct < offset) { // If reading with an offset, skip < offset vectors.
            ct++;
            continue;
        }
        // Constructs an istringstream object iss with a copy of str as content.
        std::istringstream iss(str);
        // Removes label.
        std::string sub;
        iss >> sub;
        int label = stoi(sub);
        labels[ct] = label;
        // Mark the start location.
        markers[ct - offset] = std::min(totalLen, (size_t)bufferlen - 1);
        int pos;
        float val;
        unsigned int curLen = 0; // Counting elements of the current vector.
        do {
            std::string sub;
            iss >> sub;
            pos = sub.find_first_of(":");
            if (pos == std::string::npos) {
                continue;
            }
            val = stof(sub.substr(pos + 1, (str.length() - 1 - pos)));
            pos = stoi(sub.substr(0, pos));

            if (totalLen < bufferlen) {
                indices[totalLen] = pos;
                values[totalLen] = val;
            } else {
                std::cout << "[readSparse] Buffer is too small, data is truncated!\n";
                return;
            }
            curLen++;
            totalLen++;
        } while (iss);

        ct++;
        if (ct == (offset + n)) {
            break;
        }
    }
    markers[ct - offset] = totalLen; // Final length marker.
    // std::cout << "[readSparse] Read " << totalLen << " numbers, " << ct - offset << " vectors. "
    //   << std::endl;
}

class Checker {
  private:
    set<unsigned int> *ids;
    unsigned int **sorted_ids;
    unsigned int *lengths;
    map<unsigned int, sparse_vector *> *store;
    string basefile;
    string result_file;
    unsigned int num_per_file;
    unsigned int num_files;
    unsigned int num_queries;
    unsigned int k;

  public:
    Checker(string basefile, string result_file, unsigned int num_per_file, unsigned int num_files,
            unsigned int num_queries, unsigned int k) {
        this->basefile = basefile;
        this->result_file = result_file;
        this->num_per_file = num_per_file;
        this->num_files = num_files;
        this->num_queries = num_queries;
        this->k = k;

        this->ids = new set<unsigned int>[num_files]();
        this->sorted_ids = new unsigned int *[num_files];
        this->lengths = new unsigned int[num_files];
        this->store = new map<unsigned int, sparse_vector *>();
    }

    void read_results() {
        ifstream file(result_file);
        string str;
        unsigned int total = 0;
        while (std::getline(file, str)) {
            istringstream iss(str);
            for (size_t i = 0; i < k; i++) {
                std::string item;
                iss >> item;
                unsigned int val = stoul(item, nullptr, 10);
                unsigned int file_index = val / num_per_file;
                ids[file_index].insert(val);
                total++;
            }
        }
        assert(total == num_queries * k);
        cout << "total " << total << " / " << num_queries * k << endl;
        for (unsigned int file = 0; file < num_files; file++) {
            this->lengths[file] = ids[file].size();
            this->sorted_ids[file] = new unsigned int[lengths[file]];
            unsigned int ct = 0;
            for (unsigned int i : ids[file]) {
                this->sorted_ids[file][ct] = i;
                ct++;
            }
            sort(this->sorted_ids[file], this->sorted_ids[file] + lengths[file]);
            // printf("File %u len %u\n", file, lengths[file]);
        }
        printf("<<Results File Read>>\n");
    }

    void process_file(unsigned int file_index) {
        unsigned int search_index = 0;

        unsigned int current_search = this->sorted_ids[file_index][search_index] % num_per_file;

        char *buffer = new char[BUFFER_SIZE];

        unsigned long long int num_vecs = 0;

        unsigned int next_offset = 0;

        unsigned int len;
        char *line_start;

        string filename(this->basefile);
        // filename.append(to_string(file_index));
        FILE *file = fopen(filename.c_str(), "r");
        if (file == NULL) {
            printf("Error opening file\n");
            return;
        }
        while (num_vecs <= sorted_ids[file_index][lengths[file_index] - 1]) {
            len = fread(buffer, 1, BUFFER_SIZE, file);
            // printf("Len %u\n", len);
            line_start = buffer;

            for (size_t i = 0; i < len; i++) {
                int x = buffer[i];
                if (x == '\n' && num_vecs == current_search) {
                    // printf("current search %u num_vecs %u\n", current_search, num_vecs);

                    buffer[i] = '\0';
                    // printf("%s\n\n\n", line_start);
                    next_offset = i + 1;
                    std::string str(line_start);

                    // printf("%s\n\n", line_start);
                    line_start = buffer + (i + 1);

                    std::istringstream iss(str);
                    std::string sub;
                    iss >> sub;

                    int label = stoi(sub);

                    sparse_vector *vec = make_sparse_vector(label);

                    int pos;
                    float val;
                    unsigned int cur_len = 0;
                    do {
                        std::string sub;
                        iss >> sub;
                        pos = sub.find_first_of(":");
                        if (pos == std::string::npos) {
                            continue;
                        }
                        val = stof(sub.substr(pos + 1, (str.length() - 1 - pos)));
                        pos = stoi(sub.substr(0, pos));
                        // printf("{%u, %f}\n", pos, val);
                        if (cur_len < VEC_LEN) {
                            vec->indices[cur_len] = pos;
                            vec->values[cur_len] = val;
                        } else {
                            std::cout << "[readSparse] Buffer is too small, vector is truncated!\n";
                            return;
                        }
                        cur_len++;
                    } while (iss);

                    vec->len = cur_len;

                    (*store)[sorted_ids[file_index][search_index]] = vec;

                    num_vecs++;
                    search_index++;
                    current_search = sorted_ids[file_index][search_index] % num_per_file;

                    if (num_vecs == sorted_ids[file_index][lengths[file_index] - 1]) {
                        break;
                    }

                } else if (x == '\n') {
                    buffer[i] = '\0';
                    next_offset = i + 1;
                    line_start = buffer + (i + 1);
                    num_vecs++;
                }
            }

            int delta = next_offset - BUFFER_SIZE;
            fseek(file, delta, SEEK_CUR);
        }
        fclose(file);
    }

    void process_files() {
        for (unsigned int i = 0; i < num_files; i++) {
            process_file(i);
        }
    }

    void evaluate() {
        unsigned int *indices = new unsigned int[num_queries * VEC_LEN];
        float *values = new float[num_queries * VEC_LEN];
        unsigned int *markers = new unsigned int[num_queries + 1];
        int *labels = new int[num_queries];

        readSparse(this->basefile, 0, num_queries, indices, values, markers, labels,
                   num_queries * VEC_LEN);

        unsigned int *top_k = new unsigned int[num_queries * k];

        ifstream file(result_file);
        string str;
        unsigned int total = 0;
        while (std::getline(file, str)) {
            istringstream iss(str);
            for (size_t i = 0; i < k; i++) {
                std::string item;
                iss >> item;
                unsigned int val = stoul(item, nullptr, 10);
                top_k[total] = val;
                total++;
            }
        }
        assert(total == num_queries * k);

        // int n_count = 10;
        // int n_counts[] = {1, 10, 20, 30, 32, 40, 50, 64, 100, 128};
        int n_count = 2;
        int n_counts[] = {1, 2};

        float *avgs = new float[10]();

        unsigned int correct_labels = 0;
        for (size_t q = 0; q < num_queries; q++) {
            unsigned int query_correct = 0;
            for (size_t r = 0; r < k; r++) {
                sparse_vector *other = (*store)[top_k[q * this->k + r]];
                if (labels[q] == other->label) {
                    query_correct++;
                }
                unsigned int start = markers[q];
                unsigned int end = markers[q + 1];
                // printf("Query: \n");
                // print_vec(indices + start, values + start, end - start);
                // printf("Result \n");
                // print_sparse_vector(other);
                float dist = cosineDist(other->indices, other->values, other->len, indices + start,
                                        values + start, end - start);
                // printf("Q: %llu V: %u: %f\n", q, top_k[q * this->k + r], dist);

                for (unsigned int i = 0; i < n_count; i++) {
                    if (r < n_counts[i]) {
                        avgs[i] += dist;
                    }
                }
            }
            if (query_correct >= k / 2) {
                correct_labels++;
            }
        }
        for (int i = 0; i < n_count; i++) {
            printf("S@%d = %1.3f \n", n_counts[i], avgs[i] / (n_counts[i] * num_queries));
        }

        printf("Label Accuracy: %u / %u\n", correct_labels, num_queries);
    }
};

int main() {

    Checker c(BASEFILE, RESULT_FILE, FILE_SIZE, NUM_FILES, NUM_QUERY, TOPK);

    c.read_results();

    c.process_file(0);

    c.evaluate();

    return 0;
}