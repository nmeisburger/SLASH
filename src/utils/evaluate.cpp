#include <algorithm>
#include <assert.h>
#include <fstream>
#include <iostream>
#include <mpi.h>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

/*
 * =============================
 * Parameters
 * =============================
 */
#define DATASET_PATH "/scratch/ncm5/dataset/criteo/criteo_partition"
#define QUERY_PATH "/scratch/ncm5/dataset/criteo/criteo_test_subset"
#define TOPK_PATH "/scratch/ncm5/SLASH/src/scripts/Criteo-20"
#define TEMP_PATH "/scratch/ncm5/dataset/criteo/temp/temp_partition"
#define NUM_QUERY 10000
#define TOPK 128
/*
 * =============================
 */

#define MAX_SIZE 42
#define COUNT 10000
#define MAX_LINE 150000000

typedef unsigned int uint;

using namespace std;

void readSparse(std::string fileName, unsigned int offset, unsigned int n, unsigned int *indices,
                float *values, unsigned int *markers, unsigned int bufferlen) {

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
    std::cout << "[readSparse] Read " << totalLen << " numbers, " << ct - offset << " vectors. "
              << std::endl;
}

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

set<uint> *get_subsample(uint max, uint count) {

    default_random_engine gen;
    uniform_int_distribution<uint> dist(0, max);

    set<uint> *sample = new set<uint>();

    uint remaining = count;

    while (remaining > 0) {
        uint item = dist(gen);
        if (sample->find(item) == sample->end()) {
            remaining--;
            sample->insert(item);
        }
    }
    return sample;
}

void subsample_file(string filename, set<uint> *sample, uint new_size, uint max_len,
                    string output_filename, bool keys, unsigned int offset) {

    uint *ordered = new uint[new_size];

    uint cnt = 0;
    for (uint item : (*sample)) {
        ordered[cnt] = item;
        cnt++;
    }

    sort(ordered, ordered + new_size);

    ifstream input(filename);
    ofstream output(output_filename);

    char *new_buf = new char[(1 << 30)];
    input.rdbuf()->pubsetbuf(new_buf, 1 << 30);

    uint curr = 0;
    uint search = ordered[curr];
    uint lines = 0;

    uint found = 0;

    string line;
    while (getline(input, line)) {
        if (lines == (search - offset)) {
            if (keys) {
                output << std::to_string(search) << " ";
            }
            output << line;
            output << "\n";
            curr++;
            search = ordered[curr];
            found++;
        }
        lines++;
        if (curr >= new_size) {
            break;
        }
    }

    input.close();
    output.close();

    cout << "Selected " << found << "/" << new_size << " vectors from " << filename << "\n" << endl;
}

void readTopK(std::string filename, unsigned int numQueries, unsigned int k, unsigned int *topK) {
    std::ifstream file(filename);
    std::string str;
    unsigned int total = 0;
    while (std::getline(file, str)) {
        std::istringstream iss(str);
        for (size_t i = 0; i < k; i++) {
            std::string item;
            iss >> item;
            unsigned int val = stoul(item, nullptr, 10);
            topK[total] = val;
            total++;
        }
    }
    assert(total == numQueries * k);
    printf("Read top %d vectors for %d Queries\n", k, numQueries);

    file.close();
}

void extract(int file_id, string basefile, string dest, string resultfile, int n_files,
             unsigned int max_size, unsigned int nquery, unsigned int k) {
    string filename(basefile);
    string destname(dest);
    if (file_id < 10) {
        filename.append("0");
        destname.append("0");
    }
    filename.append(to_string(file_id));
    destname.append(to_string(file_id));

    unsigned int *topk = new unsigned int[nquery * k];
    readTopK(resultfile, nquery, k, topk);

    set<unsigned int> this_file;

    for (size_t i = 0; i < nquery * k; i++) {
        if ((topk[i] / max_size) == file_id) {
            if (this_file.find(topk[i]) == this_file.end()) {
                this_file.insert(topk[i]);
            }
        }
    }

    subsample_file(filename, &this_file, this_file.size(), max_size, destname, true,
                   max_size * file_id);
}

struct Vec {
    unsigned int *indices;
    float *values;
    unsigned int len;
    unsigned int id;
    Vec(string line) {

        this->indices = new unsigned int[MAX_SIZE];
        this->values = new float[MAX_SIZE];
        this->len = 0;
        istringstream stream(line);
        string item;
        stream >> item;
        unsigned int key = stoul(item, nullptr, 10);
        this->id = key;
        string lbl;
        stream >> lbl;

        int pos;
        float val;
        do {
            std::string sub;
            stream >> sub;
            pos = sub.find_first_of(":");
            if (pos == std::string::npos) {
                continue;
            }
            val = stof(sub.substr(pos + 1, (sub.length() - 1 - pos)));
            pos = stoi(sub.substr(0, pos));
            this->indices[len] = pos;
            this->values[len] = val;
            this->len++;
        } while (stream);
    }
};

void evaluate(string datafile, string queryfile, string resultfile, int n_files, unsigned int k,
              unsigned int nquery) {
    unsigned int *topk = new unsigned int[nquery * k];
    readTopK(resultfile, nquery, k, topk);
    printf("TOPK Read\n");

    unordered_map<unsigned int, Vec *> cache;

    for (int file = 0; file < n_files; file++) {
        string filename(datafile);
        if (file < 10) {
            filename.append("0");
        }
        filename.append(to_string(file));

        ifstream data(filename);
        string line;

        while (getline(data, line)) {
            Vec *v = new Vec(line);
            cache.insert({v->id, v});
        }

        data.close();
        printf("Loaded datafile %d\n", file);
    }

    /*
    for (auto e : cache) {
        printf("Cache vector: %u: ", e.first);
        printf(" id: %u ", e.second->id);
        for (int i = 0; i < e.second->len; i++) {
            printf("(%u %f) ", e.second->indices[i], e.second->values[i]);
        }
        printf("\n");
    }
    */

    unsigned int *markers = new unsigned int[nquery + 1];
    unsigned int *indices = new unsigned int[nquery * MAX_SIZE];
    float *values = new float[nquery * MAX_SIZE];
    readSparse(queryfile, 0, nquery, indices, values, markers, nquery * MAX_SIZE);

    // unsigned int ns[] = {1, 2, 3, 5};
    unsigned int ns[] = {1, 1, 2, 2};
    float *averages = new float[4];
    for (size_t i = 0; i < nquery; i++) {
        unsigned int startA, endA;
        startA = markers[i];
        endA = markers[i + 1];
        for (unsigned int j = 0; j < k; j++) {
            Vec *b = cache[topk[i * k + j]];
            float dist = cosineDist(indices + startA, values + startA, endA - startA, b->indices,
                                    b->values, b->len);
            for (unsigned int n = 0; n < 4; n++) {
                if (j < ns[n])
                    averages[n] += dist;
            }
        }
    }

    printf("\nS@k = s_out(s_true): In top k, average output similarity (average "
           "groundtruth similarity). \n");
    for (unsigned int n = 0; n < 4; n++) {
        printf("S@%d = %1.3f \n", ns[n], averages[n] / (nquery * ns[n]));
    }
    printf("\n");
}

// For subsampling query file
// int main() {
// set<uint> *s = get_subsample(MAX_LINE, COUNT);
// printf("Subsample generated.\n");
// subsample_file("./criteo_tb.t", s, COUNT, MAX_LINE, "criteo_test_subset", false, 0);
// printf("New file created.\n");
// }

int main() {
    // extract(0, "./test", "here", "./topk", 2, 5, 2, 4);
    // extract(1, "./test", "here", "./topk", 2, 5, 2, 4);
    // evaluate("./here", "query", "topk", 2, 2, 2);

    MPI_Init(0, 0);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    extract(rank, DATASET_PATH, TEMP_PATH, TOPK_PATH, 20, 200000000, NUM_QUERY, TOPK);

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
        evaluate(TEMP_PATH, QUERY_PATH, TOPK_PATH, 20, TOPK, NUM_QUERY);
    }

    MPI_Finalize();
    return 0;
}