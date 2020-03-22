#include <algorithm>
#include <assert.h>
#include <chrono>
#include <iostream>
#include <sstream>

class Reader {
  private:
    unsigned long long int offset;
    unsigned int block_size;
    char *filename;
    FILE *file;

  public:
    Reader(char *f, unsigned int bsize) {
        filename = f;
        offset = 0;
        file = fopen(filename, "r");
        assert(file != NULL);
        block_size = bsize;
    }

    ~Reader() { fclose(file); }

    void readSparse(unsigned int n, unsigned int *indices, float *values, unsigned int *markers,
                    unsigned int _buffer_len) {

        auto start = std::chrono::system_clock::now();

        unsigned long long int buffer_len = _buffer_len;

        char *buffer = new char[block_size];

        unsigned long long int num_vecs = 0;
        unsigned long long int total_dim = 0;

        unsigned int next_offset = 0;

        unsigned int len;
        char *line_start;
        while (num_vecs < n) {
            len = fread(buffer, 1, block_size, file);

            line_start = buffer;
            for (size_t i = 0; i < len; i++) {
                int x = buffer[i];
                if (x == '\n') {
                    buffer[i] = '\0';
                    next_offset = i + 1;
                    std::string str(line_start);

                    line_start = buffer + (i + 1);

                    std::istringstream iss(str);
                    std::string sub;
                    iss >> sub;

                    markers[num_vecs] = std::min(buffer_len, total_dim);
                    int pos;
                    float val;
                    unsigned int curLen = 0;
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
                        if (total_dim < buffer_len) {
                            indices[total_dim] = pos;
                            values[total_dim] = val;
                        } else {
                            std::cout << "[readSparse] Buffer is too small, data is truncated!\n";
                            return;
                        }
                        curLen++;
                        total_dim++;
                    } while (iss);

                    num_vecs++;
                }
            }

            int delta = next_offset - block_size;
            fseek(file, delta, SEEK_CUR);
        }

        markers[num_vecs] = total_dim;

        auto end = std::chrono::system_clock::now();

        std::chrono::duration<double> elapsed = end - start;

        printf("Read Complete: %fs %llu vectors, total dim: %llu\n", elapsed.count(), num_vecs,
               total_dim);
    }
};

int main() {

    char x[] = "../../../dataset/kdd12/kdd12";

    // Reader *r = new Reader(x, 150);

    // unsigned int *indices = new unsigned int[1500];
    // unsigned int *markers = new unsigned int[104];
    // float *values = new float[1500];
    // r->readSparse("./test", 0, 3, indices, values, markers, 1500);

    // for (int i = 0; i < 3; i++) {
    //     printf("Vector %d: %u - %u\n", i, markers[i], markers[i + 1]);
    //     for (unsigned int j = markers[i]; j < markers[i + 1]; j++) {
    //         printf("(%u, %f)", indices[j], values[j]);
    //     }
    //     printf("\n");
    // }

    // r->readSparse("./test", 0, 4, indices, values, markers, 1500);

    // for (int i = 0; i < 3; i++) {
    //     printf("Vector %d\n", i);
    //     for (unsigned int j = markers[i]; j < markers[i + 1]; j++) {
    //         printf("(%u, %f)", indices[j], values[j]);
    //     }
    //     printf("\n");
    // }

    unsigned int num = 100000;

    Reader *reader = new Reader(x, 2000000);

    unsigned int *indices = new unsigned int[15 * num];
    float *values = new float[15 * num];
    unsigned int *markers = new unsigned int[num + 1];

    reader->readSparse(num, indices, values, markers, 15 * num);

    return 0;
}