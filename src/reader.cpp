#include "reader.h"

void Reader::readSparse(unsigned int n, unsigned int *indices, float *values, unsigned int *markers,
                        unsigned int _buffer_len) {

    auto start = std::chrono::system_clock::now();

    unsigned long long int buffer_len = _buffer_len;

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
                if (num_vecs == n) {
                    break;
                }
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