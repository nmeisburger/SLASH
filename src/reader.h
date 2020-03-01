#ifndef _READER_H
#define _READER_H

#include <algorithm>
#include <assert.h>
#include <chrono>
#include <iostream>
#include <sstream>
#include <string>

class Reader {
  private:
    unsigned long long int offset;
    unsigned int block_size;
    const char *filename;
    FILE *file;

    char *buffer;

  public:
    Reader(const char *f, unsigned int bsize) {
        filename = f;
        offset = 0;
        file = fopen(filename, "r");
        assert(file != NULL);
        block_size = bsize;
        buffer = new char[block_size];
    }

    ~Reader() {
        fclose(file);
        delete[] buffer;
    }

    void readSparse(unsigned int n, unsigned int *indices, float *values, unsigned int *markers,
                    unsigned int _buffer_len);
};

#endif