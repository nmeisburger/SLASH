#include <iostream>
#include <stdio.h>
#include <string>

#define NUM_PARTITIONS 10
#define PARTITION_SIZE 30000000000
#define DATAFILE "../../../dataset/criteo/criteo_tb"

class Splitter {

  private:
    unsigned int num_splits;
    unsigned long long int split_size;
    char *filename;

  public:
    Splitter(unsigned int ns, unsigned long long int ss, char *file) {
        num_splits = ns;
        split_size = ss;
        filename = file;
    }

    void split() {

        std::string convert[] = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"};

        FILE *file = fopen(filename, "r");
        if (file == NULL) {
            return;
        }

        char *buffer = new char[split_size];

        std::cout << "Starting Process\nBuffer Created\n" << std::endl;

        for (unsigned int i = 0; i < num_splits; i++) {
            unsigned long long int len = fread(buffer, 1, split_size, file);
            if (len != split_size) {
                return;
            }

            char *write_loc = buffer;

            std::string output_file(filename);

            output_file.append(convert[i]);

            std::cout << "Creating File " << i << " " << output_file << std::endl;

            FILE *output = fopen(output_file.c_str(), "w");

            if (i > 0) {
                size_t j = 0;
                while (buffer[j] != '\n')
                    j++;
                write_loc = buffer + j + 1;

                len = fwrite(write_loc, 1, split_size - j - 1, output);

            } else {
                len = fwrite(write_loc, 1, split_size, output);
            }

            fclose(output);
        }
        fclose(file);
        printf("File Split\n");
    }

    void partialSplit(unsigned int index) {
        std::string convert[] = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"};

        FILE *file = fopen(filename, "r");
        if (file == NULL) {
            return;
        }

        unsigned long long int seek_offset = index * split_size;
        fseek(file, seek_offset, SEEK_SET);

        char *buffer = new char[split_size];

        std::cout << "Starting Process\nBuffer Created\n" << std::endl;

        for (unsigned int i = index; i < num_splits; i++) {
            unsigned long long int len = fread(buffer, 1, split_size, file);
            if (len != split_size) {
                return;
            }

            char *write_loc = buffer;

            std::string output_file(filename);

            output_file.append(convert[i]);

            std::cout << "Creating File " << i << " " << output_file << std::endl;

            FILE *output = fopen(output_file.c_str(), "w");

            if (i > 0) {
                size_t j = 0;
                while (buffer[j] != '\n')
                    j++;
                write_loc = buffer + j + 1;

                len = fwrite(write_loc, 1, split_size - j - 1, output);

            } else {
                len = fwrite(write_loc, 1, split_size, output);
            }

            fclose(output);
        }
        fclose(file);
        printf("File Split\n");
    }
};

int main() {

    char x[] = DATAFILE;

    Splitter *splitter = new Splitter(NUM_PARTITIONS, PARTITION_SIZE, x);

    printf("Splitter created\n");

    // splitter->split();
    splitter->partialSplit(8);

    return 0;
}
