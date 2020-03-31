#include <iostream>
#include <string>

#define NUM_PARTITIONS 10
#define ITERATIONS 3
#define NUM_SKIP 0
#define PARTITION_SIZE 34000000000
#define DATAFILE "../../../dataset/criteo/criteo_tb"

// #define NUM_PARTITIONS 3
// #define ITERATIONS 2
// #define NUM_SKIP 3
// #define PARTITION_SIZE 150
// #define DATAFILE "../../../dataset/kdd12/kdd12"

class Splitter {

  private:
    unsigned int num_splits, iterations, skipped_splits;
    unsigned long long int split_size;
    char *filename;

  public:
    Splitter(unsigned int ns, unsigned int iters, unsigned int skipped, unsigned long long int ss,
             char *file) {
        num_splits = ns;
        iterations = iters;
        skipped_splits = skipped;
        split_size = ss;
        filename = file;
    }

    void split() {
        FILE *file = fopen(filename, "r");
        if (file == NULL) {
            printf("Error: fopen\n");
            return;
        }

        char *buffer = new char[split_size];

        unsigned long long int offset = skipped_splits * split_size * iterations;

        fseek(file, offset, SEEK_SET);

        std::cout << "Starting Process\nBuffer Created\n" << std::endl;

        for (unsigned int i = 0; i < num_splits; i++) {
            unsigned long long int len = fread(buffer, 1, split_size, file);
            // if (len != split_size) {
            //     printf("Error: fread\n");
            //     return;
            // }

            char *write_loc = buffer;

            std::string output_file(filename);

            output_file.append(std::to_string(i + skipped_splits));

            std::cout << "Creating File " << i << " " << output_file << std::endl;

            FILE *output = fopen(output_file.c_str(), "w");

            if (output == NULL) {
                printf("Error: file creation\n");
                return;
            }

            if (i > 0 || skipped_splits > 0) {
                size_t j = 0;
                while (buffer[j] != '\n')
                    j++;
                write_loc = buffer + j + 1;

                len = fwrite(write_loc, 1, split_size - j - 1, output);
                // if (len != (split_size - j - 1)) {
                //     printf("Error: fwrite\n");
                //     return;
                // }
            } else {
                len = fwrite(write_loc, 1, split_size, output);
                // if (len != split_size) {
                //     printf("Error: fwrite\n");
                //     return;
                // }
            }

            for (unsigned int iii = 1; iii < iterations; iii++) {
                len = fread(buffer, 1, split_size, file);
                // if (len != split_size) {
                //     printf("Error: fread\n");
                //     return;
                // }
                len = fwrite(buffer, 1, split_size, output);
                // if (len != split_size) {
                //     printf("Error: fwrite\n");
                //     return;
                // }
            }

            fclose(output);
            printf("Partition %d Complete\n", i);
        }
        fclose(file);
        printf("File Split\n");
    }
};

int main() {

    char x[] = DATAFILE;

    Splitter *splitter = new Splitter(NUM_PARTITIONS, ITERATIONS, NUM_SKIP, PARTITION_SIZE, x);

    printf("Splitter created\n");

    splitter->split();

    return 0;
}