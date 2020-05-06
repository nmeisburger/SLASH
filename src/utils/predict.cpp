#include <assert.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdio.h>
#include <string>

#define TOPK_USED 20
#define THRESHOLD 3
#define PATH_TO_KNN_RESULTS "CriteoKNNResults"
#define PREDICTION_OUTPUT_FILE "./predictions"

// For YOGI
#define PATH_TO_DATA_LABELS "/home/ncm5/labels/criteo_labels"
#define PATH_TO_TEST_LABELS "/home/ncm5/criteo_testing_labels"

// #define USE_MPI
#ifdef USE_MPI
#include <mpi.h>
#endif

#define BUFFER_SIZE 2000000000

using namespace std;

void read_topk(string filename, unsigned int num_queries, unsigned int k, unsigned int *topk) {
    std::ifstream file(filename);
    std::string str;
    unsigned int total = 0;
    while (std::getline(file, str)) {
        std::istringstream iss(str);
        for (size_t i = 0; i < k; i++) {
            std::string item;
            iss >> item;
            unsigned int val = stoul(item, nullptr, 10);
            topk[total] = val;
            total++;
        }
    }
    assert(total == num_queries * k);
    printf("Read top %d vectors for %d Queries\n", k, num_queries);
}

void save_labels(string filename, size_t n, string destination_file) {
    uint8_t *labels = new uint8_t[n];

    char *buffer = new char[BUFFER_SIZE];

    FILE *input = fopen(filename.c_str(), "r");
    if (input == NULL) {
        printf("Error: unable to open input file: %s\n", filename.c_str());
        exit(1);
    }

    bool line_start = true;

    size_t current = 0;
    while (current < n) {

        if (fread(buffer, 1, BUFFER_SIZE, input) != BUFFER_SIZE) {
            if (ferror(input)) {
                printf("Error: error in fread. @vector %lu/%lu.\n", current, n);
                exit(1);
            } else if (feof(input)) {
                printf("Error: reached end of file. @vector %lu/%lu.\n", current, n);
                exit(1);
            }
        }

        // printf("___\n%s\n", buffer);

        for (size_t i = 0; i < BUFFER_SIZE - 1; i++) {
            if (line_start && buffer[i + 1] == ' ') {
                // printf("@%lu %c\n", i, buffer[i]);
                if (buffer[i] == '1') {
                    labels[current] = 1;
                    current++;
                } else if (buffer[i] == '0') {
                    labels[current] = 0;
                    current++;
                } else {
                    printf("Error: invalid label located.\n");
                    exit(1);
                }
            }
            line_start = (buffer[i] == '\n');

            if (current >= n) {
                break;
            }
        }

        fseek(input, -1, SEEK_CUR);
    }

    fclose(input);

    FILE *output = fopen(destination_file.c_str(), "w");
    if (output == NULL) {
        printf("Error: unable to open output file: %s\n", destination_file.c_str());
        exit(1);
    }

    size_t nn;
    if ((nn = fwrite(labels, 1, n, output)) != n) {
        printf("HERE %lu\n", nn);
        if (ferror(output)) {
            printf("Error: fwrite failed with error.\n");
        } else {
            printf("Error: fwrite failed.\n");
        }
    }

    fclose(output);
}

void predict(string data_label_file, size_t num_data_files, size_t num_data_per_file,
             string test_label_file, size_t num_test, string result_file, unsigned int k) {

    uint8_t *data_labels = new uint8_t[num_data_files * num_data_per_file];
    uint8_t *preds = new uint8_t[num_test];

    for (size_t i = 0; i < num_data_files; i++) {
        string temp(data_label_file);
        temp.append(to_string(i));
        FILE *file = fopen(temp.c_str(), "r");
        if (file == NULL) {
            printf("Error: unable to open file %s.\n", temp.c_str());
        }
        if (fread(data_labels + i * num_data_per_file, 1, num_data_per_file, file) !=
            num_data_per_file) {
            printf("Error: data label file %s truncated.\n", temp.c_str());
            exit(1);
        }
        fclose(file);
    }

    printf("Read data labels.\n");

    uint8_t *test_labels = new uint8_t[num_test];

    FILE *test = fopen(test_label_file.c_str(), "r");
    if (test == NULL) {
        printf("Error: unable to open file %s.\n", test_label_file.c_str());
    }
    if (fread(test_labels, 1, num_test, test) != num_test) {
        printf("Error: test label file %s truncated.\n", test_label_file.c_str());
        exit(1);
    }
    fclose(test);

    printf("Read test labels.\n");

    unsigned int *results = new unsigned int[num_test * k];

    read_topk(result_file, num_test, k, results);

    unsigned int num_correct = 0;

    for (size_t q = 0; q < num_test; q++) {
        unsigned int ones = 0;
        uint8_t correct_label = test_labels[q];
        for (size_t i = 0; i < k; i++) {
            unsigned int pred = results[k * q + i];
            if (data_labels[pred] == 1) {
                ones++;
            }
        }
        if (ones >= THRESHOLD) {
            preds[q] = 1;
            if (correct_label == 1) {
                num_correct++;
            }
        } else {
            preds[q] = 0;
            if (correct_label == 0) {
                num_correct++;
            }
        }
        // if (matches >= 1) {
        //    num_correct++;
        //    preds[q] = test_labels[q];
        //} else {
        //    preds[q] = (test_labels[q] + 1) % 2;
        //}
        if (q % 10000000 == 0) {
            printf("%lu Complete\n", q);
        }
    }

    FILE *pred_file = fopen(PREDICTION_OUTPUT_FILE, "w");

    if (fwrite(preds, 1, num_test, pred_file) != num_test) {
        printf("Error writing predictions\n");
    }

    printf("Correct: %u / %lu\n", num_correct, num_test);
}

#ifdef USE_MPI

void save_labels_dist(string filename, size_t n, string destination_file) {
    MPI_Init(0, 0);
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    string temp_file(filename);
    if (my_rank < 10) {
        temp_file.append("0");
    }
    temp_file.append(to_string(my_rank));

    string temp_dest(destination_file);
    temp_dest.append(to_string(my_rank));

    printf("Node %d: Processing %s.\n", my_rank, temp_file.c_str());

    save_labels(temp_file, n, temp_dest);

    printf("Node %d: Complete.\n", my_rank);

    MPI_Finalize();
}

#endif

int main() {

    // save_labels_dist("../../../dataset/criteo/criteo_split", 200000000,
    //                  "../../../dataset/criteo/criteo_labels");

    predict(PATH_TO_DATA_LABELS, 20, 200000000, PATH_TO_TEST_LABELS, 150000000, PATH_TO_KNN_RESULTS,
            TOPK_USED);

    return 0;
}
