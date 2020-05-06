#include <assert.h>
#include <iostream>
#include <stdio.h>

#define TEST_LABEL_FILE "/home/ncm5/criteo_testing_labels"
#define PREDICTION_FILE "predictions2"

#define SIZE 150000000
int main() {
    uint8_t *labels = new uint8_t[SIZE];

    uint8_t *preds = new uint8_t[SIZE];

    FILE *labels_file = fopen(TEST_LABEL_FILE, "r");
    if (labels_file == NULL) {
        return 1;
    }
    if (fread(labels, 1, SIZE, labels_file) != SIZE) {
        return 1;
    }
    fclose(labels_file);

    FILE *preds_file = fopen(PREDICTION_FILE, "r");
    if (preds_file == NULL) {
        return 1;
    }
    if (fread(preds, 1, SIZE, preds_file) != SIZE) {
        return 1;
    }
    fclose(preds_file);

    double true0 = 0.0;
    double pred0 = 0.0;
    double total0 = 0.0;
    double true1 = 0.0;
    double pred1 = 0.0;
    double total1 = 0.0;

    for (int i = 0; i < SIZE; i++) {
        assert(labels[i] == 1 || labels[i] == 0);
        assert(preds[i] == 0 || preds[i] == 1);
        if (labels[i] == 1 && preds[i] == 1) {
            total1++;
            true1++;
            pred1++;
        } else if (labels[i] == 1 && preds[i] == 0) {
            total1++;
            pred0++;
        } else if (labels[i] == 0 && preds[i] == 1) {
            total0++;
            pred1++;
        } else if (labels[i] == 0 && preds[i] == 0) {
            total0++;
            true0++;
            pred0++;
        }
    }

    printf("True %lf\nPred %lf\nTotal %lf\n", true0, pred0, total0);

    printf("Label 0: Precision %lf Recall: %lf\n", true0 / pred0, true0 / total0);

    printf("True %lf\nPred %lf\nTotal %lf\n", true1, pred1, total1);

    printf("Label 1: Precision %lf Recall: %lf\n", true1 / pred1, true1 / total1);

    printf("Check: %lf\n", pred0 + pred1);
    printf("Check: %lf\n", total0 + total1);
    printf("Check: %lf\n", true0 + true1);

    return 0;
}