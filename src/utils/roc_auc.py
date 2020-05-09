import numpy as np
from sklearn.metrics import roc_auc_score

PREDICTIONS_FILE = "predictions2"
LABEL_FILE = "criteo_testing_labels"


predictions_file = open(PREDICTIONS_FILE, "rb")
labels_file = open(LABEL_FILE, "rb")

predictions = np.fromfile(predictions_file, dtype="uint8")
labels = np.fromfile(labels_file, dtype="uint8")

print(roc_auc_score(predictions, labels))
