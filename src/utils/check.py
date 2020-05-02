import numpy as np
from sklearn.metrics import roc_auc_score


predictions_file = open("predictions2", "rb")
labels_file = open("criteo_testing_labels", "rb")

predictions = np.fromfile(predictions_file, dtype="uint8")
labels = np.fromfile(labels_file, dtype="uint8")

print(roc_auc_score(predictions, labels))
