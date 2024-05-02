import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix, cohen_kappa_score

# Defining constants
BS = 30

DATASET = "IP"

# Model and dataset addresses
MODEL_ADDRESS = "hbs_allbands_"+DATASET
DATASE_ADDRESST = ["hbs_allbands_"+DATASET+"\\image_test_"+DATASET+".npy",
                   "hbs_allbands_"+DATASET+"\\label_test_"+DATASET+".npy"]

# Loading the training dataset.
model = tf.keras.models.load_model(MODEL_ADDRESS)

# Loading the test dataset.
image_test = np.load(DATASE_ADDRESST[0])
label_test = np.load(DATASE_ADDRESST[1])
label_test_int = np.argmax(label_test, axis=1)

num_band = image_test.shape[1]             # Number of bandwidths
num_class = label_test.shape[1]

predictions = model.predict(image_test)
predictions_int = np.argmax(predictions, axis=1)

# Accuracy metrics
accuracy_overall = 100*accuracy_score(label_test_int, predictions_int)
matrix_conf = confusion_matrix(label_test_int, predictions_int)
accuracy_class = 100*matrix_conf.diagonal()/matrix_conf.sum(axis=1)
kappa_score = 100*cohen_kappa_score(label_test_int, predictions_int)

print("Accuracy per class:")
for i in range(num_class):
    print("Class {c} : {ac:.2f}".format(c=i+1, ac=accuracy_class[i]))
print("Overall classification accuracy score: {oca:.2f}".format(oca=accuracy_overall))
print("Average classification accuracy score: {aca:.2f}".format(aca=np.sum(accuracy_class)/num_class))
print("Kappa coefficient:  {kc:.2f}".format(kc=kappa_score))