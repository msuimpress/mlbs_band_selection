import numpy as np
import tensorflow as tf
from sklearn import svm
from sklearn.metrics import accuracy_score, confusion_matrix, cohen_kappa_score

# Defining constants
BS = 30

DATASET = "IP"
TRAINING_RATIO = 0.1

# Model and dataset addresses
MODEL_ADDRESS = "hbs_bhcnn_"+DATASET
DATASE_ADDRESST = ["hbs_bhcnn_"+DATASET+"\\image_train_"+DATASET+".npy",
                   "hbs_bhcnn_"+DATASET+"\\label_train_"+DATASET+".npy",
                   "hbs_bhcnn_"+DATASET+"\\image_test_"+DATASET+".npy",
                   "hbs_bhcnn_"+DATASET+"\\label_test_"+DATASET+".npy"]

# Loading the training dataset.
model = tf.keras.models.load_model(MODEL_ADDRESS)
image_train = np.load(DATASE_ADDRESST[0])
label_train = np.load(DATASE_ADDRESST[1])
label_train_int = np.argmax(label_train, axis=1)

num_data = image_train.shape[0]  
num_band = image_train.shape[1]             # Number of bandwidths
num_class = label_train.shape[1]
number_class_label = []; class_weights = []
for i in range(num_class):
    number_class_label.append(len(np.where(label_train_int == i)[0]))
    class_weights.append(num_data/number_class_label[i])
class_weights = np.array(class_weights)
class_weights_norm = class_weights / np.sum(class_weights)
class_weights_norm = np.sqrt(class_weights_norm)
#class_weights_norm = np.square(class_weights_norm)
class_weights_dic = {}
for i in range(num_class):
    class_weights_dic[i] = 100*class_weights_norm[i]
    
# Separating the measurement layers from the overall model.
layer_input = tf.keras.Input(shape=(num_band,1))
layer_treshold_mask = model.get_layer("thresh_mask")
layer_proxy_data = model.get_layer("proxy_data")

# Defining a new model for only measurement.
x = layer_treshold_mask(layer_input)
y = layer_proxy_data([layer_input, x])
measurement_model = tf.keras.Model(inputs = layer_input, outputs = y)

measurements_train = measurement_model.predict(image_train)
measurements_train_zeroed = np.where(measurements_train < 0.0001, 0, measurements_train)[:,:,0]
non_zero_index = np.where(measurements_train_zeroed[0] != 0)[0]
measurements_train_selected = measurements_train_zeroed[:,non_zero_index]

print("Number of training data points: ", measurements_train_selected.shape[0])
print("Number of classes: ", num_class)
print("Number of total channels: ", num_band)
print("Number of selected channels: ", measurements_train_selected.shape[1])

classifier = svm.SVC(C=1.0, kernel="poly", gamma='scale', 
                     probability=True, class_weight = class_weights_dic, 
                     decision_function_shape = "ovr")
classifier.fit(measurements_train_selected, label_train_int)

# Loading the test dataset.
image_test = np.load(DATASE_ADDRESST[2])
label_test = np.load(DATASE_ADDRESST[3])
label_test_int = np.argmax(label_test, axis=1)

measurements_test = measurement_model.predict(image_test)
measurements_test_zeroed = np.where(measurements_test < 0.0001, 0, measurements_test)[:,:,0]
measurements_test_selected = measurements_test_zeroed[:,non_zero_index]

predictions = classifier.predict_proba(measurements_test_selected)
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