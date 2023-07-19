import numpy as np
import tensorflow as tf
from sklearn import svm
from sklearn.metrics import accuracy_score, confusion_matrix, cohen_kappa_score

# Defining constants
BS = 30

DATASET = "UP"
TRAINING_RATIO = 0.1

# Model and dataset addresses
MODEL_ADDRESS = "hbs_cmcnn_"+DATASET
DATASE_ADDRESST = ["hbs_cmcnn_"+DATASET+"\\image_train_"+DATASET+".npy",
                   "hbs_cmcnn_"+DATASET+"\\label_train_"+DATASET+".npy",
                   "hbs_cmcnn_"+DATASET+"\\image_test_"+DATASET+".npy",
                   "hbs_cmcnn_"+DATASET+"\\label_test_"+DATASET+".npy"]

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

layer_input = tf.keras.Input(shape=(num_band,1))
a = model.get_layer(index=1)(layer_input)
a = model.get_layer(index=2)(a)
a = model.get_layer(index=3)(a)
b = model.get_layer(index=4)(a)
a = model.get_layer(index=5)(b)
a = model.get_layer(index=6)(a)
a = model.get_layer(index=7)(a)
b = model.get_layer(index=8)(a)
c = model.get_layer(index=9)(b)

selection_model = tf.keras.Model(inputs=layer_input, outputs=c)
bs_map = np.zeros(shape=num_band)
for i in range(num_class):
    image_class = image_train[np.where(label_train_int==i)]
    preds = selection_model.predict(image_class)
    preds_avg = np.average(preds, axis=0)
    #thresh_class = np.sort(preds_avg)[-(round((i+1) * BS/num_class) - round((i) * BS/num_class))]
    #class_map = np.where(preds_avg >= thresh_class, 1, 0)
    bs_map = bs_map + preds_avg
thresh_class = np.sort(bs_map)[-BS]
bs_map = np.where(bs_map >= thresh_class, 1, 0)

measurements_train = np.multiply(image_train, np.tile(bs_map, (image_train.shape[0],1))[:,:,np.newaxis])
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

measurements_test = np.multiply(image_test, np.tile(bs_map, (image_test.shape[0],1))[:,:,np.newaxis])
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