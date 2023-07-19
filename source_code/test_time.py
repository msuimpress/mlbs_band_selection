import numpy as np
import tensorflow as tf
import time

# Defining constants
DATASET = "IP"

# Model and dataset addresses
MODEL_ADDRESS = "hbs_mlbs_"+DATASET
DATASE_ADDRESST = ["hbs_mlbs_"+DATASET+"\\image_train_"+DATASET+".npy",
                   "hbs_mlbs_"+DATASET+"\\label_train_"+DATASET+".npy",
                   "hbs_mlbs_"+DATASET+"\\image_test_"+DATASET+".npy",
                   "hbs_mlbs_"+DATASET+"\\label_test_"+DATASET+".npy"]

# Loading the training dataset.
model = tf.keras.models.load_model(MODEL_ADDRESS)
image_train = np.load(DATASE_ADDRESST[0])
label_train = np.load(DATASE_ADDRESST[1])
label_train_int = np.argmax(label_train, axis=1)

start_time = time.time()
#model.predict(image_train[0][np.newaxis])
model.predict([image_train[0][np.newaxis], image_train[0][np.newaxis]])
end_time = time.time()

# Calculate elapsed time
elapsed_time = end_time - start_time
print("Elapsed time: ", elapsed_time) 