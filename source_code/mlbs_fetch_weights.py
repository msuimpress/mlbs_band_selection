# Importing the libraries
import tensorflow as tf
import numpy as np

MODEL_ADDRESS = "hbs_model_cnn1_IP_0.1"

model = tf.keras.models.load_model(MODEL_ADDRESS)

alpha = 1000
slope_1 = model.weights[0]
slope_2 = 12

mask_weights = model.weights[1]
mask_weights_sigmoid = -tf.math.log(1. / mask_weights - 1.) / slope_1
mask_weights_sigmoid = tf.sigmoid(slope_1 * mask_weights_sigmoid)

p = tf.math.reduce_mean(mask_weights_sigmoid, axis=1)
beta = (1 - alpha) / (1 - p)
le = tf.cast(tf.greater_equal(p, alpha), tf.float32)

mask_weights_norm = le * mask_weights_sigmoid * alpha / p \
    + (1 - le) * (1 - beta * (1 - mask_weights_sigmoid))

filter_size = mask_weights_norm.shape[1]
t_init = tf.random_uniform_initializer(minval=0, maxval=1)
threshold = tf.constant(t_init(shape=(1, filter_size, 1)))

mask_weights_thresh = tf.sigmoid(slope_2 * (mask_weights_norm-threshold))
