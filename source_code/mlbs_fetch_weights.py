# Importing the libraries
import tensorflow as tf
import numpy as np

MODEL_ADDRESS = "hbs_mlbs_IP"

model = tf.keras.models.load_model(MODEL_ADDRESS)

alpha = 777000
BS = 10
slope_1 = model.weights[0]
slope_2 = 12

def force_sparsity(pixel, alpha):
    p = tf.math.reduce_mean(pixel, axis=1)
    beta = (1 - alpha) / (1 - p)
    le = tf.cast(tf.greater_equal(p, alpha), tf.float32)
    return le * pixel * alpha / p + (1 - le) * (1 - beta * (1 - pixel))

mask_weights = model.weights[1]

num_bands = mask_weights.shape[1]
sparsity = BS / num_bands

t_init = tf.random_uniform_initializer(minval=0, maxval=1)
thresh = tf.constant(t_init(shape=(1, num_bands, 1)))

mask_weights_sigm = tf.sigmoid(slope_1 * mask_weights)
mask_weights_norm = force_sparsity(mask_weights_sigm, sparsity)
mask_weights_tresh = tf.sigmoid(slope_2 * (mask_weights_norm - thresh))

weights_selected = mask_weights_tresh > 0.5;
