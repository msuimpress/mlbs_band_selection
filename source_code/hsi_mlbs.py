# Importing the libraries
import tensorflow as tf
import numpy as np
import os
import scipy.io as sio
from math import ceil

# Initializing constants
DATASET_NAME = 'UP'
K = 10
K_NUM = 1

TRAINING_RATIO = 0.1                        # The ratio of the training dataset over the whole points.
pmask_slope = 5
sample_slope = 10
BS = 30

EPOCHS = 150
LR = 0.001
BATCH_SIZE = 16

""" Data preprocessing functions """
def read_data(dataset_name):                        # Loads Dataset
    path = os.getcwd()
    if dataset_name == 'IP':
        image = sio.loadmat(path+"\\datasets\\Indian_pines_corrected.mat")["indian_pines_corrected"]
        label = sio.loadmat(path+"\\datasets\\Indian_pines_gt.mat")["indian_pines_gt"]
    elif dataset_name == 'UP':
        image = sio.loadmat(path+'\\datasets\\PaviaU.mat')['paviaU']
        label = sio.loadmat(path+'\\datasets\\PaviaU_gt.mat')['paviaU_gt']
    image = np.float64(image)
    label = np.array(label).astype(float)
    return image, label

def normalize_dataset(data):                        # Dataset Normalization
    max_val = np.amax(data, axis=(0,1,2))
    min_val = np.amin(data, axis=(0,1,2))
    data_norm = (data - min_val) / (max_val - min_val)
    return data_norm

def separate_train_test(data, labels, p):   # Separates the dataset into training and test sets
    c = int(labels.max())
    x = np.array([], dtype=float).reshape(-1, data.shape[2])
    xb = []
    x_loc1 = []
    x_loc2 = []
    x_loc = []
    y = np.array([], dtype=float).reshape(-1, data.shape[2])
    yb = []
    y_loc1 = []
    y_loc2 = []
    y_loc = []
    for i in range(1, c+1):
        loc1, loc2 = np.where(labels == i)
        num = len(loc1)
        order = np.random.permutation(range(num))
        loc1 = loc1[order]
        loc2 = loc2[order]
        num1 = int(np.round(num*p))
        x = np.vstack([x, data[loc1[:num1], loc2[:num1], :]])
        y = np.vstack([y, data[loc1[num1:], loc2[num1:], :]])
        xb.extend([i]*num1)
        yb.extend([i]*(num-num1))
        x_loc1.extend(loc1[:num1])
        x_loc2.extend(loc2[:num1])
        y_loc1.extend(loc1[num1:])
        y_loc2.extend(loc2[num1:])
        x_loc = np.vstack([x_loc1, x_loc2])
        y_loc = np.vstack([y_loc1, y_loc2])
    return x, xb, x_loc, y, yb, y_loc

def one_hot(lable,class_number):            # One-hot converter
    one_hot_array = np.zeros([len(lable),class_number])
    for i in range(len(lable)):
        one_hot_array[i,int(lable[i]-1)] = 1
    return one_hot_array

def disorder(X,Y,loc):
    index_train = np.arange(X.shape[0])
    np.random.shuffle(index_train)
    X = X[index_train, :]
    Y = Y[index_train, :]
    loc=loc[:,index_train]
    return X,Y,loc

def windowFeature(data, loc, w):
    size = np.shape(data)
    data_expand = np.zeros((int(size[0]+w-1),int(size[1]+w-1),size[2]))
    newdata = np.zeros((len(loc[0]), w, w,size[2]))
    for j in range(size[2]):    
        data_expand[:,:,j] = np.lib.pad(data[:,:,j], ((int(w / 2), int(w / 2)), (int(w / 2),int(w / 2))), 'symmetric')
        newdata[:,:,:,j] = np.zeros((len(loc[0]), w, w))
        for i in range(len(loc[0])):
            loc1 = loc[0][i]
            loc2 = loc[1][i]
            f = data_expand[loc1:loc1 + w, loc2:loc2 + w,j]
            newdata[i, :, :,j] = f
    return newdata

"""
Functions of the probability mask as described in the paper.
Each weight is passed into a sigmoid to convert them to a probability representation.
"""
class ProbMask(tf.keras.layers.Layer):
    def __init__(self, slope, filter_size, **kwargs):
        self.slope = tf.Variable(slope, dtype=tf.float32)
        self.P = filter_size
        
        w_init = tf.random_uniform_initializer(minval=0, maxval=1)
        self.w = tf.Variable(w_init(shape=(1, filter_size, 1)))
        self.w = tf.Variable(- tf.math.log(1. / self.w - 1.) / self.slope)
        super(ProbMask, self).__init__(**kwargs)
        
    def build(self, input_shape):
        super(ProbMask, self).build(input_shape)
        
    def call(self, input_tensor):
        weights = self.w 
        return weights
        #return tf.sigmoid(self.slope * weights)

    def compute_output_shape(self, input_shape):
        lst = list(input_shape)
        lst[-1] = 1
        return tuple(lst)
    
# The layer that rescales the weights given a sparsity level in probability map.
class RescaleProbMask(tf.keras.layers.Layer):
    def __init__(self, sparsity, **kwargs):
        self.alpha = tf.constant(sparsity, dtype=tf.float32)
        super(RescaleProbMask, self).__init__(**kwargs)

    def build(self, input_shape):
        self.num = input_shape[2]
        super(RescaleProbMask, self).build(input_shape)

    def call(self, input_tensor):
        prob_map = self.force_sparsity(input_tensor, alpha = self.alpha)
        return prob_map

    def force_sparsity(self, pixel, alpha):
        p = tf.math.reduce_mean(pixel, axis=1)
        beta = (1 - alpha) / (1 - p)
        le = tf.cast(tf.greater_equal(p, alpha), tf.float32)
        return le * pixel * alpha / p + (1 - le) * (1 - beta * (1 - pixel))

class ThresholdRandomMask(tf.keras.layers.Layer):
    def __init__(self, slope = 12, **kwargs):
        self.slope = None
        if slope is not None:
            self.slope = tf.Variable(slope, dtype=tf.float32) 
        super(ThresholdRandomMask, self).__init__(**kwargs)

    def build(self, input_shape):
        filter_size = input_shape[1]
        t_init = tf.random_uniform_initializer(minval=0, maxval=1)
        threshold = tf.constant(t_init(shape=(1, filter_size, 1)))
        self.thresh = threshold
        super(ThresholdRandomMask, self).build(input_shape)

    def call(self, inputs):
        #thresh = tf.zeros_like(inputs)
        if self.slope is not None:
            return tf.sigmoid(self.slope * (inputs-self.thresh))
        else:
            return inputs > self.thresh

    def compute_output_shape(self, input_shape):
        return input_shape[0]

# The layer that performs masking operation.
class UnderSample(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(UnderSample, self).__init__(**kwargs)

    def build(self, input_shape):
        super(UnderSample, self).build(input_shape)

    def call(self, inputs):
        input_pix = inputs[0]
        mask = inputs[1]
        image_masked = tf.multiply(input_pix, mask)
        return image_masked

# Learning rate scheduler 
def scheduler(epoch, lr):
    if epoch == 50:
        lr = lr / 10
    if epoch == 100:
        lr = lr / 10
    return lr

data_ori, labels_ori = read_data(DATASET_NAME)
data_norm = normalize_dataset(data_ori)

train_x, train_y, train_loc, test_x, test_y, test_loc = separate_train_test(data_norm, labels_ori, TRAINING_RATIO)

num_classification = int(np.max(labels_ori))

X_train = windowFeature(data_norm, train_loc, 1)
X_test = windowFeature(data_norm, test_loc, 1)

X_train=np.squeeze(X_train, axis=2)
X_test=np.squeeze(X_test, axis=2)

X_train.shape[-1]

Y_train = one_hot(train_y,num_classification )
Y_test = one_hot(test_y,num_classification )

X_train,Y_train,train_loc=disorder(X_train,Y_train,train_loc)
X_test,Y_test,test_loc=disorder(X_test,Y_test,test_loc)

X_train = np.transpose(X_train, axes=(0,2,1))
X_test = np.transpose(X_test, axes=(0,2,1))

print('Shape of the training images: ', X_train[0].shape)
print('Shape of the test images: ', X_test[0].shape)
print('Number of the train samples', X_train.shape[0])
print('Number of the test samples', X_test.shape[0])
print('Number of the classes',num_classification)

"Code for model creation, compilation and training."
training_input_num = X_train.shape[0]
num_bands = X_train.shape[1]
input_image = tf.keras.Input(shape=(num_bands,1), name='input_image')

prob_mask_tensor = ProbMask(slope=pmask_slope,
                            filter_size = num_bands,name='prob_mask')(input_image)
thresh_tensor = RescaleProbMask(sparsity = BS/num_bands, name='rescaled_mask')(prob_mask_tensor) 
tensor_mask = ThresholdRandomMask(slope = sample_slope, name='sampled_mask')(thresh_tensor) 

last_tensor = UnderSample(name='proxy_data')([input_image, tensor_mask])

X=tf.keras.layers.Conv1D(64, 15, activation='relu',padding='Same')(last_tensor)
X=tf.keras.layers.Conv1D(64, 15, activation='relu',padding='Same')(X)   
X=tf.keras.layers.Conv1D(64, 15, activation='relu',padding='Same')(X)
M=tf.keras.layers.MaxPooling1D(pool_size=2,strides=2, padding='Same')(X)
X=tf.keras.layers.Conv1D(32, 9, activation='relu',padding='Same')(M)
X=tf.keras.layers.Conv1D(32, 9, activation='relu',padding='Same')(X)   
X=tf.keras.layers.Conv1D(32, 9, activation='relu',padding='Same')(X)
M=tf.keras.layers.MaxPooling1D(pool_size=2,strides=2, padding='Same')(X)
Z=tf.keras.layers.Flatten()(M)
Y=tf.keras.layers.Dense(32, activation='relu')(Z)
#Y=tf.keras.layers.Dropout(0.2)(Y)
output_layer=tf.keras.layers.Dense(num_classification, activation='softmax')(Y)

model = tf.keras.Model(inputs=input_image, outputs=output_layer)
model.summary()

model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=LR), 
              loss = 'categorical_crossentropy', 
              metrics = ['accuracy'])
model.fit(x = X_train, y = Y_train,
          validation_data = (X_test, Y_test), 
          epochs=EPOCHS,
          batch_size=BATCH_SIZE,
          steps_per_epoch=ceil(training_input_num/BATCH_SIZE),
          callbacks = [tf.keras.callbacks.LearningRateScheduler(scheduler)])

model.save("hbs_mlbs_" + DATASET_NAME)
np.save("image_train_" + DATASET_NAME, X_train);np.save("label_train_" + DATASET_NAME, Y_train)
np.save("image_test_" + DATASET_NAME, X_test);np.save("label_test_" + DATASET_NAME, Y_test)