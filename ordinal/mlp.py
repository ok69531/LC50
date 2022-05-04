#%%
#%%
from utils import *

import time
import random
import warnings

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import scipy.stats as stats
from sklearn.metrics import classification_report

warnings.filterwarnings("ignore")


#%%
import tensorflow as tf
import tensorflow.keras as K
from tensorflow.keras import layers
print('TensorFlow version:', tf.__version__)
print('Eager Execution Mode:', tf.executing_eagerly())
print('available GPU:', tf.config.list_physical_devices('GPU'))
print('Num GPUs Available: ', len(tf.config.experimental.list_physical_devices('GPU')))
from tensorflow.python.client import device_lib
print('=========================================')
print(device_lib.list_local_devices())
tf.debugging.set_log_device_placement(False)



#%%
path = 'C:/Users/SOYOUNG/Desktop/github/LC50/data/'
train_mgl, train_mgl_fingerprints, train_mgl_y = train_mgl_load(path) 
train_ppm, train_ppm_fingerprints, train_ppm_y = train_ppm_load(path)

test_mgl, test_mgl_fingerprints, test_mgl_y = test_mgl_load(path)
test_ppm, test_ppm_fingerprints, test_ppm_y = test_ppm_load(path)


#%%
train_mgl_x = tf.cast(mgl_fingerprints, tf.float32)
test_mgl_x = tf.cast(test_mgl_fingerprints, tf.float32)

mgl_nn_train_y = tf.cast(train_mgl_y.category, tf.int32)
mgl_nn_test_y = tf.cast(test_mgl_y.category, tf.int32)


#%%
input1 = layers.Input((train_mgl_x.shape[1]))

dense1 = layers.Dense(100, activation = 'relu')
dense2 = layers.Dense(50, activation = 'tanh')
dense3 = layers.Dense(5, activation = 'softmax')

yhat = dense3(dense2(dense1(input1)))

model = K.models.Model(input1, yhat)
model.summary()


#%%
adam = K.optimizers.Adam(0.001)
# mae = K.losses.MeanAbsoluteError()
scc = K.losses.SparseCategoricalCrossentropy()

# model.compile(optimizer = adam, loss = bc, metrics = ['accuracy'])
model.compile(optimizer = adam, loss = scc, metrics = ['accuracy'])
result = model.fit(train_mgl_x, mgl_nn_train_y, epochs = 1000, batch_size = len(mgl_nn_train_y), verbose = 1)
# result = model.fit(x_train_smote, half_train_smote, epochs = 500, batch_size = len(half_train_smote), verbose = 1)


#%%
mgl_nn_pred_prob = model.predict(test_mgl_x)
print(scc(mgl_nn_test_y, mgl_nn_pred_prob).numpy())

mgl_nn_pred = np.argmax(mgl_nn_pred_prob, axis = 1)

print('kendall tau: ', stats.kendalltau(mgl_nn_test_y, mgl_nn_pred),
      '\n', classification_report(mgl_nn_test_y, mgl_nn_pred, digits = 5))
