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
train_mgl_x = tf.cast(train_mgl_fingerprints, tf.float32)
test_mgl_x = tf.cast(test_mgl_fingerprints, tf.float32)

mgl_train_y = tf.cast(train_mgl_y.category, tf.int32)
mgl_test_y = tf.cast(test_mgl_y.category, tf.int32)


#%%
class model1(K.Model):
    def __init__(self):
        super(model1, self).__init__()
        self.dense1 = layers.Dense(5, activation = 'softmax')
        
    def call(self, inputs):
        yhat = self.dense1(inputs)
        
        return yhat


#%%
class model3(K.Model):
    def __init__(self):
        super(model3, self).__init__()
        self.dense1 = layers.Dense(100, activation = 'relu')
        self.dense2 = layers.Dense(50, activation = 'tanh')
        self.dense3 = layers.Dense(5, activation = 'softmax')
    
    def call(self, inputs):
        h1 = self.dense1(inputs)
        h2 = self.dense2(h1)
        yhat = self.dense3(h2)
        
        return yhat


#%%
class model5(K.Model):
    def __init__(self):
        super(model5, self).__init__()
        self.dense1 = layers.Dense(100, activation = 'relu')
        self.dense2 = layers.Dense(70)
        self.dense3 = layers.Dense(50, activation = 'tanh')
        self.dense4 = layers.Dense(25)
        self.dense5 = layers.Dense(5, activation = 'softmax')
    
    def call(self, inputs):
        h1 = self.dense1(inputs)
        h2 = self.dense2(h1)
        h3 = self.dense3(h2)
        h4 = self.dense4(h3)
        yhat = self.dense5(h4)
        
        return yhat


#%%
tf.random.set_seed(0)
mgl_model1 = model1()

adam = K.optimizers.Adam(0.001)
scc = K.losses.SparseCategoricalCrossentropy()

mgl_model1.compile(optimizer = adam, loss = scc, metrics = ['accuracy'])
mgl_result1 = mgl_model1.fit(train_mgl_x, mgl_train_y, epochs = 1000, batch_size = len(mgl_train_y), verbose = 1)


#%%
mgl_mlp_prob1 = mgl_model1.predict(test_mgl_x)
print(scc(mgl_test_y, mgl_mlp_prob1).numpy())

mgl_mlp_pred1 = np.argmax(mgl_mlp_prob1, axis = 1)

print('kendall tau: ', stats.kendalltau(mgl_test_y, mgl_mlp_pred1),
      '\n', classification_report(mgl_test_y, mgl_mlp_pred1, digits = 5))


#%%
tf.random.set_seed(0)
mgl_model3 = model3()

adam = K.optimizers.Adam(0.001)
scc = K.losses.SparseCategoricalCrossentropy()

mgl_model3.compile(optimizer = adam, loss = scc, metrics = ['accuracy'])
mgl_result3 = mgl_model3.fit(train_mgl_x, mgl_train_y, epochs = 1000, batch_size = len(mgl_train_y), verbose = 1)


#%%
mgl_mlp_prob3 = mgl_model3.predict(test_mgl_x)
print(scc(mgl_test_y, mgl_mlp_prob3).numpy())

mgl_mlp_pred3 = np.argmax(mgl_mlp_prob3, axis = 1)

print('kendall tau: ', stats.kendalltau(mgl_test_y, mgl_mlp_pred3),
      '\n', classification_report(mgl_test_y, mgl_mlp_pred3, digits = 5))


#%%
tf.random.set_seed(0)
mgl_model5 = model5()

adam = K.optimizers.Adam(0.001)
scc = K.losses.SparseCategoricalCrossentropy()

mgl_model5.compile(optimizer = adam, loss = scc, metrics = ['accuracy'])
mgl_result5 = mgl_model5.fit(train_mgl_x, mgl_train_y, epochs = 1000, batch_size = len(mgl_train_y), verbose = 1)


#%%
mgl_mlp_prob5 = mgl_model5.predict(test_mgl_x)
print(scc(mgl_test_y, mgl_mlp_prob5).numpy())

mgl_mlp_pred5 = np.argmax(mgl_mlp_prob5, axis = 1)

print('kendall tau: ', stats.kendalltau(mgl_test_y, mgl_mlp_pred5),
      '\n', classification_report(mgl_test_y, mgl_mlp_pred5, digits = 5))


#%%
class ordinal(layers.Layer):
    def __init__(self, num_class):
        super(ordinal, self).__init__()
        self.num_class = num_class
        self.theta = tf.Variable(tf.cumsum(tf.random.uniform((1, num_class - 1)), axis = 1))
        self.dense = layers.Dense(1)
        
    def call(self, inputs):
        x = tf.expand_dims(self.theta, 0) - self.dense(inputs)
        cum_prob = tf.squeeze(tf.nn.sigmoid(x))
        prob = tf.concat([
            cum_prob[:, :1], 
            cum_prob[:, 1:] - cum_prob[:, :-1],
            1 - cum_prob[:, -1:]], axis = 1)
        
        return prob

class ord_model(K.Model):
    def __init__(self):
        super(ord_model, self).__init__()
        self.dense1 = layers.Dense(100, activation = 'relu')
        self.dense2 = layers.Dense(50, activation = 'tanh')
        self.dense3 = layers.Dense(5, activation = 'relu')
        self.dense4 = ordinal(5)
    
    def call(self, inputs):
        h1 = self.dense1(inputs)
        h2 = self.dense2(h1)
        h3 = self.dense3(h2)
        yhat = self.dense4(h3)
        
        return yhat


#%%
tf.random.set_seed(0)
mgl_ord_model = ord_model()

adam = K.optimizers.Adam(0.0001)
scc = K.losses.SparseCategoricalCrossentropy()

mgl_ord_model.compile(optimizer = adam, loss = scc, metrics = ['accuracy'])
mgl_ord_result = mgl_ord_model.fit(train_mgl_x, mgl_train_y, epochs = 1000, batch_size = len(mgl_train_y), verbose = 1)
# epochs = 10000
# lr = 0.0005, epochs = 5000
# lr = 0.0001, epochs = 1000

#%%
mgl_ord_prob = mgl_ord_model.predict(test_mgl_x)
print(scc(mgl_test_y, mgl_ord_prob).numpy())

mgl_ord_pred = np.argmax(mgl_ord_prob, axis = 1)

print('kendall tau: ', stats.kendalltau(mgl_test_y, mgl_ord_pred),
      '\n', classification_report(mgl_test_y, mgl_ord_pred, digits = 5))


#%%
'''
    ppm data
'''
train_ppm_x = tf.cast(train_ppm_fingerprints, tf.float32)
test_ppm_x = tf.cast(test_ppm_fingerprints, tf.float32)

ppm_train_y = tf.cast(train_ppm_y.category, tf.int32)
ppm_test_y = tf.cast(test_ppm_y.category, tf.int32)



#%%
tf.random.set_seed(0)
ppm_model3 = model3()

adam = K.optimizers.Adam(0.001)
scc = K.losses.SparseCategoricalCrossentropy()

ppm_model3.compile(optimizer = adam, loss = scc, metrics = ['accuracy'])
ppm_result3 = ppm_model3.fit(train_ppm_x, ppm_train_y, epochs = 1000, batch_size = len(ppm_train_y), verbose = 1)


#%%
ppm_mlp_prob3 = ppm_model3.predict(test_ppm_x)
print(scc(ppm_test_y, ppm_mlp_prob3).numpy())

ppm_mlp_pred3 = np.argmax(ppm_mlp_prob3, axis = 1)

print('kendall tau: ', stats.kendalltau(ppm_test_y, ppm_mlp_pred3),
      '\n', classification_report(ppm_test_y, ppm_mlp_pred3, digits = 5))


#%%
tf.random.set_seed(0)
ppm_ord_model = ord_model()

adam = K.optimizers.Adam(0.005)
scc = K.losses.SparseCategoricalCrossentropy()

ppm_ord_model.compile(optimizer = adam, loss = scc, metrics = ['accuracy'])
ppm_ord_result = ppm_ord_model.fit(train_ppm_x, ppm_train_y, epochs = 1000, batch_size = len(ppm_train_y), verbose = 1)


#%%
ppm_ord_prob = ppm_ord_model.predict(test_ppm_x)
print(scc(ppm_test_y, ppm_ord_prob).numpy())

ppm_ord_pred = np.argmax(ppm_ord_prob, axis = 1)

print('kendall tau: ', stats.kendalltau(ppm_test_y, ppm_ord_pred),
      '\n', classification_report(ppm_test_y, ppm_ord_pred, digits = 5))
