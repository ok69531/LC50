#%%
from utils import *

import time
import random
import warnings

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression

import scipy.stats as stats
from sklearn.metrics import classification_report, roc_auc_score, recall_score

from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score


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
train_mgl, train_mgl_fingerprints, train_mgl_y = binary_mgl_load('train', path) 
train_ppm, train_ppm_fingerprints, train_ppm_y = binary_ppm_load('train', path)

print('train 범주에 포함된 데이터의 수\n', 
      train_mgl_y['category'].value_counts().sort_index(),
      '\n비율\n', 
      train_mgl_y['category'].value_counts(normalize = True).sort_index())

print('train 범주에 포함된 데이터의 수\n', 
      train_ppm_y['category'].value_counts().sort_index(),
      '\n비율\n', 
      train_ppm_y['category'].value_counts(normalize = True).sort_index())


#%%
# mgl_val_idx = random.sample(list(mgl_y.index), int(len(mgl_y) * 0.1))
# mgl_train_idx =  list(set(mgl_y.index) - set(mgl_val_idx))

# train_mgl_fingerprints = mgl_fingerprints.iloc[mgl_train_idx].reset_index(drop = True)
# train_mgl_y = mgl_y.iloc[mgl_train_idx].reset_index(drop = True)

# val_mgl_fingeprints = mgl_fingerprints.iloc[mgl_val_idx].reset_index(drop = True)
# val_mgl_y = mgl_y.iloc[mgl_val_idx].reset_index(drop = True)


#%%
# ppm_val_idx = random.sample(list(ppm_y.index), int(len(ppm_y) * 0.1))
# ppm_train_idx =  list(set(ppm_y.index) - set(ppm_val_idx))

# train_ppm_fingerprints = ppm_fingerprints.iloc[ppm_train_idx].reset_index(drop = True)
# train_ppm_y = ppm_y.iloc[ppm_train_idx].reset_index(drop = True)

# val_ppm_fingeprints = ppm_fingerprints.iloc[ppm_val_idx].reset_index(drop = True)
# val_ppm_y = ppm_y.iloc[ppm_val_idx].reset_index(drop = True)


#%%
test_mgl, test_mgl_fingerprints, test_mgl_y = binary_mgl_load('test', path)
test_ppm, test_ppm_fingerprints, test_ppm_y = binary_ppm_load('test', path)

print('test 범주에 포함된 데이터의 수\n', 
      test_mgl_y['category'].value_counts().sort_index(),
      '\n비율\n', 
      test_mgl_y['category'].value_counts(normalize = True).sort_index())
print('test 범주에 포함된 데이터의 수\n', 
      test_ppm_y['category'].value_counts().sort_index(),
      '\n비율\n', 
      test_ppm_y['category'].value_counts(normalize = True).sort_index())


#%%
'''
      Logistic Regression with mg/l data
'''
params_dict = {
      'random_state': [0], 
      'penalty': ['l1', 'l2'],
      'C': np.linspace(1e-6, 1e+2, 200),
      'solver': ['saga', 'liblinear']
}

params = ParameterGrid(params_dict)


#%%
mgl_logit_result = BinaryCV(
      train_mgl_fingerprints, 
      train_mgl_y, 
      LogisticRegression,
      params)


#%%
mgl_logit_result.iloc[mgl_logit_result.val_macro_f1.argmax(axis = 0)]


#%%
mgl_logit_f1 = mgl_logit_result.groupby(['C'])[['train_macro_f1', 'val_macro_f1']].mean().reset_index()

plt.plot(mgl_logit_f1.C, mgl_logit_f1.train_macro_f1)
plt.plot(mgl_logit_f1.C, mgl_logit_f1.val_macro_f1)
plt.title("F1 score")
plt.xlabel('C')
plt.legend(['train', 'validation'])
plt.show()
plt.close()


#%%
mgl_logit = LogisticRegression(
      random_state = 0,
      penalty = 'l1',
      C = 14,
      solver = 'liblinear'
)

mgl_logit.fit(train_mgl_fingerprints, train_mgl_y.category)
# train_mgl_pred = mgl_logit.predict(train_mgl_fingerprints)
mgl_logit_pred = mgl_logit.predict(test_mgl_fingerprints)


# print('train results: \n', pd.crosstab(train_mgl_y.category, train_mgl_pred, rownames = ['true'], colnames = ['pred']))
# print("\nauc = ", roc_auc_score(train_mgl_y.category, train_mgl_pred))
# print('\n', classification_report(train_mgl_y.category, train_mgl_pred, digits = 5))

print('test results: \n', pd.crosstab(test_mgl_y.category, mgl_logit_pred, rownames = ['true'], colnames = ['pred']))
print("\nauc = ", roc_auc_score(test_mgl_y.category, mgl_logit_pred))
print('\n', classification_report(test_mgl_y.category, mgl_logit_pred, digits = 5))


#%%
'''
    weighted logistic regression with mg/l data
'''
class WeightedLogitLoss(K.losses.Loss):
    def __init__(self, alpha):
        super(WeightedLogitLoss, self).__init__()
        self.alpha = alpha
    
    def call(self, y_true, prob):
        y_pred = tf.math.log(prob / (1 - prob))
        
        cond = y_true == 1
        
        loss_ = tf.math.log(1 + tf.math.exp(- 2 * y_true * y_pred))
        loss = tf.where(cond, 2 * self.alpha * loss_, 2 * (1 - self.alpha) * loss_)
        
        return tf.reduce_mean(loss)

class Logit(K.Model):
    def __init__(self):
        super(Logit, self).__init__()
        self.dense = layers.Dense(1)
    
    def call(self, inputs):
        p = 1 / (1 + tf.math.exp(-self.dense(inputs)))
        
        return p


@tf.function
def ridge(weight, lambda_):
	penalty = tf.math.square(weight) * lambda_
	return  tf.reduce_sum(penalty)


class ridge_dense(K.layers.Layer):
      def __init__(self, h, output_dim, lambda_, **kwargs):
            super(ridge_dense, self).__init__(**kwargs)
            self.input_dim = h.shape[-1]
            self.output_dim = output_dim
            self.lambda_ = lambda_
            self.ridge = ridge
            w_init = tf.random_normal_initializer()
            self.w = tf.Variable(initial_value=w_init(shape=(self.input_dim, 1), dtype='float32'), 
                                 trainable=True)
      
      def call(self, x):
            h = tf.matmul(x, self.w)
            self.add_loss(self.ridge(self.w, self.lambda_))
            
            return h


class RidgeLogit(K.Model):
    def __init__(self, h, output_dim, lambda_, **kwargs):
        super(RidgeLogit, self).__init__()
        self.dense = ridge_dense(h, output_dim, lambda_)
    
    def call(self, inputs):
        p = 1 / (1 + tf.math.exp(-self.dense(inputs)))
        
        return p



#%%
train_mgl_x = tf.cast(train_mgl_fingerprints, tf.float32)
# val_mgl_x = tf.cast(val_mgl_fingeprints, tf.float32)
test_mgl_x = tf.cast(test_mgl_fingerprints, tf.float32)

mgl_train_y = tf.cast(train_mgl_y.category, tf.float32)[..., tf.newaxis]
# mgl_val_y = tf.cast(val_mgl_y.category, tf.float32)[..., tf.newaxis]
mgl_test_y = tf.cast(test_mgl_y.category, tf.float32)[..., tf.newaxis]


#%%
lr = 0.001
adam = K.optimizers.Adam(lr)

epochs = 3000
result_ = []
alpha = np.arange(0, 1.1, 0.1)
# alpha = np.linspace(1e-6, 10, 150)


for alpha_ in alpha:
    tf.random.set_seed(0)
    
    wlogit = Logit()
    logitloss = WeightedLogitLoss(alpha_)
    loss = []
    
    num_epochs = tqdm(range(epochs), file = sys.stdout)
    
    for i in num_epochs:
        with tf.GradientTape(persistent=True) as tape:
            prob = wlogit(train_mgl_x)
            loss_ = logitloss(mgl_train_y, prob) 
        
        grad = tape.gradient(loss_, wlogit.trainable_weights)
        adam.apply_gradients(zip(grad, wlogit.trainable_weights))
        loss.append(loss_.numpy())
        
        num_epochs.set_postfix({'alpha': alpha_})
    
    mgl_pred_prob = wlogit(test_mgl_x)
    mgl_pred = pd.Series([1 if i >= 0.5 else -1 for i in mgl_pred_prob])
    
    result_.append({
        'alpha': alpha_,
        'loss': loss,
        'recall': recall_score(mgl_test_y, mgl_pred),
        'auc': roc_auc_score(mgl_test_y, mgl_pred),
        'pred_prob': mgl_pred,
        'pred': mgl_pred
    })


#%%
weight_result = pd.DataFrame(result_)

mgl_max_idx = weight_result.auc.argmax(axis = 0)
mgl_max_idx = weight_result.recall.argmax(axis = 0)
weight_result.iloc[mgl_max_idx]


plt.plot(weight_result.alpha, weight_result.recall)
plt.xlabel('alpha', fontsize=15)
plt.ylabel('recall', fontsize=15)
plt.show()
plt.close()

plt.plot(weight_result.alpha, weight_result.auc)
plt.xlabel('alpha', fontsize=15)
plt.ylabel('auc', fontsize=15)
plt.show()
plt.close()


print('test results: \n', pd.crosstab(test_mgl_y.category, weight_result.pred[mgl_max_idx], rownames = ['true'], colnames = ['pred']))
print("\nauc = ", roc_auc_score(test_mgl_y.category, weight_result.pred[mgl_max_idx]))
print('\n', classification_report(test_mgl_y.category, weight_result.pred[mgl_max_idx], digits = 5))



#%%
tf.random.set_seed(0)

mgl_weight_model = Logit()
mgl_loss_func = WeightedLogitLoss(weight_result.alpha[mgl_max_idx])
mgl_loss = []

for i in tqdm(range(epochs)):
      with tf.GradientTape(persistent=True) as tape:
            prob = mgl_weight_model(train_mgl_x)
            loss_ = mgl_loss_func(mgl_train_y, prob) 
            
      grad = tape.gradient(loss_, mgl_weight_model.trainable_weights)
      adam.apply_gradients(zip(grad, mgl_weight_model.trainable_weights))
      mgl_loss.append(loss_.numpy())


#%%
plt.plot(mgl_loss)
plt.show()
plt.close()


mgl_weight_pred_prob = mgl_weight_model.predict(test_mgl_x)
mgl_weight_pred = pd.Series([1 if i >= 0.5 else -1 for i in mgl_weight_pred_prob])

print('test results: \n', pd.crosstab(test_mgl_y.category, mgl_weight_pred, rownames = ['true'], colnames = ['pred']))
print("\nauc = ", roc_auc_score(test_mgl_y.category, mgl_weight_pred))
print('\n', classification_report(test_mgl_y.category, mgl_weight_pred, digits = 5))



#%%
# h = layers.Input((167))
# output_dim = 1
# lambda_ = np.arange(0, 1.1, 0.1, dtype = np.float32)

# lr = 0.001
# adam = K.optimizers.Adam(lr)

# epochs = 3000 
# result_ = []
# alpha = np.arange(0, 1.1, 0.1)


# for alpha_ in alpha:
#       for lam_ in lambda_:
#             tf.random.set_seed(0)
            
#             wlogit = RidgeLogit(h, output_dim, lam_)
#             logitloss = WeightedLogitLoss(alpha_)
#             loss = []
            
#             num_epochs = tqdm(range(epochs), file = sys.stdout)
            
#             for i in num_epochs:
#                   with tf.GradientTape(persistent=True) as tape:
#                         prob = wlogit(train_mgl_x)
#                         loss_ = logitloss(mgl_train_y, prob) 
                  
#                   grad = tape.gradient(loss_, wlogit.trainable_weights)
#                   adam.apply_gradients(zip(grad, wlogit.trainable_weights))
#                   loss.append(loss_.numpy())
                  
#                   num_epochs.set_postfix({'alpha': alpha_, 'lambda': lam_})
            
#             mgl_pred_prob = wlogit(test_mgl_x)
#             mgl_pred = pd.Series([1 if i >= 0.5 else -1 for i in mgl_pred_prob])
            
#             result_.append({
#                   'alpha': alpha_,
#                   'labmda': lam_,
#                   'loss': loss,
#                   'recall': recall_score(mgl_test_y, mgl_pred),
#                   'auc': roc_auc_score(mgl_test_y, mgl_pred),
#                   'pred_prob': mgl_pred,
#                   'pred': mgl_pred
#             })
    
    
# #%%
# lambda_result = pd.DataFrame(result_) 

# mgl_max_idx = lambda_result.auc.argmax(axis = 0)
# mgl_max_idx = lambda_result.recall.argmax(axis = 0)
# lambda_result.iloc[mgl_max_idx]

# plt.plot(weight_result.recall)
# plt.show()
# plt.close()

# print('test results: \n', pd.crosstab(test_mgl_y.category, lambda_result.pred[mgl_max_idx], rownames = ['true'], colnames = ['pred']))
# print("\nauc = ", roc_auc_score(test_mgl_y.category, lambda_result.pred[mgl_max_idx]))
# print('\n', classification_report(test_mgl_y.category, lambda_result.pred[mgl_max_idx], digits = 5))



#%%
'''
      Logistic Regression with ppm data
'''
ppm_logit_result = BinaryCV(
      train_ppm_fingerprints, 
      train_ppm_y, 
      LogisticRegression,
      params)


#%%
ppm_logit_result.iloc[ppm_logit_result.val_macro_f1.argmax(axis = 0)]
ppm_logit_result.iloc[ppm_logit_result.val_tau.argmax(axis = 0)]

# ppm_l2 = ppm_logit_result[ppm_logit_result.penalty == 'l2']
# ppm_l2.iloc[ppm_l2.val_tau.argmax(axis = 0)]


#%%
# logit_f1 = mgl_logit_result.groupby(['C'])[['train_macro_f1', 'val_macro_f1']].mean().reset_index()
ppm_logit_tau = ppm_logit_result.groupby(['C'])[['train_tau', 'val_tau']].mean().reset_index()

# plt.plot(logit_f1.C, logit_f1.train_macro_f1)
# plt.plot(logit_f1.C, logit_f1.val_macro_f1)
# plt.title('F1 score')
plt.plot(ppm_logit_tau.C, ppm_logit_tau.train_tau)
plt.plot(ppm_logit_tau.C, ppm_logit_tau.val_tau)
plt.title("Kendall's tau")
plt.xlabel('C')
plt.legend(['train', 'validation'])
plt.show()
plt.close()


#%%
ppm_logit = LogisticRegression(
      random_state = 0,
      penalty = 'l1',
      C = 1.5,
      multi_class = 'multinomial',
      solver = 'saga'
)
ppm_logit.fit(train_ppm_fingerprints, train_ppm_y.category)

train_ppm_pred = ppm_logit.predict(train_ppm_fingerprints)
ppm_logit_pred = ppm_logit.predict(test_ppm_fingerprints)

print('train results: \n', pd.crosstab(train_ppm_y.category, train_ppm_pred, rownames = ['true'], colnames = ['pred']))
print("\nkendall's tau = ", stats.kendalltau(train_ppm_y.category, train_ppm_pred))
print('\n', classification_report(train_ppm_y.category, train_ppm_pred, digits = 5))

print('test results: \n', pd.crosstab(test_ppm_y.category, ppm_logit_pred, rownames = ['true'], colnames = ['pred']))
print("\nkendall's tau = ", roc_auc_score(test_ppm_y.category, ppm_logit_pred))
print('\n', classification_report(test_ppm_y.category, ppm_logit_pred, digits = 5))




#%%
'''
      weighted logistic regression with mg/l data
'''
train_ppm_x = tf.cast(train_ppm_fingerprints, tf.float32)
# val_ppm_x = tf.cast(val_ppm_fingeprints, tf.float32)
test_ppm_x = tf.cast(test_ppm_fingerprints, tf.float32)

ppm_train_y = tf.cast(train_ppm_y.category, tf.float32)[..., tf.newaxis]
# ppm_val_y = tf.cast(val_ppm_y.category, tf.float32)[..., tf.newaxis]
ppm_test_y = tf.cast(test_ppm_y.category, tf.float32)[..., tf.newaxis]


#%%
lr = 0.001
adam = K.optimizers.Adam(lr)

epochs = 3000
ppm_result_ = []
alpha = np.arange(0.9, 1.01, 0.01)
# alpha = np.linspace(1e-6, 10, 150)


for alpha_ in alpha:
    tf.random.set_seed(0)
    
    wlogit = dLogit()
    logitloss = WeightedLogitLoss(alpha_)
    loss = []
    
    num_epochs = tqdm(range(epochs), file = sys.stdout)
    
    for i in num_epochs:
        with tf.GradientTape(persistent=True) as tape:
            prob = wlogit(train_ppm_x)
            loss_ = logitloss(ppm_train_y, prob) 
        
        grad = tape.gradient(loss_, wlogit.trainable_weights)
        adam.apply_gradients(zip(grad, wlogit.trainable_weights))
        loss.append(loss_.numpy())
        
        num_epochs.set_postfix({'alpha': alpha_})
    
    ppm_pred_prob = wlogit(test_ppm_x)
    ppm_pred = pd.Series([1 if i >= 0.5 else -1 for i in ppm_pred_prob])
    
    ppm_result_.append({
        'alpha': alpha_,
        'loss': loss,
        'recall': recall_score(ppm_test_y, ppm_pred),
        'auc': roc_auc_score(ppm_test_y, ppm_pred),
        'pred_prob': ppm_pred,
        'pred': ppm_pred
    })
    
    


#%%
ppm_weight_result = pd.DataFrame(ppm_result_)

ppm_max_idx = ppm_weight_result.auc.argmax(axis = 0)
ppm_max_idx = ppm_weight_result.recall.argmax(axis = 0)
ppm_weight_result.iloc[ppm_max_idx]

plt.plot(ppm_weight_result.alpha, ppm_weight_result.recall)
plt.xlabel('alpha', fontsize=15)
plt.ylabel('recall', fontsize=15)
plt.show()
plt.close()

plt.plot(ppm_weight_result.alpha, ppm_weight_result.auc)
plt.xlabel('alpha', fontsize=15)
plt.ylabel('auc', fontsize=15)
plt.show()
plt.close()



print('test results: \n', 
      pd.crosstab(test_ppm_y.category, ppm_weight_result.pred[mgl_max_idx], 
                  rownames = ['true'], colnames = ['pred']))
print("\nauc = ", roc_auc_score(test_ppm_y.category, 
                                ppm_weight_result.pred[ppm_max_idx]))
print('\n', classification_report(test_ppm_y.category, 
                                  ppm_weight_result.pred[ppm_max_idx], digits = 5))



#%%
tf.random.set_seed(0)

ppm_weight_model = Logit()
ppm_loss_func = WeightedLogitLoss(weight_result.alpha[ppm_max_idx])
ppm_loss = []

for i in tqdm(range(epochs)):
      with tf.GradientTape(persistent=True) as tape:
            prob = mgl_weight_model(train_ppm_x)
            loss_ = mgl_loss_func(ppm_train_y, prob) 
            
      grad = tape.gradient(loss_, ppm_weight_model.trainable_weights)
      adam.apply_gradients(zip(grad, ppm_weight_model.trainable_weights))
      ppm_loss.append(loss_.numpy())


#%%
plt.plot(ppm_loss)
plt.show()
plt.close()


ppm_weight_pred_prob = ppm_weight_model.predict(test_ppm_x)
ppm_weight_pred = pd.Series([1 if i >= 0.5 else -1 for i in ppm_weight_pred_prob])

print('test results: \n', pd.crosstab(test_ppm_y.category, 
                                      ppm_weight_pred, rownames = ['true'], 
                                      colnames = ['pred']))
print("\nauc = ", roc_auc_score(test_ppm_y.category, ppm_weight_pred))
print('\n', classification_report(test_ppm_y.category, ppm_weight_pred, digits = 5))




#%%
'''
      Logistic with SMOTE
'''

from imblearn.over_sampling import SMOTE

oversample = SMOTE()
mgl_smote_fingerprints, mgl_smote_y = oversample.fit_resample(train_mgl_fingerprints, 
                                                              train_mgl_y.category)
ppm_smote_fingerprints, ppm_smote_y = oversample.fit_resample(train_ppm_fingerprints,
                                                              train_ppm_y.category)

#%%
params_dict = {
      'random_state': [0], 
      'penalty': ['l1', 'l2'],
      'C': np.linspace(1e-6, 1e+2, 200),
      'solver': ['saga', 'liblinear']
}

params = ParameterGrid(params_dict)


#%%
mgl_smote_result = BinaryCV(
      mgl_smote_fingerprints, 
      mgl_smote_y, 
      LogisticRegression,
      params)


#%%
mgl_smote_result.iloc[mgl_smote_result.val_macro_recall.argmax(axis = 0)]


#%%
mgl_logit_f1 = mgl_smote_result.groupby(['C'])[['train_macro_f1', 'val_macro_f1']].mean().reset_index()

plt.plot(mgl_smote_result.C, mgl_smote_result.train_macro_f1)
plt.plot(mgl_smote_result.C, mgl_smote_result.val_macro_f1)
plt.title("F1 score")
plt.xlabel('C')
plt.legend(['train', 'validation'])
plt.show()
plt.close()


#%%
mgl_logit = LogisticRegression(
      random_state = 0,
      penalty = 'l1',
      C = 81,
      solver = 'liblinear'
)

mgl_logit.fit(mgl_smote_fingerprints, mgl_smote_y)
train_mgl_pred = mgl_logit.predict(train_mgl_fingerprints)
mgl_logit_pred = mgl_logit.predict(test_mgl_fingerprints)


print('test results: \n', pd.crosstab(test_mgl_y.category, mgl_logit_pred, rownames = ['true'], colnames = ['pred']))
print("\nauc = ", roc_auc_score(test_mgl_y.category, mgl_logit_pred))
print('\n', classification_report(test_mgl_y.category, mgl_logit_pred, digits = 5))



#%%
ppm_smote_result = BinaryCV(
      ppm_smote_fingerprints, 
      ppm_smote_y, 
      LogisticRegression,
      params)


#%%
ppm_smote_result.iloc[ppm_smote_result.val_macro_recall.argmax(axis = 0)]


#%%
ppm_logit_f1 = ppm_smote_result.groupby(['C'])[['train_macro_f1', 'val_macro_f1']].mean().reset_index()

plt.plot(ppm_smote_result.C, ppm_smote_result.train_macro_f1)
plt.plot(ppm_smote_result.C, ppm_smote_result.val_macro_f1)
plt.title("F1 score")
plt.xlabel('C')
plt.legend(['train', 'validation'])
plt.show()
plt.close()


#%%
ppm_logit = LogisticRegression(
      random_state = 0,
      penalty = 'l1',
      C = 3,
      solver = 'saga'
)

ppm_logit.fit(ppm_smote_fingerprints, ppm_smote_y)
ppm_logit_pred = ppm_logit.predict(test_ppm_fingerprints)


print('test results: \n', pd.crosstab(test_ppm_y.category, ppm_logit_pred, rownames = ['true'], colnames = ['pred']))
print("\nauc = ", roc_auc_score(test_ppm_y.category, ppm_logit_pred))
print('\n', classification_report(test_ppm_y.category, ppm_logit_pred, digits = 5))
