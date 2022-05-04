#%%
from utils import *

import time
import random
import openpyxl
import warnings

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import scipy.stats as stats
from sklearn.metrics import classification_report

from statsmodels.miscmodels.ordinal_model import OrderedModel

warnings.filterwarnings("ignore")


#%%
path = 'C:/Users/SOYOUNG/Desktop/github/LC50/data/'
train_mgl, train_mgl_fingerprints, train_mgl_y = train_mgl_load(path) 
train_ppm, train_ppm_fingerprints, train_ppm_y = train_ppm_load(path)

# print('train mgl')
# print('기초통계량:\n', train_mgl['value'].describe())
# print('분위수: ', np.quantile(train_mgl['value'], [0.2, 0.4, 0.6, 0.8, 1]))

# print('\ntrain ppm')
# print('기초통계량:\n', train_ppm['value'].describe())
# print('분위수: ', np.quantile(train_ppm['value'], [0.2, 0.4, 0.6, 0.8, 1]))

print('train 범주에 포함된 데이터의 수\n', 
      train_mgl_y['category'].value_counts().sort_index(),
      '\n비율\n', 
      train_mgl_y['category'].value_counts(normalize = True).sort_index())
print('train 범주에 포함된 데이터의 수\n', 
      train_ppm_y['category'].value_counts().sort_index(),
      '\n비율\n', 
      train_ppm_y['category'].value_counts(normalize = True).sort_index())


#%%
test_mgl, test_mgl_fingerprints, test_mgl_y = test_mgl_load(path)
test_ppm, test_ppm_fingerprints, test_ppm_y = test_ppm_load(path)

# print('\ntest mgl')
# print('기초통계량:\n', test_mgl['value'].describe())
# print('분위수: ', np.quantile(test_mgl['value'], [0.2, 0.4, 0.6, 0.8, 1]))

# print('\ntest ppm')
# print('기초통계량:\n', test_ppm['value'].describe())
# print('분위수: ', np.quantile(test_ppm['value'], [0.2, 0.4, 0.6, 0.8, 1]))

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
      Logistic Regression
'''
from sklearn.linear_model import LogisticRegression

params_dict = {
      'random_state': [0], 
      'penalty': ['l1', 'l2'],
      'C': np.linspace(1e-6, 1e+2, 200),
      'multi_class': ['multinomial'],
      'solver': ['saga']
      }

params = ParameterGrid(params_dict)

mgl_logit_result = CV(
      train_mgl_fingerprints, 
      train_mgl_y, 
      LogisticRegression,
      params)


#%%
mgl_logit_result.iloc[mgl_logit_result.val_macro_f1.argmax(axis = 0)]
mgl_logit_result.iloc[mgl_logit_result.val_tau.argmax(axis = 0)]


#%%
# logit_f1 = mgl_logit_result.groupby(['C'])[['train_macro_f1', 'val_macro_f1']].mean().reset_index()
logit_f1 = mgl_logit_result.groupby(['C'])[['train_tau', 'val_tau']].mean().reset_index()

# plt.plot(logit_f1.C, logit_f1.train_macro_f1)
# plt.plot(logit_f1.C, logit_f1.val_macro_f1)
# plt.title('F1 score')
plt.plot(logit_f1.C, logit_f1.train_tau)
plt.plot(logit_f1.C, logit_f1.val_tau)
plt.title("Kendall's tau")
plt.xlabel('C')
plt.legend(['train', 'validation'])
plt.show()
plt.close()


#%%
'''
      Ordinal Logistic Regression
'''
mgl_ordinal = OrderedModel(train_mgl_y.category,
                           mgl_fingerprints, 
                           distr = 'probit')

mgl_ordinal_fit = mgl_ordinal.fit(method = 'lbfgs', maxiter = 1000)
mgl_ordinal_fit.summary()

pred_prob = mgl_ordinal_fit.predict(test_mgl_fingerprints)
mgl_y_pred = np.argmax(np.array(pred_prob), axis = 1)

print("kendall's tau = ", stats.kendalltau(test_mgl_y.category, mgl_y_pred))
print(classification_report(test_mgl_y.category, mgl_y_pred, digits = 5))


#%%
ppm_ordinal = OrderedModel(train_ppm_y.category,
                           ppm_fingerprints, 
                           distr = 'logit')

ppm_ordinal_fit = ppm_ordinal.fit(method = 'lbfgs', maxiter = 1000)
ppm_ordinal_fit.summary()

pred_prob = ppm_ordinal_fit.predict(test_ppm_fingerprints)
ppm_y_pred = np.argmax(np.array(pred_prob), axis = 1)

print("kendall's tau = ", stats.kendalltau(test_ppm_y.category, ppm_y_pred))
print(classification_report(test_ppm_y.category, ppm_y_pred, digits = 5))








