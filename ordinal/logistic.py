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
      Logistic Regression with mg/l data
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
mgl_logit_tau = mgl_logit_result.groupby(['C'])[['train_tau', 'val_tau']].mean().reset_index()

# plt.plot(logit_f1.C, logit_f1.train_macro_f1)
# plt.plot(logit_f1.C, logit_f1.val_macro_f1)
# plt.title('F1 score')
plt.plot(mgl_logit_tau.C, mgl_logit_tau.train_tau)
plt.plot(mgl_logit_tau.C, mgl_logit_tau.val_tau)
plt.title("Kendall's tau")
plt.xlabel('C')
plt.legend(['train', 'validation'])
plt.show()
plt.close()


#%%
mgl_logit = LogisticRegression(
      random_state = 0,
      penalty = 'l2',
      C = 3.5,
      multi_class = 'multinomial',
      solver = 'saga'
)

mgl_logit.fit(train_mgl_fingerprints, train_mgl_y.category)
train_mgl_pred = mgl_logit.predict(train_mgl_fingerprints)
mgl_logit_pred = mgl_logit.predict(test_mgl_fingerprints)

print('train results: \n', pd.crosstab(train_mgl_y.category, train_mgl_pred, rownames = ['true'], colnames = ['pred']))
print("\nkendall's tau = ", stats.kendalltau(train_mgl_y.category, train_mgl_pred))
print('\n', classification_report(train_mgl_y.category, train_mgl_pred, digits = 5))

print('test results: \n', pd.crosstab(test_mgl_y.category, mgl_logit_pred, rownames = ['true'], colnames = ['pred']))
print("\nkendall's tau = ", stats.kendalltau(test_mgl_y.category, mgl_logit_pred))
print('\n', classification_report(test_mgl_y.category, mgl_logit_pred, digits = 5))


#%%
'''
      Ordinal Logistic Regression with mg/l data
'''
mgl_ordinal = OrderedModel(
      train_mgl_y.category,
      train_mgl_fingerprints, 
      distr = 'logit')

mgl_ordinal_fit = mgl_ordinal.fit(method = 'lbfgs', maxiter = 1000)
mgl_ordinal_fit.summary()

mgl_pred_prob = mgl_ordinal_fit.predict(test_mgl_fingerprints)
mgl_ord_pred = np.argmax(np.array(mgl_pred_prob), axis = 1)

print("kendall's tau = ", stats.kendalltau(test_mgl_y.category, mgl_ord_pred))
print(classification_report(test_mgl_y.category, mgl_ord_pred, digits = 5))


#%%
'''
      Logistic Regression with ppm data
'''
ppm_logit_result = CV(
      train_ppm_fingerprints, 
      train_ppm_y, 
      LogisticRegression,
      params)


#%%
ppm_logit_result.iloc[ppm_logit_result.val_macro_f1.argmax(axis = 0)]
ppm_logit_result.iloc[ppm_logit_result.val_tau.argmax(axis = 0)]

ppm_l2 = ppm_logit_result[ppm_logit_result.penalty == 'l2']
ppm_l2.iloc[ppm_l2.val_tau.argmax(axis = 0)]


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
      penalty = 'l2',
      C = 3.5,
      multi_class = 'multinomial',
      solver = 'saga'
)
ppm_logit.fit(train_ppm_fingerprints, train_ppm_y.category)

train_ppm_pred = mgl_logit.predict(train_ppm_fingerprints)
ppm_logit_pred = mgl_logit.predict(test_ppm_fingerprints)

print('train results: \n', pd.crosstab(train_ppm_y.category, train_ppm_pred, rownames = ['true'], colnames = ['pred']))
print("\nkendall's tau = ", stats.kendalltau(train_ppm_y.category, train_ppm_pred))
print('\n', classification_report(train_ppm_y.category, train_ppm_pred, digits = 5))

print('test results: \n', pd.crosstab(test_ppm_y.category, ppm_logit_pred, rownames = ['true'], colnames = ['pred']))
print("\nkendall's tau = ", stats.kendalltau(test_ppm_y.category, ppm_logit_pred))
print('\n', classification_report(test_ppm_y.category, ppm_logit_pred, digits = 5))


#%%
'''
      Ordinal Logistic Regression with ppm data
'''
ppm_ordinal = OrderedModel(
      train_ppm_y.category,
      train_ppm_fingerprints, 
      distr = 'logit')

ppm_ordinal_fit = ppm_ordinal.fit(method = 'lbfgs', maxiter = 1000)
ppm_ordinal_fit.summary()

ppm_pred_prob = ppm_ordinal_fit.predict(test_ppm_fingerprints)
ppm_ord_pred = np.argmax(np.array(ppm_pred_prob), axis = 1)

print("kendall's tau = ", stats.kendalltau(test_ppm_y.category, ppm_ord_pred))
print(classification_report(test_ppm_y.category, ppm_ord_pred, digits = 5))

