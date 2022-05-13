#%%
from utils import (
      Smiles2Fing,
      mgl_load,
      ppm_load, 
      data_split,
      ParameterGrid,
      MultiCV
)
from models import OrdinalLogitClassifier

import time
import random
import warnings

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression

import scipy.stats as stats
from sklearn.metrics import classification_report
from sklearn.metrics import (
    precision_score, 
    recall_score, 
    f1_score, 
    accuracy_score, 
    roc_auc_score
    )

warnings.filterwarnings("ignore")


#%%
seed_ = 0

path = 'C:/Users/SOYOUNG/Desktop/github/LC50/data/'

mgl, mgl_fingerprints, mgl_y = mgl_load(path)
train_mgl_fingerprints, train_mgl_y, test_mgl_fingerprints, test_mgl_y = data_split(
      mgl_fingerprints,
      mgl_y.category,
      seed = seed_
)


print('mg/l',
      '\n기초통계량:\n', mgl.value.describe(),
      '\n분위수: ', np.quantile(mgl.value, [0.2, 0.4, 0.6, 0.8, 1]))

print('범주에 포함된 데이터의 수\n', mgl_y.category.value_counts().sort_index(),
      '\n비율\n', mgl_y.category.value_counts(normalize = True).sort_index())

print('train 범주에 포함된 데이터의 수\n', train_mgl_y.value_counts().sort_index(),
      '\n비율\n', train_mgl_y.value_counts(normalize = True).sort_index())

print('test 범주에 포함된 데이터의 수\n', test_mgl_y.value_counts().sort_index(),
      '\n비율\n', test_mgl_y.value_counts(normalize = True).sort_index())


#%%
ppm, ppm_fingerprints, ppm_y = ppm_load(path)
train_ppm_fingerprints, train_ppm_y, test_ppm_fingerprints, test_ppm_y = data_split(
      ppm_fingerprints, 
      ppm_y.category,
      seed = seed_
)


print('ppm', 
      '\n기초통계량:\n', ppm.value.describe(),
      '\n분위수: ', np.quantile(ppm.value, [0.2, 0.4, 0.6, 0.8, 1]))

print('범주에 포함된 데이터의 수\n', ppm_y.category.value_counts().sort_index(),
      '\n비율\n', ppm_y.category.value_counts(normalize = True).sort_index())

print('train 범주에 포함된 데이터의 수\n', train_ppm_y.value_counts().sort_index(),
      '\n비율\n', train_ppm_y.value_counts(normalize = True).sort_index())

print('test 범주에 포함된 데이터의 수\n', test_ppm_y.value_counts().sort_index(),
      '\n비율\n', test_ppm_y.value_counts(normalize = True).sort_index())


#%%
import neptune.new as neptune

params_dict = {
      'random_state': [seed_], 
      'penalty': ['l1', 'l2'],
      'C': np.logspace(-10, 10, num = 100, base = 2),
      # 'C': np.linspace(1e-6, 1e+2, 200),
      'solver': ['liblinear', 'saga']
      }

params = ParameterGrid(params_dict)

def mgl_logistic():
      '''
            Logistic Regression with mg/l data
      '''
      mgl_logit_result = MultiCV(
            train_mgl_fingerprints, 
            train_mgl_y, 
            LogisticRegression,
            params
      )

      max_tau_idx = mgl_logit_result.val_tau.argmax(axis = 0)
      best_params = mgl_logit_result.iloc[max_tau_idx][:4].to_dict()


      logit = LogisticRegression(**best_params)

      logit.fit(train_mgl_fingerprints, train_mgl_y)
      mgl_logit_pred = logit.predict(test_mgl_fingerprints)
      
      # run = neptune.init(
      # project="ok69531/LC50-mgl-logistic",
      # api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJmOGQxYzY2YS02MmVjLTQ3MDUtOTlmNS0xYWYyODY3ZmE2MzYifQ==",
      # ) 
      
      # run['parameters'] = best_params
      # run['precision'] = precision_score(test_mgl_y, mgl_logit_pred, average = 'macro')
      # run['recall'] = recall_score(test_mgl_y, mgl_logit_pred, average = 'macro')
      # run['f1'] = f1_score(test_mgl_y, mgl_logit_pred, average = 'macro')
      # run['accuracy'] = accuracy_score(test_mgl_y, mgl_logit_pred)
      # run['tau'] = stats.kendalltau(test_mgl_y, mgl_logit_pred).correlation
      
      # run.stop()

for seed_ in range(4):
      mgl_logistic()






#%%
'''
      여기서부터 수정
'''
mgl_ord_result = MultiCV(
      train_mgl_fingerprints, 
      train_mgl_y, 
      OrdinalLogitClassifier,
      params)

mgl_ord_result.iloc[mgl_ord_result.val_macro_f1.argmax(axis = 0)]
mgl_ord_result.iloc[mgl_ord_result.val_tau.argmax(axis = 0)]

mgl_l2 = mgl_ord_result[mgl_ord_result.penalty == 'l2']
mgl_l2.iloc[mgl_l2.val_tau.argmax(axis = 0)]



#%%
mgl_ordlogit = LogitOrdinalClassifier(
    random_state = 0,
    penalty = 'l2',
    C = 8.5,
    multi_class = 'multinomial',
    solver = 'saga'
)

mgl_ordlogit.fit(train_mgl_fingerprints, train_mgl_y.category)

mgl_ord_pred = mgl_ordlogit.predict(test_mgl_fingerprints)

print("kendall's tau = ", stats.kendalltau(test_mgl_y.category, mgl_ord_pred))
print(classification_report(test_mgl_y.category, mgl_ord_pred, digits = 5))



#%%
'''
      Logistic Regression with ppm data
'''
ppm_logit_result = MultiCV(
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
print("\nkendall's tau = ", stats.kendalltau(test_ppm_y.category, ppm_logit_pred))
print('\n', classification_report(test_ppm_y.category, ppm_logit_pred, digits = 5))


#%%
'''
      Ordinal Logistic Regression with ppm data
'''
ppm_ord_result = MultiCV(
      train_ppm_fingerprints, 
      train_ppm_y, 
      LogitOrdinalClassifier,
      params)

ppm_ord_result.iloc[ppm_ord_result.val_macro_f1.argmax(axis = 0)]
ppm_ord_result.iloc[ppm_ord_result.val_tau.argmax(axis = 0)]


#%%
ppm_ordlogit = LogitOrdinalClassifier(
    random_state = 0,
    penalty = 'l1',
    C = 2,
    multi_class = 'multinomial',
    solver = 'saga'
)

ppm_ordlogit.fit(train_ppm_fingerprints, train_ppm_y.category)

ppm_ord_pred = ppm_ordlogit.predict(test_ppm_fingerprints)

print("kendall's tau = ", stats.kendalltau(test_ppm_y.category, ppm_ord_pred))
print(classification_report(test_ppm_y.category, ppm_ord_pred, digits = 5))



#%%
'''
      mg/l binary
'''
from sklearn.metrics import cohen_kappa_score, roc_auc_score

mgl_binary = pd.DataFrame({
   'y': [0 if i != 1 else 1 for i in test_mgl_y.category],
   'logit_pred': [0 if i != 1 else 1 for i in mgl_logit_pred],
   'ord_pred': [0 if i != 1 else 1 for i in mgl_ord_pred],
})


print(pd.crosstab(mgl_binary.y, mgl_binary.logit_pred, rownames = ['true'], colnames = ['pred']))
print('cohens kappa = ', cohen_kappa_score(mgl_binary.y, mgl_binary.logit_pred))
print('auc = ', roc_auc_score(mgl_binary.y, mgl_binary.logit_pred))
print(classification_report(mgl_binary.y, mgl_binary.logit_pred, digits = 5))

print(pd.crosstab(mgl_binary.y, mgl_binary.ord_pred, rownames = ['true'], colnames = ['pred']))
print('cohens kappa = ', cohen_kappa_score(mgl_binary.y, mgl_binary.ord_pred))
print('auc = ', roc_auc_score(mgl_binary.y, mgl_binary.ord_pred))
print(classification_report(mgl_binary.y, mgl_binary.ord_pred, digits = 5))


#%%
'''
      ppm binary
'''
ppm_binary = pd.DataFrame({
   'y': [0 if i != 1 else 1 for i in test_ppm_y.category],
   'logit_pred': [0 if i != 1 else 1 for i in ppm_logit_pred],
   'ord_pred': [0 if i != 1 else 1 for i in ppm_ord_pred],
})


print(pd.crosstab(ppm_binary.y, ppm_binary.logit_pred, rownames = ['true'], colnames = ['pred']))
print('cohens kappa = ', cohen_kappa_score(ppm_binary.y, ppm_binary.logit_pred))
print('auc = ', roc_auc_score(ppm_binary.y, ppm_binary.logit_pred))
print(classification_report(ppm_binary.y, ppm_binary.logit_pred, digits = 5))

print(pd.crosstab(ppm_binary.y, ppm_binary.ord_pred, rownames = ['true'], colnames = ['pred']))
print('cohens kappa = ', cohen_kappa_score(ppm_binary.y, ppm_binary.ord_pred))
print('auc = ', roc_auc_score(ppm_binary.y, ppm_binary.ord_pred))
print(classification_report(ppm_binary.y, ppm_binary.ord_pred, digits = 5))