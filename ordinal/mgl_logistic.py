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

try:
      import wandb
except: 
      import sys
      import subprocess
      subprocess.check_call([sys.executable, "-m", "pip", "install", "wandb"])
      import wandb


warnings.filterwarnings("ignore")

wandb.login(key="1c2f31977d15e796871c32701e62c5ec1167070e")
wandb.init(project="LC50-mgl-logistic", entity="soyoung")


def mgl_logit_main(seed_):
      path = 'C:/Users/SOYOUNG/Desktop/github/LC50/data/'

      mgl, mgl_fingerprints, mgl_y = mgl_load(path)
      train_mgl_fingerprints, train_mgl_y, test_mgl_fingerprints, test_mgl_y = data_split(
            mgl_fingerprints,
            mgl_y.category,
            seed = seed_
      )


      # print('mg/l',
      #       '\n기초통계량:\n', mgl.value.describe(),
      #       '\n분위수: ', np.quantile(mgl.value, [0.2, 0.4, 0.6, 0.8, 1]))

      # print('범주에 포함된 데이터의 수\n', mgl_y.category.value_counts().sort_index(),
      #       '\n비율\n', mgl_y.category.value_counts(normalize = True).sort_index())

      # print('train 범주에 포함된 데이터의 수\n', train_mgl_y.value_counts().sort_index(),
      #       '\n비율\n', train_mgl_y.value_counts(normalize = True).sort_index())

      # print('test 범주에 포함된 데이터의 수\n', test_mgl_y.value_counts().sort_index(),
      #       '\n비율\n', test_mgl_y.value_counts(normalize = True).sort_index())


      ppm, ppm_fingerprints, ppm_y = ppm_load(path)
      train_ppm_fingerprints, train_ppm_y, test_ppm_fingerprints, test_ppm_y = data_split(
            ppm_fingerprints, 
            ppm_y.category,
            seed = seed_
      )


      # print('ppm', 
      #       '\n기초통계량:\n', ppm.value.describe(),
      #       '\n분위수: ', np.quantile(ppm.value, [0.2, 0.4, 0.6, 0.8, 1]))

      # print('범주에 포함된 데이터의 수\n', ppm_y.category.value_counts().sort_index(),
      #       '\n비율\n', ppm_y.category.value_counts(normalize = True).sort_index())

      # print('train 범주에 포함된 데이터의 수\n', train_ppm_y.value_counts().sort_index(),
      #       '\n비율\n', train_ppm_y.value_counts(normalize = True).sort_index())

      # print('test 범주에 포함된 데이터의 수\n', test_ppm_y.value_counts().sort_index(),
      #       '\n비율\n', test_ppm_y.value_counts(normalize = True).sort_index())


      '''
            Logistic Regression with mg/l data
      '''
      
      params_dict = {
            'random_state': [seed_], 
            'penalty': ['l1', 'l2'],
            # 'C': np.logspace(-10, 10, num = 100, base = 2),
            'C': np.linspace(1e-6, 50, 150),
            'solver': ['liblinear', 'saga']
            }

      params = ParameterGrid(params_dict)

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
      

      result = pd.DataFrame([{
            'seed': seed_,
            'parameters': best_params,
            'precision': precision_score(test_mgl_y, mgl_logit_pred, average = 'macro'), 
            'recall': recall_score(test_mgl_y, mgl_logit_pred, average = 'macro'), 
            'f1': f1_score(test_mgl_y, mgl_logit_pred, average = 'macro'), 
            'accuracy': accuracy_score(test_mgl_y, mgl_logit_pred),
            'tau': stats.kendalltau(test_mgl_y, mgl_logit_pred).correlation
      }])


      wandb.log({
            'table': result
      })
      
      
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