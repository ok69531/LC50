import sys
sys.path.append('../')

from util import (
    data_load,
    data_split,
    ParameterGrid,
    MultiCV
)

import time
import random
import warnings

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier

import scipy.stats as stats
from sklearn.metrics import (
    precision_score, 
    recall_score, 
    f1_score, 
    accuracy_score, 
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
wandb.init(project="LC50-rf", entity="soyoung")


def rf_main(seed_):
      
    data, fingerprints, y = data_load()
    
    train_fingerprints, train_y, test_fingerprints, test_y = data_split(
        fingerprints,
        y.category,
        seed = seed_
    )

    
    params_dict = {
        'random_state': [seed_], 
        'n_estimators': np.arange(30, 155, 10),
        'min_samples_split': list(range(2, 9)),
        'max_features': ['auto', 'sqrt', 'log2'],
        'class_weight': [None, {0:1.6, 1:1.2, 2:2.7, 3:3.4, 4:1.1}]
    }

    params = ParameterGrid(params_dict)

    rf_result = MultiCV(
        train_fingerprints, 
        train_y, 
        RandomForestClassifier,
        params
    )

    max_tau_idx = rf_result.val_tau.argmax(axis = 0)
    best_params = rf_result.iloc[max_tau_idx][:5].to_dict()
    
    rf = random.Random(**best_params)
    rf.fit(train_fingerprints, train_y)
    pred = rf.predict(test_fingerprints)
      
    result_ = {
        'seed': seed_,
        'parameters': best_params,
        'precision': precision_score(test_y, pred, average = 'macro'), 
        'recall': recall_score(test_y, pred, average = 'macro'), 
        'f1': f1_score(test_y, pred, average = 'macro'), 
        'accuracy': accuracy_score(test_y, pred),
        'tau': stats.kendalltau(test_y, pred).correlation
    }
            

    wandb.log({
        'seed': seed_,
        'parameters': best_params,
        'precision': precision_score(test_y, pred, average = 'macro'), 
        'recall': recall_score(test_y, pred, average = 'macro'), 
        'f1': f1_score(test_y, pred, average = 'macro'), 
        'accuracy': accuracy_score(test_y, pred),
        'tau': stats.kendalltau(test_y, pred).correlation
    })
      
      
    return result_


result = []
for seed_ in range(200):
      result.append(rf_main(seed_))
      
pd.DataFrame(result).to_csv('../test_results/rf.csv', header = True, index = False)
wandb.finish()

