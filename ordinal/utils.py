import openpyxl

import pandas as pd
import numpy as np

from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import MACCSkeys

from itertools import product
from collections.abc import Iterable

import scipy.stats as stats
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score


def Smiles2Fing(smiles):
    ms_tmp = [Chem.MolFromSmiles(i) for i in smiles]
    ms_none_idx = [i for i in range(len(ms_tmp)) if ms_tmp[i] == None]
    
    ms = list(filter(None, ms_tmp))
    
    maccs = [MACCSkeys.GenMACCSKeys(i) for i in ms]
    maccs_bit = [i.ToBitString() for i in maccs]
    
    fingerprints = pd.DataFrame({'maccs': maccs_bit})
    fingerprints = fingerprints['maccs'].str.split(pat = '', n = 167, expand = True)
    fingerprints.drop(fingerprints.columns[0], axis = 1, inplace = True)
    
    colname = ['maccs_' + str(i) for i in range(1, 168)]
    fingerprints.columns = colname
    fingerprints = fingerprints.astype(int).reset_index(drop = True)
    
    return ms_none_idx, fingerprints


def train_mgl_load(path):
    train_mgl = pd.read_excel(path + 'train_mgl.xlsx', sheet_name = 'Sheet1')
    
    # smiles to fingerprints
    mgl_drop_idx, train_mgl_fingerprints = Smiles2Fing(train_mgl.SMILES)
    train_mgl_y = train_mgl.value.drop(mgl_drop_idx).reset_index(drop = True)
    
    # LC50 데이터의 범주 구성
    train_mgl_y = pd.DataFrame(
        {'value': train_mgl_y,
         'category': pd.cut(train_mgl_y, 
                            bins = [0, 0.5, 2.0, 10, 20, np.infty], 
                            labels = range(5))}
        )
    # quantile 기준으로 범주 구성
    # mgl_y = pd.DataFrame({'value': mgl_y, 'category': pd.qcut(mgl_y, 5, labels = range(5))})
    
    # ppm
    return(train_mgl,
           train_mgl_fingerprints, 
           train_mgl_y)


def train_ppm_load(path):
    train_ppm = pd.read_excel(path + 'train_ppm.xlsx', sheet_name = 'Sheet1')

    ppm_drop_idx, train_ppm_fingerprints = Smiles2Fing(train_ppm.SMILES)
    train_ppm_y = train_ppm.value.drop(ppm_drop_idx).reset_index(drop = True)
    
    train_ppm_y = pd.DataFrame(
        {'value': train_ppm_y,
         'category': pd.cut(train_ppm_y,
                            bins = [0, 100, 500, 2500, 20000, np.infty], 
                            labels = range(5))}
        )
    # ppm_y = pd.DataFrame({'value': ppm_y, 'category': pd.qcut(ppm_y, 5, labels = range(5))})
    
    return(train_ppm,
           train_ppm_fingerprints,
           train_ppm_y)


def test_mgl_load(path):
    # mg/l
    test_mgl = pd.read_excel(path + 'test_mgl.xlsx', sheet_name = 'Sheet1')
    
    # smiles to fingerprints
    mgl_drop_idx, test_mgl_fingerprints = Smiles2Fing(test_mgl.SMILES)
    test_mgl_y = test_mgl.value.drop(mgl_drop_idx).reset_index(drop = True)
    
    # LC50 데이터의 범주 구성
    test_mgl_y = pd.DataFrame(
        {'value': test_mgl_y,
         'category': pd.cut(test_mgl_y, 
                            bins = [0, 0.5, 2.0, 10, 20, np.infty], 
                            labels = range(5))}
        )
    # quantile 기준으로 범주 구성
    # mgl_y = pd.DataFrame({'value': test_mgl_y, 'category': pd.qcut(test_mgl_y, 5, labels = range(5))})
    
    return(test_mgl,
           test_mgl_fingerprints, 
           test_mgl_y)


def test_ppm_load(path):
    test_ppm = pd.read_excel(path + 'test_ppm.xlsx', sheet_name = 'Sheet1')
    
    ppm_drop_idx, test_ppm_fingerprints = Smiles2Fing(test_ppm.SMILES)
    test_ppm_y = test_ppm.value.drop(ppm_drop_idx).reset_index(drop = True)
    
    test_ppm_y = pd.DataFrame(
        {'value': test_ppm_y,
         'category': pd.cut(test_ppm_y,
                            bins = [0, 100, 500, 2500, 20000, np.infty], 
                            labels = range(5))}
        )
    # ppm_y = pd.DataFrame({'value': test_ppm_y, 'category': pd.qcut(test_ppm_y, 5, labels = range(5))})
    
    return(test_ppm,
           test_ppm_fingerprints,
           test_ppm_y)


def ParameterGrid(param_dict):
    if not isinstance(param_dict, dict):
        raise TypeError('Parameter grid is not a dict ({!r})'.format(param_dict))
    
    if isinstance(param_dict, dict):
        for key in param_dict:
            if not isinstance(param_dict[key], Iterable):
                raise TypeError('Parameter grid value is not iterable '
                                '(key={!r}, value={!r})'.format(key, param_dict[key]))
    
    items = sorted(param_dict.items())
    keys, values = zip(*items)
    
    params_grid = []
    for v in product(*values):
        params_grid.append(dict(zip(keys, v))) 
    
    return params_grid



def CV(x, y, model, params_grid):
    kf = KFold(n_splits = 5)
    
    result_ = []
    metrics = ['macro_precision', 'weighted_precision', 'macro_recall', 
               'weighted_recall', 'macro_f1', 'weighted_f1', 
               'accuracy', 'tau']
    
    train_metrics = list(map(lambda x: 'train_' + x, metrics))
    val_metrics = list(map(lambda x: 'val_' + x, metrics))
    
    for i in tqdm(range(len(params_grid))):
        train_macro_precision_, train_weighted_precision_ = [], []
        train_macro_recall_, train_weighted_recall_ = [], []
        train_macro_f1_, train_weighted_f1_ = [], []
        train_accuracy_, train_tau_ = [], []
        
        val_macro_precision_, val_weighted_precision_ = [], []
        val_macro_recall_, val_weighted_recall_ = [], []
        val_macro_f1_, val_weighted_f1_ = [], []
        val_accuracy_, val_tau_ = [], []
        
        for train_idx, val_idx in kf.split(x):
            train_x, train_y = x.iloc[train_idx], y.category.iloc[train_idx]
            val_x, val_y = x.iloc[val_idx], y.category.iloc[val_idx]
            
            clf = model(**params_grid[i])
            clf.fit(train_x, train_y)
            
            train_pred = clf.predict(train_x)
            val_pred = clf.predict(val_x)
            
            train_macro_precision_.append(precision_score(train_y, train_pred, average = 'macro'))
            train_weighted_precision_.append(precision_score(train_y, train_pred, average = 'weighted'))
            train_macro_recall_.append(recall_score(train_y, train_pred, average = 'macro'))
            train_weighted_recall_.append(recall_score(train_y, train_pred, average = 'weighted'))
            train_macro_f1_.append(f1_score(train_y, train_pred, average = 'macro'))
            train_weighted_f1_.append(f1_score(train_y, train_pred, average = 'weighted'))
            train_accuracy_.append(accuracy_score(train_y, train_pred))
            train_tau_.append(stats.kendalltau(train_y, train_pred))

            val_macro_precision_.append(precision_score(val_y, val_pred, average = 'macro'))
            val_weighted_precision_.append(precision_score(val_y, val_pred, average = 'weighted'))
            val_macro_recall_.append(recall_score(val_y, val_pred, average = 'macro'))
            val_weighted_recall_.append(recall_score(val_y, val_pred, average = 'weighted'))
            val_macro_f1_.append(f1_score(val_y, val_pred, average = 'macro'))
            val_weighted_f1_.append(f1_score(val_y, val_pred, average = 'weighted'))
            val_accuracy_.append(accuracy_score(val_y, val_pred))
            val_tau_.append(stats.kendalltau(val_y, val_pred))
            
        result_.append(dict(
            zip(list(params_grid[i].keys()) + train_metrics + val_metrics, 
                list(params_grid[i].values()) + 
                [np.mean(train_macro_precision_), 
                 np.mean(train_weighted_precision_),
                 np.mean(train_macro_recall_), 
                 np.mean(train_weighted_recall_),
                 np.mean(train_macro_f1_), 
                 np.mean(train_weighted_f1_),
                 np.mean(train_accuracy_), 
                 np.mean(train_tau_),
                 np.mean(val_macro_precision_), 
                 np.mean(val_weighted_precision_),
                 np.mean(val_macro_recall_), 
                 np.mean(val_weighted_recall_),
                 np.mean(val_macro_f1_), 
                 np.mean(val_weighted_f1_),
                 np.mean(val_accuracy_), 
                 np.mean(val_tau_)])))
        
    result = pd.DataFrame(result_)
    return(result)


def BinaryCV(x, y, model, params_grid):
    kf = KFold(n_splits = 5)
    
    result_ = []
    metrics = ['macro_precision', 'weighted_precision', 'macro_recall', 
               'weighted_recall', 'macro_f1', 'weighted_f1', 
               'accuracy', 'tau', 'auc']
    
    train_metrics = list(map(lambda x: 'train_' + x, metrics))
    val_metrics = list(map(lambda x: 'val_' + x, metrics))
    
    for i in tqdm(range(len(params_grid))):
        train_macro_precision_, train_weighted_precision_ = [], []
        train_macro_recall_, train_weighted_recall_ = [], []
        train_macro_f1_, train_weighted_f1_ = [], []
        train_accuracy_, train_tau_ = [], []
        train_auc_ = []
        
        val_macro_precision_, val_weighted_precision_ = [], []
        val_macro_recall_, val_weighted_recall_ = [], []
        val_macro_f1_, val_weighted_f1_ = [], []
        val_accuracy_, val_tau_ = [], []
        val_auc_ = []
        
        for train_idx, val_idx in kf.split(x):
            train_x, train_y = x.iloc[train_idx], y.category.iloc[train_idx]
            val_x, val_y = x.iloc[val_idx], y.category.iloc[val_idx]
            
            clf = model(**params_grid[i])
            clf.fit(train_x, train_y)
            
            train_pred = clf.predict(train_x)
            val_pred = clf.predict(val_x)
            
            train_y = [0 if i != 1 else 1 for i in train_y]
            train_pred = [0 if i != 1 else 1 for i in train_pred]
            
            val_y = [0 if i != 1 else 1 for i in val_y]
            val_pred = [0 if i != 1 else 1 for i in val_pred]
            
            train_macro_precision_.append(precision_score(train_y, train_pred, average = 'macro'))
            train_weighted_precision_.append(precision_score(train_y, train_pred, average = 'weighted'))
            train_macro_recall_.append(recall_score(train_y, train_pred, average = 'macro'))
            train_weighted_recall_.append(recall_score(train_y, train_pred, average = 'weighted'))
            train_macro_f1_.append(f1_score(train_y, train_pred, average = 'macro'))
            train_weighted_f1_.append(f1_score(train_y, train_pred, average = 'weighted'))
            train_accuracy_.append(accuracy_score(train_y, train_pred))
            train_tau_.append(stats.kendalltau(train_y, train_pred))
            train_auc_.append(roc_auc_score(train_y, train_pred))

            val_macro_precision_.append(precision_score(val_y, val_pred, average = 'macro'))
            val_weighted_precision_.append(precision_score(val_y, val_pred, average = 'weighted'))
            val_macro_recall_.append(recall_score(val_y, val_pred, average = 'macro'))
            val_weighted_recall_.append(recall_score(val_y, val_pred, average = 'weighted'))
            val_macro_f1_.append(f1_score(val_y, val_pred, average = 'macro'))
            val_weighted_f1_.append(f1_score(val_y, val_pred, average = 'weighted'))
            val_accuracy_.append(accuracy_score(val_y, val_pred))
            val_tau_.append(stats.kendalltau(val_y, val_pred))
            val_auc_.append(roc_auc_score(val_y, val_pred))
            
        result_.append(dict(
            zip(list(params_grid[i].keys()) + train_metrics + val_metrics, 
                list(params_grid[i].values()) + 
                [np.mean(train_macro_precision_), 
                 np.mean(train_weighted_precision_),
                 np.mean(train_macro_recall_), 
                 np.mean(train_weighted_recall_),
                 np.mean(train_macro_f1_), 
                 np.mean(train_weighted_f1_),
                 np.mean(train_accuracy_), 
                 np.mean(train_tau_),
                 np.mean(train_auc_),
                 np.mean(val_macro_precision_), 
                 np.mean(val_weighted_precision_),
                 np.mean(val_macro_recall_), 
                 np.mean(val_weighted_recall_),
                 np.mean(val_macro_f1_), 
                 np.mean(val_weighted_f1_),
                 np.mean(val_accuracy_), 
                 np.mean(val_tau_),
                 np.mean(val_auc_)])))
        
    result = pd.DataFrame(result_)
    return(result)

