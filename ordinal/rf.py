#%%
from utils import *

import time
import random
import warnings

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier

import scipy.stats as stats
from sklearn.metrics import classification_report

warnings.filterwarnings("ignore")


#%%
path = 'C:/Users/SOYOUNG/Desktop/github/LC50/data/'
train_mgl, train_mgl_fingerprints, train_mgl_y = train_mgl_load(path) 
train_ppm, train_ppm_fingerprints, train_ppm_y = train_ppm_load(path)

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
      'n_estimators': np.arange(30, 155, 5),
      'min_samples_split': list(range(2, 11)),
      'max_features': ['auto', 'sqrt', 'log2']
}

params = ParameterGrid(params_dict)


#%%
mgl_rf_result = CV(
      train_mgl_fingerprints, 
      train_mgl_y, 
      RandomForestClassifier,
      params)

mgl_rf_result.iloc[mgl_rf_result.val_macro_f1.argmax(axis = 0)]
mgl_rf_result.iloc[mgl_rf_result.val_tau.argmax(axis = 0)]


#%%
plt.plot(mgl_rf_result.n_estimators, mgl_rf_result.train_tau)
plt.plot(mgl_rf_result.n_estimators, mgl_rf_result.val_tau)
plt.title("Kendall's tau")
plt.xlabel('C')
plt.legend(['train', 'validation'])
plt.show()
plt.close()


#%%
mgl_rf = RandomForestClassifier(
      random_state = 0,
      n_estimators = 30,
      max_features = 'auto',
      min_samples_split = 6
)

mgl_rf.fit(train_mgl_fingerprints, train_mgl_y.category)
train_mgl_pred = mgl_rf.predict(train_mgl_fingerprints)
mgl_rf_pred = mgl_rf.predict(test_mgl_fingerprints)

print('train results: \n', pd.crosstab(train_mgl_y.category, train_mgl_pred, rownames = ['true'], colnames = ['pred']))
print("\nkendall's tau = ", stats.kendalltau(train_mgl_y.category, train_mgl_pred))
print('\n', classification_report(train_mgl_y.category, train_mgl_pred, digits = 5))

print('test results: \n', pd.crosstab(test_mgl_y.category, mgl_rf_pred, rownames = ['true'], colnames = ['pred']))
print("\nkendall's tau = ", stats.kendalltau(test_mgl_y.category, mgl_rf_pred))
print('\n', classification_report(test_mgl_y.category, mgl_rf_pred, digits = 5))


#%%
'''
      Ordinal Logistic Regression with mg/l data
'''
# https://stackoverflow.com/questions/57561189/multi-class-multi-label-ordinal-classification-with-sklearn

from sklearn.base import clone, BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted, check_array
from sklearn.utils.multiclass import check_classification_targets

class RFOrdinalClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self, 
        n_estimators=100, 
        *, 
        criterion='gini', 
        max_depth=None, 
        min_samples_split=2, 
        min_samples_leaf=1, 
        min_weight_fraction_leaf=0.0, 
        max_features='auto', 
        max_leaf_nodes=None, 
        min_impurity_decrease=0.0, 
        bootstrap=True, 
        oob_score=False, 
        n_jobs=None, 
        random_state=None, 
        verbose=0, 
        warm_start=False, 
        class_weight=None, 
        ccp_alpha=0.0, 
        max_samples=None
    ):
        self.n_estimators = n_estimators 
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf 
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.warm_start = warm_start
        self.class_weight = class_weight
        self.ccp_alpha = ccp_alpha
        self.max_samples = max_samples
        
    def fit(self, X, y):
        X, y = check_X_y(X, y)
        check_classification_targets(y)
        
        self.clf_ = RandomForestClassifier(**self.get_params())
        self.clfs_ = {}
        self.classes_ = np.sort(np.unique(y))
        if self.classes_.shape[0] > 2:
            for i in range(self.classes_.shape[0]-1):
                # for each k - 1 ordinal value we fit a binary classification problem
                binary_y = (y > self.classes_[i]).astype(np.uint8)
                clf = clone(self.clf_)
                clf.fit(X, binary_y)
                self.clfs_[i] = clf
        return self
    
    def predict_proba(self, X):
        X = check_array(X)
        check_is_fitted(self, ['classes_', 'clf_', 'clfs_'])
        
        clfs_predict = {k:self.clfs_[k].predict_proba(X) for k in self.clfs_}
        predicted = []
        for i,y in enumerate(self.classes_):
            if i == 0:
                # V1 = 1 - Pr(y > V1)
                predicted.append(1 - clfs_predict[y][:,1])
            elif y in clfs_predict:
                # Vi = Pr(y > Vi-1) - Pr(y > Vi)
                 predicted.append(clfs_predict[y-1][:,1] - clfs_predict[y][:,1])
            else:
                # Vk = Pr(y > Vk-1)
                predicted.append(clfs_predict[y-1][:,1])
        return np.vstack(predicted).T
    
    def predict(self, X):
        X = check_array(X)
        check_is_fitted(self, ['classes_', 'clf_', 'clfs_'])
        
        return np.argmax(self.predict_proba(X), axis=1)


#%%
mgl_ord_result = CV(
      train_mgl_fingerprints, 
      train_mgl_y, 
      RFOrdinalClassifier,
      params)

mgl_ord_result.iloc[mgl_ord_result.val_macro_f1.argmax(axis = 0)]
mgl_ord_result.iloc[mgl_ord_result.val_tau.argmax(axis = 0)]


#%%
mgl_ordrf = RFOrdinalClassifier(
    random_state = 0,
    n_estimators = 35,
    max_features = 'log2',
    min_samples_split = 5
)

mgl_ordrf.fit(train_mgl_fingerprints, train_mgl_y.category)

mgl_ord_pred = mgl_ordrf.predict(test_mgl_fingerprints)

print("kendall's tau = ", stats.kendalltau(test_mgl_y.category, mgl_ord_pred))
print(classification_report(test_mgl_y.category, mgl_ord_pred, digits = 5))


#%%
'''
      Logistic Regression with ppm data
'''
ppm_rf_result = CV(
      train_ppm_fingerprints, 
      train_ppm_y, 
      RandomForestClassifier,
      params)


ppm_rf_result.iloc[ppm_rf_result.val_macro_f1.argmax(axis = 0)]
ppm_rf_result.iloc[ppm_rf_result.val_tau.argmax(axis = 0)]


#%%
# plt.plot(ppm_rf_result.C, ppm_rf_result.train_tau)
# plt.plot(ppm_rf_result.C, ppm_rf_result.val_tau)
# plt.title("Kendall's tau")
# plt.xlabel('C')
# plt.legend(['train', 'validation'])
# plt.show()
# plt.close()


#%%
ppm_rf = RandomForestClassifier(
    random_state = 0,
    n_estimators = 35,
    max_features = 'log2',
    min_samples_split = 5
)

ppm_rf.fit(train_ppm_fingerprints, train_ppm_y.category)

train_ppm_pred = ppm_rf.predict(train_ppm_fingerprints)
ppm_rf_pred = ppm_rf.predict(test_ppm_fingerprints)

# print('train results: \n', pd.crosstab(train_ppm_y.category, train_ppm_pred, rownames = ['true'], colnames = ['pred']))
# print("\nkendall's tau = ", stats.kendalltau(train_ppm_y.category, train_ppm_pred))
# print('\n', classification_report(train_ppm_y.category, train_ppm_pred, digits = 5))

print('test results: \n', pd.crosstab(test_ppm_y.category, ppm_rf_pred, rownames = ['true'], colnames = ['pred']))
print("\nkendall's tau = ", stats.kendalltau(test_ppm_y.category, ppm_rf_pred))
print('\n', classification_report(test_ppm_y.category, ppm_rf_pred, digits = 5))


#%%
'''
      Ordinal Logistic Regression with ppm data
'''
ppm_ord_result = CV(
      train_ppm_fingerprints, 
      train_ppm_y, 
      RFOrdinalClassifier,
      params)

ppm_ord_result.iloc[ppm_ord_result.val_macro_f1.argmax(axis = 0)]
ppm_ord_result.iloc[ppm_ord_result.val_tau.argmax(axis = 0)]


#%%
ppm_ordrf = RFOrdinalClassifier(
    random_state = 0,
    n_estimators = 40,
    max_features = 'log2',
    min_samples_split = 3
)

ppm_ordrf.fit(train_ppm_fingerprints, train_ppm_y.category)

ppm_ord_pred = ppm_ordrf.predict(test_ppm_fingerprints)

print("kendall's tau = ", stats.kendalltau(test_ppm_y.category, ppm_ord_pred))
print(classification_report(test_ppm_y.category, ppm_ord_pred, digits = 5))


#%%
'''
      mg/l binary
'''
from sklearn.metrics import cohen_kappa_score, roc_auc_score

mgl_binary = pd.DataFrame({
   'y': [0 if i != 1 else 1 for i in test_mgl_y.category],
   'rf_pred': [0 if i != 1 else 1 for i in mgl_rf_pred],
   'ord_pred': [0 if i != 1 else 1 for i in mgl_ord_pred],
})


print(pd.crosstab(mgl_binary.y, mgl_binary.rf_pred, rownames = ['true'], colnames = ['pred']))
print('cohens kappa = ', cohen_kappa_score(mgl_binary.y, mgl_binary.rf_pred))
print('auc = ', roc_auc_score(mgl_binary.y, mgl_binary.rf_pred))
print(classification_report(mgl_binary.y, mgl_binary.rf_pred, digits = 5))

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
   'rf_pred': [0 if i != 1 else 1 for i in ppm_rf_pred],
   'ord_pred': [0 if i != 1 else 1 for i in ppm_ord_pred],
})


print(pd.crosstab(ppm_binary.y, ppm_binary.rf_pred, rownames = ['true'], colnames = ['pred']))
print('cohens kappa = ', cohen_kappa_score(ppm_binary.y, ppm_binary.rf_pred))
print('auc = ', roc_auc_score(ppm_binary.y, ppm_binary.rf_pred))
print(classification_report(ppm_binary.y, ppm_binary.rf_pred, digits = 5))

print(pd.crosstab(ppm_binary.y, ppm_binary.ord_pred, rownames = ['true'], colnames = ['pred']))
print('cohens kappa = ', cohen_kappa_score(ppm_binary.y, ppm_binary.ord_pred))
print('auc = ', roc_auc_score(ppm_binary.y, ppm_binary.ord_pred))
print(classification_report(ppm_binary.y, ppm_binary.ord_pred, digits = 5))



