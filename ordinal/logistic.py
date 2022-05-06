#%%
from utils import *

import time
import random
import warnings

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from statsmodels.miscmodels.ordinal_model import OrderedModel

import scipy.stats as stats
from sklearn.metrics import classification_report

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
# mgl_ordinal = OrderedModel(
#       train_mgl_y.category,
#       train_mgl_fingerprints, 
#       distr = 'logit')

# mgl_ordinal_fit = mgl_ordinal.fit(method = 'lbfgs', maxiter = 1000)
# mgl_ordinal_fit.summary()

# mgl_pred_prob = mgl_ordinal_fit.predict(test_mgl_fingerprints)
# mgl_ord_pred = np.argmax(np.array(mgl_pred_prob), axis = 1)

# print("kendall's tau = ", stats.kendalltau(test_mgl_y.category, mgl_ord_pred))
# print(classification_report(test_mgl_y.category, mgl_ord_pred, digits = 5))


#%%
from sklearn.base import clone, BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted, check_array
from sklearn.utils.multiclass import check_classification_targets

class LogitOrdinalClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self, 
        penalty='l2', 
        *, 
        dual=False, 
        tol=0.0001, 
        C=1.0, 
        fit_intercept=True, 
        intercept_scaling=1, 
        class_weight=None, 
        random_state=None, 
        solver='lbfgs', 
        max_iter=100, 
        multi_class='auto', 
        verbose=0, 
        warm_start=False, 
        n_jobs=None, 
        l1_ratio=None
    ):
        self.penalty = penalty 
        self.dual = dual
        self.tol = tol
        self.C = C
        self.fit_intercept = fit_intercept 
        self.intercept_scaling = intercept_scaling
        self.class_weight = class_weight
        self.random_state = random_state
        self.solver = solver
        self.max_iter = max_iter
        self.multi_class = multi_class
        self.verbose = verbose
        self.warm_start = warm_start
        self.n_jobs = n_jobs
        self.l1_ratio = l1_ratio
        
    def fit(self, X, y):
        X, y = check_X_y(X, y)
        check_classification_targets(y)
        
        self.clf_ = LogisticRegression(**self.get_params())
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
    n_estimators = 60,
    max_features = 'auto',
    min_samples_split = 3
)

mgl_ordrf.fit(train_mgl_fingerprints, train_mgl_y.category)

mgl_ord_pred = mgl_ordrf.predict(test_mgl_fingerprints)

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
      C = 0.5,
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