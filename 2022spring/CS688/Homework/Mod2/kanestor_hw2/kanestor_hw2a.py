"""
Name: Kimberly Nestor
Class: CS 688 - Spring 2
Date: 03/31/22
Homework Problem: Hw 2A
Description of Problem: This program uses logistic regression to predict which
buyer used a credit card. Examines different hyperparameters and feature generation.
Dataset:
train, test = validation 0, 1
neg, pos = buyer 0, 1 - logreg default 1=pos
"""

import sys
from os.path import join as opj

import pandas as pd
import numpy as np

import sklearn.metrics as sk
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures
from joblib import dump, load

# pd.set_option('display.max_rows', None)


# dataset info
cardkey = {1: 'amex', 2: 'store', 3: 'visa', 4: 'master', 5: 'discover'}

categorykey = {1:'grocery', 2:'gas & car maintenance', 3: 'dining', 4:'misc', \
               5:'health', 6:'education', 7:'travel', 8:'home improvement & furnishings', \
               9:'cell phones, cable & utilities', 10:'books', 11:'clothing'}

# load credit card data
curr_dir = sys.path[0]
df_cc = pd.read_csv(opj(curr_dir, 'creditcardspend.csv')).iloc[:,:-3]
# print(df_cc)


# separate train and test, shuffle data
df_cc_train = df_cc[df_cc['Validation'] == 0].sample(frac=1, random_state=1).iloc[:,:-1]
df_cc_test = df_cc[df_cc['Validation'] == 1].sample(frac=1, random_state=1).iloc[:,:-1]

# class info
train_class = df_cc_train.iloc[:,-1].values
test_class = df_cc_test.iloc[:,-1].values


# scale train data
scaler = StandardScaler()
scaler.fit( df_cc_train.iloc[:,:-1].values )
cc_train = scaler.transform( df_cc_train.iloc[:,:-1].values )
# cc_train = df_cc_train.iloc[:,:-1].values

# scale test data
scaler.fit( df_cc_test.iloc[:,:-1].values )
cc_test = scaler.transform( df_cc_test.iloc[:,:-1].values )
# cc_test = df_cc_test.iloc[:,:-1].values


#### Q2 SUPERVISED LEARNING - Logistic Regression
# train
lr_clf = LogisticRegression(solver='liblinear', random_state=1)
lr_clf.fit(cc_train, train_class) # X, Y

# test
new_instance = np.asmatrix(cc_test)
predict = lr_clf.predict(new_instance)
# df_cc_test.insert(len(df_cc_test.columns), column='LogRes', value=predict)

# performance
rec_lr = round(sk.recall_score(test_class, predict), 2)
pre_lr = round(sk.precision_score(test_class, predict), 2)
f1_lr = round(sk.f1_score(test_class, predict), 2)
acc_lr = round(sk.accuracy_score(test_class, predict), 2)
print('Logistic Regression')
print('  Recall: {}'.format(rec_lr))
print('  Precision: {}'.format(pre_lr))
print('  F1 score: {}'.format(f1_lr))
print('  Accuracy: {}\n'.format(acc_lr))


#### Q3 OPTIMIZATION
## hyperparameter, standard scaler, intercept scaling = 0.3
# train
lr_clf = LogisticRegression(solver='liblinear', random_state=1, intercept_scaling=.3)
lr_clf.fit(cc_train, train_class) # X, Y

# test
new_instance = np.asmatrix(cc_test)
predict = lr_clf.predict(new_instance)

# performance
rec_lr = round(sk.recall_score(test_class, predict), 2)
pre_lr = round(sk.precision_score(test_class, predict), 2)
f1_lr = round(sk.f1_score(test_class, predict), 2)
acc_lr = round(sk.accuracy_score(test_class, predict), 2)
print('Logistic Regression, intercept scaling = 0.3')
print('  Recall: {}'.format(rec_lr))
print('  Precision: {}'.format(pre_lr))
print('  F1 score: {}'.format(f1_lr))
print('  Accuracy: {}\n'.format(acc_lr))

# save model info
dump(lr_clf, opj(curr_dir, 'logreg_model.joblib'))
# clf = load('logreg_model.joblib')
# print(lr_clf)
# print(lr_clf.coef_)

## hyperparameter, standard scaler, intercept scaling = 0.5
# train
lr_clf = LogisticRegression(solver='liblinear', random_state=1, intercept_scaling=.5)
lr_clf.fit(cc_train, train_class) # X, Y

# test
new_instance = np.asmatrix(cc_test)
predict = lr_clf.predict(new_instance)

# performance
rec_lr = round(sk.recall_score(test_class, predict), 2)
pre_lr = round(sk.precision_score(test_class, predict), 2)
f1_lr = round(sk.f1_score(test_class, predict), 2)
acc_lr = round(sk.accuracy_score(test_class, predict), 2)
print('Logistic Regression, intercept scaling = 0.5')
print('  Recall: {}'.format(rec_lr))
print('  Precision: {}'.format(pre_lr))
print('  F1 score: {}'.format(f1_lr))
print('  Accuracy: {}\n'.format(acc_lr))


## feature engineering, transforms min max
# scale train data
scaler = MinMaxScaler()
# scaler = PolynomialFeatures()
scaler.fit( df_cc_train.iloc[:,:-1].values )
cc_train = scaler.transform( df_cc_train.iloc[:,:-1].values )

# scale test data
scaler.fit( df_cc_test.iloc[:,:-1].values )
cc_test = scaler.transform( df_cc_test.iloc[:,:-1].values )

# train
lr_clf = LogisticRegression(solver='liblinear', random_state=1)
lr_clf.fit(cc_train, train_class) # X, Y

# test
new_instance = np.asmatrix(cc_test)
predict = lr_clf.predict(new_instance)

# performance
rec_lr = round(sk.recall_score(test_class, predict), 2)
pre_lr = round(sk.precision_score(test_class, predict), 2)
f1_lr = round(sk.f1_score(test_class, predict), 2)
acc_lr = round(sk.accuracy_score(test_class, predict), 2)
print('Logistic Regression with MinMaxScaler')
print('  Recall: {}'.format(rec_lr))
print('  Precision: {}'.format(pre_lr))
print('  F1 score: {}'.format(f1_lr))
print('  Accuracy: {}\n'.format(acc_lr))
