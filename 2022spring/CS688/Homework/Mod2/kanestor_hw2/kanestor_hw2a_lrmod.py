"""
Name: Kimberly Nestor
Class: CS 688 - Spring 2
Date: 03/31/22
Homework Problem: Hw 2A
Description of Problem: This program loads a saved and trained logistic
regression model and uses it to predict same performance values on test data as
well as predict on a new instance.
"""

import sys
from os.path import join as opj

import numpy as np
import pandas as  pd

from joblib import dump, load
from sklearn.preprocessing import StandardScaler
import sklearn.metrics as sk


# load credit card data
curr_dir = sys.path[0]
df_cc = pd.read_csv(opj(curr_dir, 'creditcardspend.csv')).iloc[:,:-3]

# separate train and test, shuffle data
df_cc_test = df_cc[df_cc['Validation'] == 1].sample(frac=1, random_state=1).iloc[:,:-1]

# class info
test_class = df_cc_test.iloc[:,-1].values

# scale test data
scaler = StandardScaler()
scaler.fit( df_cc_test.iloc[:,:-1].values )
cc_test = scaler.transform( df_cc_test.iloc[:,:-1].values )


# load model info
clf = load('logreg_model.joblib')
# print(cc_test)

## PART A Q 5
### test data with load model
new_instance = np.asmatrix(cc_test)
predict = clf.predict(new_instance)
df_cc_test.insert(len(df_cc_test.columns), column='LogRes', value=predict)
# print(df_cc_test)

# performance
rec_lr = round(sk.recall_score(test_class, predict), 2)
pre_lr = round(sk.precision_score(test_class, predict), 2)
f1_lr = round(sk.f1_score(test_class, predict), 2)
acc_lr = round(sk.accuracy_score(test_class, predict), 2)
print('Saved Log Res, intercept scaling = 0.3')
print('  Recall: {}'.format(rec_lr))
print('  Precision: {}'.format(pre_lr))
print('  F1 score: {}'.format(f1_lr))
print('  Accuracy: {}\n'.format(acc_lr))


## PART A Q 6 - predict from new transaction
new_inst = [[3, 1, 135.86]]
scaler.fit( new_inst )
new_test = scaler.transform(new_inst )

predict = clf.predict(new_inst)
print('The new buyer would be: {}\n'.format(predict[0]))
