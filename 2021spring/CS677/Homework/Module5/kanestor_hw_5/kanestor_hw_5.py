"""
Name: Kimberly Nestor
Class: CS 677 - Spring 2
Date: 04/20/21
Homework #5
Description of Problem: Comparison of performance values between Naive Bayesian
and Decision Tree Classifiers.
Cardiac Dataset: https://archive.ics.uci.edu/ml/datasets/Cardiotocography
"""

import sys
import os
import pandas as pd
import numpy as np
# from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import sklearn.metrics as sk
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import warnings

# pd.set_option("display.max_rows", None, "display.max_columns", None)
warnings.filterwarnings("ignore")

# function to unlist
unlist = lambda nest: [i for list in nest for i in list]

#relative dir info
here = os.path.abspath(__file__)
input_dir = os.path.abspath(os.path.join(here, os.pardir))
data_file = os.path.join(input_dir, 'CTG.xls')

#### QUESTION 1
GP1_LST = ["LB", "ALTV", "Min", "Mean", "NSP"]

# Q1 Part1 - read data file to df
df_fetal = pd.read_excel(data_file, sheet_name="Raw Data")
df_fetal_gp1 = df_fetal[GP1_LST][1:-3]

# Q1 Part2 - class 0 = abnormal, class 1 = normal
#replace fetal class with binary
df_fetal_gp1[GP1_LST[-1]] = [0 if i==2 or i==3 else 1 for i in \
                             df_fetal_gp1[GP1_LST[-1]].values]

#### QUESTION 2 - NAIVE BAYESIAN
# Q2 Part1 - split into test and train
fetal_train_nb, fetal_test_nb = train_test_split(df_fetal_gp1.values, \
                                test_size=0.5, random_state=2026) #random_state

# training data
df_fetal_train_nb = pd.DataFrame(fetal_train_nb, columns=GP1_LST)
X_train = df_fetal_train_nb[GP1_LST[:-1]].values
Y_train = df_fetal_train_nb[GP1_LST[-1]].values

#train
NB_classifier = GaussianNB().fit(X_train, Y_train)

# testing data
df_fetal_test_nb = pd.DataFrame(fetal_test_nb, columns=GP1_LST)
new_instance = df_fetal_test_nb[GP1_LST[:-1]].values

#predict
prediction = NB_classifier.predict(new_instance)
df_fetal_test_nb.insert(loc=5, column='NSP_pred', value=prediction)

# Q2 Part2 - accuracy
nb_acc = round(sk.accuracy_score(df_fetal_test_nb['NSP'], \
                                df_fetal_test_nb['NSP_pred']), 2)
print("\n")
print("Accuracy rating for Naive Bayesian = %.2f" % nb_acc)

# Q2 Part3 - confusion matrix
nb_conmat = sk.confusion_matrix(list(df_fetal_test_nb['NSP']), \
                                list(df_fetal_test_nb['NSP_pred']))


#### QUESTION 3 - DECISION TREE
# Q3 Part1 - split into test and train
fetal_train_dt, fetal_test_dt = train_test_split(df_fetal_gp1.values, \
                                test_size=0.5, random_state=2026)

# training data
df_fetal_train_dt = pd.DataFrame(fetal_train_dt, columns=GP1_LST)
X_train = df_fetal_train_dt[GP1_LST[:-1]].values
Y_train = df_fetal_train_dt[GP1_LST[-1]].values

#train
clf = tree.DecisionTreeClassifier(criterion = 'entropy')
clf = clf.fit(X_train, Y_train)

# testing data
df_fetal_test_dt = pd.DataFrame(fetal_test_dt, columns=GP1_LST)
new_instance = df_fetal_test_dt[GP1_LST[:-1]].values

#predict
prediction = clf.predict(new_instance)
df_fetal_test_dt.insert(loc=5, column='NSP_pred', value=prediction)

# Q2 Part2 - accuracy
dt_acc = round(sk.accuracy_score(df_fetal_test_dt['NSP'], \
                                df_fetal_test_dt['NSP_pred']), 2)

print("Accuracy rating for Decision Tree = %.2f" % dt_acc)

# Q3 Part3 - confusion matrix
dt_conmat = sk.confusion_matrix(list(df_fetal_test_dt['NSP']), \
                                list(df_fetal_test_dt['NSP_pred']))


#### QUESTION 4 - RANDOM FOREST
# Q4 Part1 - test depths and subtrees
#split into test and train
fetal_train_rf, fetal_test_rf = train_test_split(df_fetal_gp1.values, \
                                test_size=0.5, random_state=2026)

# training data
df_fetal_train_rf = pd.DataFrame(fetal_train_rf, columns=GP1_LST)
X_train = df_fetal_train_rf[GP1_LST[:-1]].values
Y_train = df_fetal_train_rf[GP1_LST[-1]].values

# testing data
df_fetal_test_rf = pd.DataFrame(fetal_test_rf, columns=GP1_LST)
Y_test = df_fetal_test_rf[GP1_LST[:-1]].values

# n and d list info
subtree = list(range(1, 11))
maxdepth = list(range(1, 6))

train_crit = []
rf_acc_lst = []
error_lst = []

#loop to text several depths and subtrees
for d in maxdepth:
    for n in subtree:
        # append structure
        train_crit.append([d, n])

        # train
        model = RandomForestClassifier(n_estimators=n, max_depth=d,
                                       criterion='entropy')
        model.fit(X_train, Y_train)

        # predict
        test_instance = Y_test
        rf_label = model.predict(test_instance)

        #accuracy
        rf_acc = round(sk.accuracy_score(df_fetal_test_rf[GP1_LST[-1]].values, rf_label), 2)
        rf_acc_lst.append(rf_acc)

        #error rate
        error_rate = round(np.mean(rf_label != df_fetal_test_rf[GP1_LST[-1]].values), 2)
        error_lst.append(error_rate)

# random forest performance df
df_rf_perf = pd.DataFrame(train_crit, columns=["Depth", "Subtree"])
df_rf_perf.insert(2, column="Acc", value=rf_acc_lst)
df_rf_perf.insert(3, column="Err", value=error_lst)

# Q4 Part2 - plot error rates
plt.plot(df_rf_perf[df_rf_perf["Depth"]==1]["Subtree"], \
         df_rf_perf[df_rf_perf["Depth"]==1]["Err"], marker='o', label="depth 1")
plt.plot(df_rf_perf[df_rf_perf["Depth"]==2]["Subtree"], \
         df_rf_perf[df_rf_perf["Depth"]==2]["Err"], marker='o', label="depth 2")
plt.plot(df_rf_perf[df_rf_perf["Depth"]==3]["Subtree"], \
         df_rf_perf[df_rf_perf["Depth"]==3]["Err"], marker='o', label="depth 3")
plt.plot(df_rf_perf[df_rf_perf["Depth"]==4]["Subtree"], \
         df_rf_perf[df_rf_perf["Depth"]==4]["Err"], marker='o', label="depth 4")
plt.plot(df_rf_perf[df_rf_perf["Depth"]==5]["Subtree"], \
         df_rf_perf[df_rf_perf["Depth"]==5]["Err"], marker='o', label="depth 5")

plt.xlabel("Subtree", fontweight='bold')
plt.xticks(np.arange(1, 11))
plt.ylabel("Error", fontweight='bold')
plt.legend()
plt.savefig(os.path.join(input_dir, 'randfor_err_graph.png'), dpi=200)
plt.show()

# Q4 Part3 - best accuracy, depth = 5, subtree = 8
rf_best_acc = df_rf_perf[df_rf_perf['Depth']==5] [df_rf_perf['Subtree']==8]\
                                                 ['Acc'].values[0]
print("Accuracy rating for best Random Forest = %.2f" % rf_best_acc)


# Q4 Part4 - confusion matrix, best random forest
# train
model = RandomForestClassifier(n_estimators=8, max_depth=5, criterion='entropy')
model.fit(X_train, Y_train)

# predict
test_instance = Y_test
rf_label = model.predict(test_instance)

# confusion matrix
rf_conmat = sk.confusion_matrix(df_fetal_test_rf[GP1_LST[-1]].values, rf_label)


#### QUESTION 5 - summary table

nb_conmat_tot = sum(unlist(nb_conmat))
dt_conmat_tot = sum(unlist(dt_conmat))
rf_conmat_tot = sum(unlist(rf_conmat))


# list of list of performance vals
perf_lst = [ [round(i, 2) for i in [nb_conmat[1][1]/nb_conmat_tot, nb_conmat[0][1]/nb_conmat_tot, \
            nb_conmat[0][0]/nb_conmat_tot, nb_conmat[1][0]/nb_conmat_tot, nb_acc, \
            (nb_conmat[1][1]/(nb_conmat[1][1] + nb_conmat[1][0])), \
            (nb_conmat[0][0]/(nb_conmat[0][0] + nb_conmat[0][1]))]], \

             [round(i, 2) for i in [dt_conmat[1][1] / dt_conmat_tot, dt_conmat[0][1] / dt_conmat_tot, \
               dt_conmat[0][0] / dt_conmat_tot, dt_conmat[1][0] / dt_conmat_tot, dt_acc, \
               (dt_conmat[1][1] / (dt_conmat[1][1] + dt_conmat[1][0])), \
               (dt_conmat[0][0] / (dt_conmat[0][0] + dt_conmat[0][1]))]], \

             [round(i, 2) for i in [rf_conmat[1][1] / rf_conmat_tot, rf_conmat[0][1] / rf_conmat_tot, \
               rf_conmat[0][0] / rf_conmat_tot, rf_conmat[1][0] / rf_conmat_tot, rf_acc, \
               (rf_conmat[1][1] / (rf_conmat[1][1] + rf_conmat[1][0])), \
               (rf_conmat[0][0] / (rf_conmat[0][0] + rf_conmat[0][1]))]] ]


# SUMMARY TABLE
# table header info
tab_headers = ["Model", "TP", "FP", "TN", "FN", "accuracy", "TPR", "TNR", ""]
model_info = ["naive bayesian", "decision tree", "random forest"]

# table format info
format_info = "{:<2} {:<17} {:<8.2} {:<8.2} {:<8.2} {:<8.3} {:<8.3} {:<8.3} {:<8.3}"

# print headers
print("\n")
print("PERFORMANCE SUMMARY TABLE")
print(format_info.format("", *tab_headers))

# print table
for model, perf in zip(model_info, perf_lst):
    print(format_info.format("", model, *perf))
print("\n")



