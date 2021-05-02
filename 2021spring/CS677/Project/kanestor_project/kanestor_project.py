"""
Name: Kimberly Nestor
Class: CS 677 - Spring 2
Date: 04/29/21
Final Project
Description of Problem: Comparison of performance of models to classify seven
species of Dry Beans.
Dry Bean Dataset: https://archive.ics.uci.edu/ml/datasets/Dry+Bean+Dataset
"""

import sys
import os
import warnings

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

import shap
import sklearn.metrics as sk
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from xgboost.sklearn import XGBRegressor
from sklearn import tree
from sklearn import svm

# pd.set_option("display.max_rows", None, "display.max_columns", None)
pd.set_option('display.width', None)
warnings.filterwarnings("ignore")

# shap virtual environment information
# conda create -n shap-env python=3.8
# conda activate shap-env
# conda install -c conda-forge shap

#relative dir info
here = os.path.abspath(__file__)
input_dir = os.path.abspath(os.path.join(here, os.pardir))
data_file = os.path.join(input_dir, 'Dry_Bean_Dataset.xlsx')

#beans data
ATTR_INFO_ALL = ['Area', 'Perimeter', 'MajorAxisLength', 'MinorAxisLength', \
                 'AspectRation', 'Eccentricity', 'ConvexArea', 'EquivDiameter', \
                 'Extent', 'Solidity', 'roundness', 'Compactness',  'Class']

ATTR_INFO = ['MinorAxisLength', 'Compactness', 'roundness', 'Class']

df_beans_full = pd.read_excel(data_file)
df_beans = df_beans_full[ATTR_INFO]

class_dict = {0:'SEKER', 1:'BARBUNYA', 2:'BOMBAY', 3:'CALI', 4:'HOROZ', \
              5:'SIRA', 6:'DERMASON'}

#change species to numbers
class_bin = [i  for i in class_dict for species in df_beans[ATTR_INFO[-1]] if \
             species == class_dict[i] ]
df_beans.insert(len(df_beans.columns), column='ClassBin', value=class_bin)
ATTR_INFO = ATTR_INFO + ['ClassBin']

#split into test and train
beans_train, beans_test = train_test_split(df_beans.values, \
                            test_size=0.5, random_state=2022) #random_state

#make dataframe
df_beans_train = pd.DataFrame(beans_train, columns=ATTR_INFO)
df_beans_test = pd.DataFrame(beans_test, columns=ATTR_INFO)

classbin_train = list(map(int, df_beans_train[ATTR_INFO[-1]].values))
classbin_test = list(map(int, df_beans_test[ATTR_INFO[-1]].values))

#scale train data
scaler = StandardScaler()
scaler.fit( df_beans_train[ATTR_INFO[:-2]].values )
beans_train_nc = scaler.transform( df_beans_train[ATTR_INFO[:-2]].values )

#scale test data
scaler.fit( df_beans_test[ATTR_INFO[:-2]].values )
beans_test_nc = scaler.transform( df_beans_test[ATTR_INFO[:-2]].values )


#### SUPERVISED LEARNING
## Logistic Regression
#train
log_reg_classifier = LogisticRegression()
log_reg_classifier.fit(beans_train_nc, classbin_train) # X, Y

#test
new_instance = np.asmatrix(beans_test_nc)
predict = log_reg_classifier.predict(new_instance)
df_beans_test.insert(len(df_beans.columns), column='LogRes', value=predict)

#performance
acc_logres = round(sk.accuracy_score(classbin_test, predict), 2)
conmat_logres = sk.confusion_matrix(classbin_test, predict)
# print(df_beans_test[['ClassBin', 'LogRes']])


## Decision Trees
#train
dec_tree_classifier = tree.DecisionTreeClassifier(criterion = 'entropy')
dec_tree_classifier = dec_tree_classifier.fit(beans_train_nc, classbin_train) # X, Y

#test
new_instance = beans_test_nc
predict = dec_tree_classifier.predict(new_instance)
df_beans_test.insert(len(df_beans.columns), column='DecTree', value=predict)

#performance
acc_dectree = round(sk.accuracy_score(classbin_test, predict), 2)
conmat_dectree = sk.confusion_matrix(classbin_test, predict)
# print(df_beans_test[['ClassBin', 'DecTree']])


## Support Vector Machines - linear
#train
svm_classifier = svm.SVC(kernel='linear')
svm_classifier.fit(beans_train_nc, classbin_train)

#test
new_instance = np.asmatrix(beans_test_nc)
predict = svm_classifier.predict(new_instance)
df_beans_test.insert(len(df_beans.columns), column='LinSVM', value=predict)

#performance
acc_linsvm = round(svm_classifier.score(beans_test_nc, \
                                list(df_beans_test['ClassBin'].values)), 2)
conmat_linsvm = sk.confusion_matrix(classbin_test, predict)
# print(df_beans_test[['ClassBin', 'LinSVM']])


#### STATS
#number of classification
num_species = [len(df_beans_test[df_beans_test['Class']==class_dict[i]]) \
               for i in class_dict]
num_species_logres = [len(df_beans_test[df_beans_test['LogRes']==i]) \
                       for i in class_dict]
num_species_dectree = [len(df_beans_test[df_beans_test['DecTree']==i]) \
                       for i in class_dict]
num_species_linsvm = [len(df_beans_test[df_beans_test['LinSVM']==i]) \
                      for i in class_dict]

tot_class = sum(num_species)

#feature mean and std
feat_mean = [round(np.mean(df_beans[ATTR_INFO[i]].values), 2) for i in range(3)]
feat_std = [round(np.std(df_beans[ATTR_INFO[i]].values), 2) for i in range(3)]


#### GRAPHS
## SHAPLEY TEST
#fit model
shap_model = XGBRegressor(n_estimators=1000, max_depth=10, learning_rate=0.001)
shap_model.fit(df_beans_full[ATTR_INFO_ALL[:-1]].values, class_bin)
shap.initjs()
explainer = shap.TreeExplainer(shap_model)
shap_values = explainer.shap_values(df_beans_full[ATTR_INFO_ALL[:-1]].values)

#summary plot
shap.summary_plot(shap_values, features=df_beans_full[ATTR_INFO_ALL[:-1]].values, \
                  feature_names=ATTR_INFO_ALL[:-1], show=False)
plt.savefig(os.path.join(input_dir, 'shapley_feat_import_sumplot.png'), dpi=200)
plt.show() #if you know how to change the font size of the xticks please let me know!

#summary barplot
shap.summary_plot(shap_values, features=df_beans_full[ATTR_INFO_ALL[:-1]].values, \
                  feature_names=ATTR_INFO_ALL[:-1], plot_type='bar', show=False)
plt.savefig(os.path.join(input_dir, 'shapley_feat_import_barplot.png'), dpi=200)
plt.show()


## PAIRWISE PLOT
# pairplot = sns.PairGrid(list(map(round, df_beans_full[ATTR_INFO_ALL[:-1]].values)))
pairplot = sns.PairGrid(df_beans[ATTR_INFO[:-2]])
pairplot.map_upper(sns.scatterplot, color='#A91B0D')
pairplot.map_lower(sns.scatterplot, color='#A91B0D')
pairplot.map_diag(plt.hist, color='#A91B0D')
plt.savefig(os.path.join(input_dir, 'feat_main3_pairplot.png'), dpi=300)
plt.show()


## BARPLOT - num real vs classified species
barWidth = 0.22

#position of bars
r1 = np.arange(len(num_species))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
r4 = [x + barWidth for x in r3]

#plot
plt.bar(r1, num_species, width=barWidth, edgecolor='white', color='#30453B', \
        label='actual')
plt.bar(r2, num_species_logres, width=barWidth, edgecolor='white', color='#4B6C5C', \
        label='logistic regression')
plt.bar(r3, num_species_dectree, width=barWidth, edgecolor='white', color='#6F897D', \
        label='decision tree')
plt.bar(r4, num_species_linsvm, width=barWidth, edgecolor='white', color='#A3B4AC', \
        label='SVM')
plt.xlabel("Species", fontsize=10, fontweight='bold')
plt.ylabel("Number", fontsize=10, fontweight='bold')
plt.xticks([i + barWidth for i in range(len(num_species))], \
           [class_dict[i] for i in class_dict], fontsize=8)
plt.legend()
plt.savefig(os.path.join(input_dir, 'num_species_barplot.png'), dpi=200)
plt.show()


## ACCURACY LINE GRAPH
plt.plot([acc_logres, acc_dectree, acc_linsvm], marker='o', color='#008080', markersize=10)
plt.text(0, acc_logres, acc_logres, fontsize=15, fontweight='bold')
plt.text(1, acc_dectree, acc_dectree, fontsize=15, fontweight='bold')
plt.text(2, acc_linsvm, acc_linsvm, fontsize=15, fontweight='bold')

plt.xticks(np.arange(0,3), ['logistic\nregression', 'decision tree', 'SVM'])
plt.xlabel("Model", fontsize=15, fontweight='bold')
plt.ylabel("Accuracy", fontsize=15, fontweight='bold')
plt.savefig(os.path.join(input_dir, 'model_acc_lineplot.png'), dpi=200)
plt.show()
