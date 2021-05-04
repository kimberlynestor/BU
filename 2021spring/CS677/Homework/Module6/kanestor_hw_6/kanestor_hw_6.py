"""
Name: Kimberly Nestor
Class: CS 677 - Spring 2
Date: 04/27/21
Homework #6
Description of Problem: Comparison of performance of various SVM and k means clustering.
Seeds Dataset: https://archive.ics.uci.edu/ml/datasets/seeds
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn import svm
import sklearn.metrics as sk
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import random
from scipy.spatial import distance


# pd.set_option("display.max_rows", None, "display.max_columns", None)

#relative dir info
here = os.path.abspath(__file__)
input_dir = os.path.abspath(os.path.join(here, os.pardir))
data_file = os.path.join(input_dir, 'seeds_dataset.xlsx')

#seeds data
ATTR_INFO = ['area', 'perimeter', 'compactness', 'length of kernel', \
             'width of kernel', 'asymmetry coefficient', 'length of kernel groove', \
             'Class']
df_seeds = pd.read_excel(data_file, names=ATTR_INFO)
df_seeds_r1 = df_seeds[df_seeds.Class.isin(range(2,4))]

class_dict = {1:'Kama', 2:'Rosa', 3:'Canadian'}

#group R0 and R2 respectively
# df_seeds_r1 = df_seeds[df_seeds.Class.isin([1,2])]
# df_seeds_r1 = df_seeds[df_seeds.Class.isin([1,3])]

#### QUESTION 1
#split into test and train
seeds_train, seeds_test = train_test_split(df_seeds_r1.values, \
                                test_size=0.5, random_state=2026) #random_state

#make dataframe
df_seeds_train = pd.DataFrame(seeds_train, columns=ATTR_INFO)
df_seeds_test = pd.DataFrame(seeds_test, columns=ATTR_INFO)

#scale train data
scaler = StandardScaler()
scaler.fit( df_seeds_train[ATTR_INFO[:-1]].values )
seeds_train_nc = scaler.transform( df_seeds_train[ATTR_INFO[:-1]].values )

#scale test data
scaler.fit( df_seeds_test[ATTR_INFO[:-1]].values )
seeds_test_nc = scaler.transform( df_seeds_test[ATTR_INFO[:-1]].values )


# Q1 Part1 - implement linear kernel SVM
#training section
svm_classifier = svm.SVC(kernel='linear')
svm_classifier.fit(seeds_train_nc, df_seeds_train[ATTR_INFO[-1]].values)

#testing section
new_x = np.asmatrix(seeds_test_nc)
predicted = svm_classifier.predict(new_x)

#performance
acc_linsvm = svm_classifier.score (seeds_test_nc, df_seeds_test[ATTR_INFO[-1]].values) #X, Y
conmat_linsvm = sk.confusion_matrix(df_seeds_test[ATTR_INFO[-1]].values, predicted)


# Q1 Part2 - gaussian linear kernel SVM
#training section
svm_classifier = svm.SVC(kernel='rbf')
svm_classifier.fit(seeds_train_nc, df_seeds_train[ATTR_INFO[-1]].values)

#testing section
new_x = np.asmatrix(seeds_test_nc)
predicted = svm_classifier.predict(new_x)

#performance
acc_gaussvm = svm_classifier.score (seeds_test_nc, df_seeds_test[ATTR_INFO[-1]].values)
conmat_gaussvm = sk.confusion_matrix(df_seeds_test[ATTR_INFO[-1]].values, predicted)


# Q1 Part3 - polynomial linear kernel SVM, degree = 3
#training section
svm_classifier = svm.SVC(kernel='poly', degree=3)
svm_classifier.fit(seeds_train_nc, df_seeds_train[ATTR_INFO[-1]].values)

#testing section
new_x = np.asmatrix(seeds_test_nc)
predicted = svm_classifier.predict(new_x)

#performance
acc_polsvm = svm_classifier.score (seeds_test_nc, df_seeds_test[ATTR_INFO[-1]].values)
conmat_polsvm = sk.confusion_matrix(df_seeds_test[ATTR_INFO[-1]].values, predicted)


#### QUESTION 2
# Q2 Part1 - logistic regression
#training section
log_reg_classifier = LogisticRegression()
log_reg_classifier.fit(seeds_train_nc, df_seeds_train[ATTR_INFO[-1]].values)

#testing section
new_x = np.asmatrix(seeds_test_nc)
predicted = log_reg_classifier.predict(new_x)

#performance
acc_logressvm = sk.accuracy_score(df_seeds_test[ATTR_INFO[-1]].values, predicted)
conmat_logressvm = sk.confusion_matrix(df_seeds_test[ATTR_INFO[-1]].values, predicted)


# Q2 Part2 - summary of performance
conmat_tot = sum([ii for i in conmat_linsvm for ii in i])

# list of list of performance vals
perf_lst = [[round(i, 2) for i in
             [conmat_linsvm[1][1] / conmat_tot, conmat_linsvm[0][1] / conmat_tot, \
              conmat_linsvm[0][0] / conmat_tot, conmat_linsvm[1][0] / conmat_tot,
              acc_linsvm, (conmat_linsvm[1][1] / (conmat_linsvm[1][1] + conmat_linsvm[1][0])), \
              (conmat_linsvm[0][0] / (conmat_linsvm[0][0] + conmat_linsvm[0][1]))]],

            [round(i, 2) for i in
             [conmat_gaussvm[1][1] / conmat_tot, conmat_gaussvm[0][1] / conmat_tot, \
              conmat_gaussvm[0][0] / conmat_tot, conmat_gaussvm[1][0] / conmat_tot,
              acc_gaussvm, (conmat_gaussvm[1][1] / (conmat_gaussvm[1][1] + conmat_gaussvm[1][0])), \
              (conmat_gaussvm[0][0] / (conmat_gaussvm[0][0] + conmat_gaussvm[0][1]))]],

            [round(i, 2) for i in
             [conmat_polsvm[1][1] / conmat_tot, conmat_polsvm[0][1] / conmat_tot, \
              conmat_polsvm[0][0] / conmat_tot, conmat_polsvm[1][0] / conmat_tot,
              acc_polsvm, (conmat_polsvm[1][1] / (conmat_polsvm[1][1] + conmat_polsvm[1][0])), \
              (conmat_polsvm[0][0] / (conmat_polsvm[0][0] + conmat_polsvm[0][1]))]],

           [round(i, 2) for i in
            [conmat_logressvm[1][1] / conmat_tot, conmat_logressvm[0][1] / conmat_tot, \
              conmat_logressvm[0][0] / conmat_tot, conmat_logressvm[1][0] / conmat_tot,
              acc_logressvm, (conmat_logressvm[1][1] / (conmat_logressvm[1][1] + conmat_logressvm[1][0])), \
              (conmat_logressvm[0][0] / (conmat_logressvm[0][0] + conmat_logressvm[0][1]))]]
            ]


# SUMMARY TABLE
# table header info
tab_headers = ["Model", "TP", "FP", "TN", "FN", "accuracy", "TPR", "TNR", ""]
model_info = ["linear SVM", "Gaussian SVM", "polynomial SVM", "logistic regression"]

# table format info
format_info = "{:<2} {:<20} {:<8.2} {:<8.2} {:<8.2} {:<8.3} {:<8.3} {:<8.3} {:<8.3}"

# print headers
print("\n")
print("PERFORMANCE SUMMARY TABLE")
print(format_info.format("", *tab_headers))

# print table
for model, perf in zip(model_info, perf_lst):
    print(format_info.format("", model, *perf))
print("\n")


#### QUESTION 3
# Q3 Part1 - kmeans, k = 1,2,...8, k = init centroids

#run loop of kmean algorithm
kmean_lst = []
centr_lst = []
inert_lst = []
for k in range(1,9):
    #scale data
    scaler = StandardScaler()
    scaler.fit(df_seeds[ATTR_INFO[:-1]].values)
    seeds_scale_nc = scaler.transform(df_seeds[ATTR_INFO[:-1]].values)

    #predict
    kmeans_classifier = KMeans(n_clusters=k, init='random')
    y_kmeans = kmeans_classifier.fit_predict(seeds_scale_nc)
    centroids = kmeans_classifier.cluster_centers_
    inertia = kmeans_classifier.inertia_
    #append
    kmean_lst.append(y_kmeans)
    centr_lst.append(centroids)
    inert_lst.append(inertia)

#plot inertia of kmeans clusters 1-8
plt.plot(range(1,9), inert_lst, marker='o')
plt.xlabel("k clusters", fontweight='bold')
plt.ylabel("Inertia", fontweight='bold')
plt.title("Find best k using knee method", fontweight='bold')
plt.savefig(os.path.join(input_dir, 'find_bestk_graph.png'), dpi=200)
plt.show()

best_k = 3
print("Best kmean cluster based on knee method = 3", "\n")


# Q3 Part2 - pick two random features, use best k, plot kmeans
#pick two
random.seed(26)
rand_feat = random.sample(ATTR_INFO[:-1], 2)

#scale data
scaler = StandardScaler()
scaler.fit( df_seeds[rand_feat].values )
seeds_scale_nc = scaler.transform( df_seeds[rand_feat].values )

#predict
kmeans_classifier = KMeans(n_clusters=best_k, init='random')
y_kmeans = kmeans_classifier.fit_predict(seeds_scale_nc)
centroids = kmeans_classifier.cluster_centers_
inertia = kmeans_classifier.inertia_
# print(seeds_scale_nc)
#make df
# df_seeds_rand = df_seeds[rand_feat + [ATTR_INFO[-1]]]
df_seeds_rand = pd.DataFrame(seeds_scale_nc, columns=rand_feat)
df_seeds_rand.insert(2, column='Class', value=df_seeds[ATTR_INFO[-1]].values)
df_seeds_rand.insert(3, column='cluster', value=y_kmeans)
# print(df_seeds_rand[df_seeds_rand['cluster']==1])

colmap = {0: '#029386', 1: '#D2691E', 2: '#A52A2A'}
#plot data points
for i in range(best_k):
    new_df = df_seeds_rand[df_seeds_rand['cluster']==i]
    plt.scatter(new_df[rand_feat[0]], new_df[rand_feat[1]], s=50, \
                label='cluster' + str(i+1), color=colmap[i])

#plot centroids
for i in range (best_k):
    plt.scatter(centroids[i][0], centroids[i][1], marker='x', s=500, \
            label='centroid' + str(i+1), color=colmap[i])

plt.xlabel(rand_feat[0], fontweight='bold', fontsize=12)
plt.ylabel(rand_feat[1], fontweight='bold', fontsize=12)
plt.title("Random features with centroids", fontweight='bold')
plt.legend()
plt.savefig(os.path.join(input_dir, 'bestk_randfeat_graph.png'), dpi=150)
plt.show()

#plot data points according to real class
# for i in range(1, 4):
#     new_df = df_seeds_rand[df_seeds_rand['Class']==i]
#     plt.scatter(new_df[rand_feat[0]], new_df[rand_feat[1]], s=50, \
#                 label='Class' + str(i+1))
# plt.show()


# Q3 Part3 - majority class in cluster
#number of class vals in cluster
clust_sz_lst = []
for cluster in range(0,3):
    df_cluster = df_seeds_rand[df_seeds_rand['cluster'] == cluster]
    class_lst = [len(df_cluster[df_cluster['Class'] == i]) for i in range(1, 4)]
    clust_sz_lst.append(class_lst)

# find majority class in cluster
max_class_lst = []
for cluster in clust_sz_lst:
    max_class = max(cluster)
    max_class_lst.append(cluster.index(max_class)+1)

#print cluster and centroid info
for i in range(0,3):
    print(f'For cluster{i+1} the max class is Class{max_class_lst[i]} ' + \
          f'{class_dict[max_class_lst[i]]} wheat. ' + \
          f'Cluster{i+1} centroids = {centroids[i]}')
print("\n")


# Q3 Part4 - find Euclidean distance from point to centroids, nearest k
#loop to find all distance, then min distance
near_class_lst = []
for x in seeds_scale_nc:
    cent_dist_lst = []
    for i in centroids:
        dist = distance.euclidean(x, i)
        cent_dist_lst.append(dist)
    class_val = cent_dist_lst.index(min(cent_dist_lst)) + 1
    near_class_lst.append(class_val)

#update df and find accuracy
df_seeds_rand.insert(4, column='NearCent', value=near_class_lst)
nearcent_acc = round(sk.accuracy_score(df_seeds_rand['Class'].values, \
                                       df_seeds_rand['NearCent']), 2)
print(f'Accuracy of nearest centroid classifier compared to real class = '+\
      f'{nearcent_acc}', "\n")


# Q3 Part5 - compare performance values
conmat_nearcent = sk.confusion_matrix(df_seeds_rand['Class'].values, df_seeds_rand['NearCent'])
conmat_tot = sum([ii for i in conmat_nearcent for ii in i])

nearcent_perf_lst = [round(i, 2) for i in [conmat_nearcent[1][1] / conmat_tot, \
                    conmat_nearcent[0][1] / conmat_tot, conmat_nearcent[0][0] / conmat_tot, \
                    conmat_nearcent[1][0] / conmat_tot, nearcent_acc, \
                    (conmat_nearcent[1][1] / (conmat_nearcent[1][1] + conmat_nearcent[1][0])), \
                    (conmat_nearcent[0][0] / (conmat_nearcent[0][0] + conmat_nearcent[0][1]))]]

print("PERFORMANCE OF NEAREST CENTROID CLASSIFIER")
format_info = "{:<2} {:<20} {:<8.2} {:<8.2} {:<8.2} {:<8.3} {:<8.3} {:<8.3} {:<8.3}"
print(format_info.format("", "nearest centroid", *nearcent_perf_lst), "\n")
