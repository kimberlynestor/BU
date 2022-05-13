"""
Name: Kimberly Nestor
Class: CS 688 - Spring 2
Date: 04/11/22
Homework Problem: Hw 4
Description of Problem: This program
Q dataset:
"""

import sys
from os.path import join as opj

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

import random
from itertools import combinations

import editdistance # lev
from scipy.spatial.distance import cityblock # man
from scipy.spatial import distance as dst # euc

import sklearn.metrics as sk
from sklearn.preprocessing import MinMaxScaler
from sklearn import datasets
from sklearn.cluster import KMeans

pd.set_option('display.max_rows', None)

def knn_lev_dist(word, word_lst, k):
    """Function returns words k Levenshtein distance away from the target word."""
    word_lst.remove(word)
    dist_lst = list(map(lambda x: editdistance.eval(word, x), word_lst))
    tup_lst = sorted(list(zip(word_lst, dist_lst)), key=lambda x:x[1])
    knn_lst = [i for i in tup_lst if i[1] <= k]
    return(knn_lst)


def dist_mat(data_lst, type):
    """This function returns a pairwise distance matrix from an input list of
    data, of either euclidean or manhattan distance based on type input."""
    # matrix structure
    n = len(data_lst)
    iuu = np.mask_indices(n, np.triu, 1)
    iul = np.mask_indices(n, np.tril, -1)
    mat = np.eye(n)
    # make list of all poss combos, make matrix
    pairs = list(combinations(data_lst, 2))
    if type == 'man':
        # manhattan dist
        man_dist = list(map(lambda x: cityblock(x[0], x[1]), pairs))
        mat[iuu] = man_dist
        mat[iul] = man_dist
        df_man = pd.DataFrame(mat)
        return(df_man)
    elif type == 'euc':
        # euclidean dist
        euc_dist = list(map(lambda x: dst.euclidean(x[0], x[1]), pairs))
        mat[iuu] = euc_dist
        mat[iul] = euc_dist
        df_euc = pd.DataFrame(mat)
        return (df_euc)


#### Q1 - find cities with similar spelling to 5 targets
# load misspelled city name data
curr_dir = sys.path[0]
city_names = open(opj(curr_dir, 'CS688_Misspelled_City_Names.csv'), 'r')\
                    .read().split('\n')

# set seed, pick 5 random city names, wor
random.seed(0)
city_samp = random.sample(city_names, 5)

# get neighbours for samp words, k <= 2
k = 2
city_samp_knn = [knn_lev_dist(i, city_names, k=k) for i in city_samp]

# print knn words
print(f'\nNeighbouring words at most {k} edits apart using Levenshtein distance.')
for i,j in zip(city_samp, city_samp_knn):
    print(f'\nTarget: {i}\n{[ii[0] for ii in j]}\n')


#### Q3 - pairwise distance matrices of man and euc dist
# iris data
iris = datasets.load_iris().data

# scale iris data using MinMax, select every tenth
scaler = MinMaxScaler()
scaler.fit( iris )
iris_sc = scaler.transform( iris ) [::10]

print('\nManhattan distance matrix on iris data:\n', dist_mat(iris_sc, type='man'), '\n')
print('Euclidean distance matrix on iris data:\n', dist_mat(iris_sc, type='euc'), '\n')


#### Q2 - cluster seeds data
# import seeds data, remove \n and \t
seeds_data = [i.replace('\n', '').split('\t') for i in open(opj(curr_dir, 'seeds_dataset.txt'), 'r').readlines()]

# remove empty string and convert to float
seeds_data_cl = []
for list in seeds_data:
    if '' in list:
        while '' in list:
            list.remove('')
        list = [float(i) for i in list]
        seeds_data_cl.append(list)
    else:
        list = [float(i) for i in list]
        seeds_data_cl.append(list)

# make df with clean data
df_seeds = pd.DataFrame(seeds_data_cl)
# print(df_seeds)

# plot pairplot
sns.pairplot(df_seeds, hue=7)
plt.savefig('pairplot_seeds.png', dpi=300)
plt.show()

# perform kmeans clustering, no target, k=3
kmeans = KMeans(n_clusters=3, random_state=0).fit(df_seeds.iloc[:,0:7])
k_trans = kmeans.transform(df_seeds.iloc[:,0:7])
cluster_labels = kmeans.labels_

# plot clusters
plt.scatter(df_seeds.iloc[:,0].values, df_seeds.iloc[:,1].values)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], \
                        s=200, c='red', marker='x')
plt.savefig('kmeans_cluster_seeds.png', dpi=300)
plt.show()
