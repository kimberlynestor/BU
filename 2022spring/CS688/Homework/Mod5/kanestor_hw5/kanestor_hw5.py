"""
Name: Kimberly Nestor
Class: CS 688 - Spring 2
Date: 04/20/22
Homework Problem: Hw 5
Description of Problem: This program does agglomerative hierarchical clustering
        on document titles. Select best k cluster using intrinsic evaluation
        using sillouette coefficient. Makes truncated dendrogram of clustered titles.
Dataset: crawl = 2022-03-25
https://webrobots.io/indiegogo-dataset/
"""

import sys
from os.path import join as opj
import zipfile
from joblib import dump, load

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from datetime import datetime
import itertools

import sklearn.metrics as sk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import ward, dendrogram
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# np.set_printoptions(threshold=sys.maxsize)

# load data from zip
curr_dir = sys.path[0]
zip = zipfile.ZipFile(opj(curr_dir, 'Indiegogo_2022-03-25T20_40_41_780Z.zip'), 'r')
zip_names = zip.namelist()

# make concat df, get rows from 2017 and 2021 (i couldn't get 15k using only months)
df_zip = pd.concat([pd.read_csv(zip.open(i)) for i in zip_names])
df_zip_yr = df_zip[df_zip['close_date'].str.startswith(tuple(['2017', '2021']), na=False)]

# list of doc titles only
titles = df_zip_yr['title'].values.astype('U')


#### Cluster each doc title
# define vectorizer parameters
tfidf_vectorizer = TfidfVectorizer(stop_words='english',
                                 use_idf=True)
tfidf_matrix = tfidf_vectorizer.fit_transform(titles)


# agglomerative hierarchical cluster, select best k
k=2
# model = AgglomerativeClustering(distance_threshold=None, n_clusters=k, \
#                             compute_distances=True).fit(tfidf_matrix.toarray())

# save model, load model
# dump(model, opj(curr_dir, f'agg_hier_k{k}.joblib'))
model = load(opj(curr_dir, f'agg_hier_k{k}.joblib'))

# model info
dist = model.distances_.reshape(-1, 1)
labels = model.labels_

# Silhouette Coefficient - intrinsic evaluation
sil_coef = sk.silhouette_score(tfidf_matrix.toarray(), labels, metric='euclidean')
print(f'\nSilhouette Coefficient, k={k}:  {round(sil_coef, 4)}\n')

# make linkage matrix, save and reload
# linkage_matrix = ward(tfidf_matrix.toarray())
# np.save(opj(curr_dir, f'linkage_matrix_k{k}.npy'), linkage_matrix)
linkage_matrix = np.load(opj(curr_dir, f'linkage_matrix_k{k}.npy'))


# plot dendrogram, truncated using level
fig, ax = plt.subplots(figsize=(15, 20))
ax = dendrogram(linkage_matrix, p=5, truncate_mode='level', orientation="right", labels=list(titles))
plt.tick_params(axis= 'x', which='both', bottom='off', top='off', labelbottom='off')
plt.xticks(fontsize=13)
plt.yticks(fontsize=15)
plt.title("Agglomerative Hierarchical Clustering Dendrogram", fontsize=14)
plt.tight_layout()
plt.savefig(f'dendro_agg_hier_titl_k{k}.png', dpi=300)
plt.show()









"""
#### IGNORE THIS
### this section is for each individual word
# get all words from titles, make unique list
bag = list(map(lambda x: x.split(), titles))
vocab = list(set(itertools.chain.from_iterable(bag)))

# pipeline for vector and tdidf
pipe = Pipeline([('count', CountVectorizer(vocabulary=vocab)), \
                  ('tfidf', TfidfTransformer())]).fit(titles)

# fit agglomerative hierachical clusters
model = AgglomerativeClustering(distance_threshold=0, n_clusters=None).fit(pipe['tfidf'].idf_.reshape(-1, 1))
tfidf = pipe['tfidf'].idf_.reshape(-1, 1)
vec = pipe['count'].transform(titles).toarray()

terms = tfidf_vectorizer.get_feature_names()
dist = 1 - cosine_similarity(tfidf_matrix)

linkage_matrix = ward(tfidf)

ax = dendrogram(linkage_matrix, p=50, truncate_mode='lastp', orientation="right", labels=vocab)
ax = dendrogram(linkage_matrix, p=5, truncate_mode='level', orientation="right", labels=vocab)
plt.savefig('dendro_agg_hier.png', dpi=300)
"""