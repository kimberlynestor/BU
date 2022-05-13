"""
Name: Kimberly Nestor
Class: CS 688 - Spring 2
Date: 04/29/22
Homework Problem: Hw 6
Description of Problem: This program gets bigram count of text data and makes
                        weighted undirected graph.
Dataset:
https://www.kaggle.com/datasets/balabaskar/harry-potter-books-corpora-part-1-7
"""

import os
import sys
from os import listdir
from os.path import join as opj

import itertools

import pandas as pd
import numpy as np
import math

# import igraph as ig
import networkx as nx
import matplotlib.pyplot as plt

import nltk # nltk.word_tokenize
import sklearn.metrics as sk
from sklearn.feature_extraction.text import CountVectorizer


# load hp data
curr_dir = sys.path[0]
hp_path_lst = os.listdir(opj(curr_dir, 'hp_corpora'))

hp_bk_all = list(itertools.chain.from_iterable([open(opj(curr_dir, \
                f'hp_corpora/{i}'), 'r').read().split() for i in hp_path_lst]))

hp_bk1 = open(opj(curr_dir, 'hp_corpora/Book1.txt'), 'r').read().split()

# preprocess text data, subsample data
vectorizer = CountVectorizer(stop_words='english') #ngram_range=(2, 2)
hp_vec = vectorizer.fit_transform(hp_bk1[0:math.floor(len(hp_bk1)/400)])
hp_token = [i for i in vectorizer.get_feature_names() if any(map(str.isdigit, i)) == False]

# make bigram
hp_bigram = list(nltk.bigrams(hp_token))
count_dict = dict(nltk.FreqDist(hp_bigram))
count_bigram = nltk.FreqDist(hp_bigram).items()

# make df of bigrams
df_bigram = pd.DataFrame(count_bigram, columns=['bigram', 'count'])
df_bigram.insert(0, 'bg1', df_bigram['bigram'].str[0])
df_bigram.insert(1, 'bg2', df_bigram['bigram'].str[1])
df_bg = df_bigram[['bg1', 'bg2', 'count']]

# save output to csv
df_bg.to_csv('bigram_count.csv', index=False)

# undirected graph
G = nx.Graph()
G.add_edges_from(df_bigram['bigram'].tolist())
# nx.set_edge_attributes(G, values=count_dict, name='weight')

nodes = G.nodes()
nx_adj = nx.adjacency_matrix(G) #.todense()
np_adj = nx.to_numpy_matrix(G)

pos = nx.spring_layout(G, seed=0, k=8)


# plot undirected graph
nx.draw_networkx(G, font_size=6, edge_color='#767676', width=0.8, pos=pos, \
                    node_size=50, node_color='cadetblue', alpha=0.5)
nx.draw_networkx_edge_labels(G, pos, edge_labels=count_dict, font_size=4, \
                             font_color='darkslategrey')
plt.title("Undirected weighted graph of bigram count", fontsize=8)
plt.tight_layout()
plt.savefig('bigram_count_graph.png', dpi=300)
plt.show()