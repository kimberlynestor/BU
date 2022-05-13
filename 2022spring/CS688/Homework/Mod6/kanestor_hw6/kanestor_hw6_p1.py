"""
Name: Kimberly Nestor
Class: CS 688 - Spring 2
Date: 04/25/22
Homework Problem: Hw 6
Description of Problem: This program
"""

import sys
from os.path import join as opj

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import networkx as nx
from networkx import random_layout


# load and clean twitter data
curr_dir = sys.path[0]
twit_data = list(map(lambda x:x.split(), map(lambda x:x.replace('\n', '').strip(), \
                            open(opj(curr_dir, 'twit_foll.csv'), "r").readlines())))
df_twit_data = pd.DataFrame(twit_data[1:], columns=twit_data[0])

targets = list(set(df_twit_data.iloc[:,0].values))
print("Target twitter accounts:\n", targets, "\n")
print(twit_data)
sys.exit()
# get adj matrix from twitter friends
# undirected graph
G = nx.Graph()
G.add_edges_from(twit_data[1:])
nodes = G.nodes()
nx_adj = nx.adjacency_matrix(G) #.todense()
np_adj = nx.to_numpy_matrix(G)

# directed graph
G_dir = nx.DiGraph()
G_dir.add_edges_from(twit_data[1:])

#### PART C - plot undirected and directed graphs
# set dark colour for four target nodes, and center
color_map = ['rosybrown' if i in targets else 'mistyrose' for i in nodes]
pos = nx.spring_layout(G, seed=0)

# plot undirected graph
nx.draw_networkx(G, font_size=8, edge_color='#767676', width=0.8, \
                        node_color=color_map, pos=pos)
plt.title("Undirected graph of Twitter targets and accounts following", fontsize=8)
plt.tight_layout()
plt.savefig('twit_foll_und_graph.png', dpi=300)
plt.show()

# plot directed graph
nx.draw_networkx(G, font_size=8, edge_color='#767676', width=0.8, \
                        node_color=color_map, arrows=True, pos=pos)
plt.title("Directed graph of Twitter targets and accounts following", fontsize=8)
plt.tight_layout()
plt.savefig('twit_foll_dir_graph.png', dpi=300)
plt.show()


#### PART D - implement dijkstra on two nodes, > 1 hop
dij_path = nx.dijkstra_path(G,'WhiteHouse','SecDef')
print("Dijkstra shortest path:\n", dij_path, "\n")


#### Part E - centrality measures on graphs
cent_meas = ['Degree', 'Closeness', 'Betweenness', 'Eigenvector']

## undirected graph
dg_und = nx.degree_centrality(G)
cg_und = nx.closeness_centrality(G)
bg_und = nx.betweenness_centrality(G)
eg_und = nx.eigenvector_centrality(G)

# make list of centrality dicts
cent_dicts = [dg_und, cg_und, bg_und, eg_und]

# print measure name and output for centrality of each node
for meas, dicts in zip(cent_meas, cent_dicts):
  print(f'{meas} centrality, undirected:')
  for key, val in dicts.items():
    print(f'   {key}: {round(val,2)}')
  print('\n')

## directed graph
dg_dir = nx.degree_centrality(G_dir)
cg_dir = nx.closeness_centrality(G_dir)
bg_dir = nx.betweenness_centrality(G_dir)
eg_dir = nx.eigenvector_centrality(G_dir)

# make list of centrality dicts
cent_dicts = [dg_dir, cg_dir, bg_dir, eg_dir]

# print measure name and output for centrality of each node
for meas, dicts in zip(cent_meas, cent_dicts):
  print(f'{meas} centrality, directed:')
  for key, val in dicts.items():
    print(f'   {key}: {round(val,2)}')
  print('\n')
