"""
Name: Kimberly Nestor
Class: CS 688 - Spring 2
Date: 03/30/22
Homework Problem: Hw 2B
Description of Problem: This program uses five visualisation techniques to
                        examine the distribution of data from the dataset below.
Dataset:
https://www.kaggle.com/datasets/laavanya/stress-level-detection?select=Stress-Lysis.csv
"""

import os
import sys
from os.path import join as opj
from collections import Counter

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.collections as clt
import ptitprince as pt

# set path, import data, (0- low stress, 1- normal stress, 2- high stress)
curr_dir = sys.path[0]
stress_data = pd.read_csv(opj(curr_dir, 'stress_data/Stress-Lysis.csv'))
# print(stress_data.iloc[:,-1]) # [:,:-1]

## plot 1 - pairplot, scatter and kernel density estimate (kde)
sns.pairplot(stress_data, hue='Stress Level', corner=True)
plt.savefig('pairplot_kde.png', dpi=300)
plt.show()


## plot 2 - pairplot, scatter and histogram
sns.pairplot(stress_data, hue='Stress Level', diag_kind="hist", corner=True)
plt.savefig('pairplot_hist.png', dpi=300)
plt.show()


## plot 3 - correlation matrix heatmap
stress_mat = round(stress_data.iloc[:,:-1].corr(), 2)
# make mask
mask = np.zeros_like(stress_mat)
mask[np.triu_indices_from(mask)] = True

# plot matrix, mask upper right triangle
with sns.axes_style("white"):
    f, ax = plt.subplots(figsize=(7, 5))
    ax = sns.heatmap(stress_mat, mask=mask, annot=True, square=True)
plt.title("Stress Data\n Correlation Heatmap")
plt.tight_layout()
plt.savefig('heatmap_corr_mat.png', dpi=300)
plt.show()


## plot 4 - rain cloud plot, step count var only
# source: https://tinyurl.com/ycksk7mn

# plot the clouds
fig, ax = plt.subplots(figsize=(7, 5))
dy = "Stress Level"; dx = "Step count"; ort = "h"; pal = sns.cubehelix_palette(n_colors=3)

ax = pt.half_violinplot(x=dx, y=dy, data=stress_data.iloc[:,2:], palette=pal, \
                        bw=.2, cut=0., scale="area", width=.6, inner=None, orient= ort)

# plot rain
ax=sns.stripplot( x = dx, y = dy, data = stress_data.iloc[:,2:], palette = pal, \
                  edgecolor = "white",size = 3, jitter = 1, zorder = 0, orient = ort)

# plot boxplots
ax=sns.boxplot( x = dx, y = dy, data = stress_data.iloc[:,2:], color = "white", width = .15, zorder = 10,\
                showcaps = True, boxprops = {'facecolor':'none', "zorder":10},\
                showfliers=True, whiskerprops = {'linewidth':2, "zorder":10},\
                saturation = 1, orient = ort)

plt.title("Stress Data\n Raincloud Plot showing Step Count")
plt.tight_layout()
plt.savefig('raincloud_step_count.png', dpi=300)
plt.show()


## plot 5 - donut pie chart
# source: https://tinyurl.com/4s47ftzy

# make dict of num class in dataset
dict_stress_n = dict(sorted( dict(Counter(stress_data.iloc[:,-1])) .items(), \
                             key=lambda i:i[0]))
stress_class = list(map(str,  dict_stress_n.keys()))
n_stress_class = list( dict_stress_n.values())

# plot pie chart
pal = sns.cubehelix_palette(n_colors=3)
fig, ax = plt.subplots()

ax.pie(n_stress_class, labels = ['Class '+i for i in stress_class], \
                        colors = pal, autopct='%.0f%%')

# make donut
circ = plt.Circle((0,0),0.70, fc='white')
fig = plt.gcf()
fig.gca().add_artist(circ)


plt.title("Stress Data\n Donut Chart showing Class Distribution")
plt.tight_layout()
plt.savefig('donut_stress_per.png', dpi=300)
plt.show()