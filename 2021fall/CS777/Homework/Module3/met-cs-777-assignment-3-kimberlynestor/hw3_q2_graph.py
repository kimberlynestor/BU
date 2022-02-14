"""
Kimberly Nestor
CS777 Big Data Analytics
09/27/2021
Homework 3 Question2 graph
Description:
"""

import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
import numpy as np
import math


data_file = '/Users/kimberlynestor/Desktop/BU/2021fall/CS777/Homework/Module3/cost_smalldata/part-00000'
outdir = '/Users/kimberlynestor/Desktop/BU/2021fall/CS777/Homework/Module3/met-cs-777-assignment-3-kimberlynestor/'

df_cost = pd.read_csv(data_file, header=None, names=['epoch', 'm_grad', 'cost'])
epoch = [float(i.replace('[', '')) for i in df_cost['epoch'].tolist()]
cost = [float(i.replace(']', '')) for i in df_cost['cost'].tolist()]
#
# print(math.ceil(max(cost)))
#
# sys.exit()

#plot x= iteration, y= cost
plt.plot(epoch, cost, label="cost")
plt.title("Gradient descent cost", fontweight='bold', size=10)
plt.xlabel("Iteration", fontweight='bold', size=9)
plt.ylabel("Cost (mil)", fontweight='bold', size=8.5)
plt.legend()
plt.savefig(os.path.join(outdir, 'cost_graph.png'), dpi=200)
plt.show()