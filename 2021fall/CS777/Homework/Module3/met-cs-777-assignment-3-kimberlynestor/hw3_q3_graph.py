"""
Kimberly Nestor
CS777 Big Data Analytics
09/27/2021
Homework 3 Question3 graph
Description:
"""

import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
import numpy as np
import math


data_file = '/Users/kimberlynestor/Desktop/BU/2021fall/CS777/Homework/Module3/cost_day/part-00000'
outdir = '/Users/kimberlynestor/Desktop/BU/2021fall/CS777/Homework/Module3/met-cs-777-assignment-3-kimberlynestor/'

df_cost = pd.read_csv(data_file, header=None) #, names=['epoch', 'm_grad_vec', 'b_inter', 'cost'])


epoch = [float(str(i).replace('[', '')) for i in df_cost[0].tolist()]
cost = [float(str(i).replace(']', '')) for i in df_cost[6].tolist()]


#plot x= iteration, y= cost
plt.plot(epoch, cost, label="cost")
plt.title("Gradient descent cost, Multiple LR", fontweight='bold', size=10)
plt.xlabel("Iteration", fontweight='bold', size=9)
plt.ylabel("Cost (mil)", fontweight='bold', size=8.5)
plt.legend()
plt.savefig(os.path.join(outdir, 'cost_graph_q3.png'), dpi=200)
plt.show()