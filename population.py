import pandas as pd
import numpy as np
import matplotlib as mb
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
df6 = pd.read_csv('population_density_106.csv')
df7 = pd.read_csv('population_density_107.csv')
df8 = pd.read_csv('population_density_108.csv')
x1 = np.arange(1, 67, 1)
df106 = df6.groupby("county")["people_total"].sum()
df107 = df7.groupby("county")["people_total"].sum()
df108 = df8.groupby("county")["people_total"].sum()

x = x1[:,np.newaxis]
y1 = df106[:,np.newaxis]
y2 = df107[:,np.newaxis]
y3 = df108[:,np.newaxis]
#print(x,"y=",y)
y = np.vstack((y1,y2,y3))
A = np.hstack((x,y))
#B = np.hstack()
