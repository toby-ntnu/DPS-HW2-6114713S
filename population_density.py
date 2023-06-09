import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df6 = pd.read_csv('population_density_106.csv')
df7 = pd.read_csv('population_density_107.csv')
df8 = pd.read_csv('population_density_108.csv')

sum=df6.groupby(['county']).sum()
sum1=df7.groupby(['county']).sum()
sum2=df8.groupby(['county']).sum()
#print(sum)
x = np.arange(1, 23, 1)
item = sum.people_total.sort_values()
item1 = sum1.people_total.sort_values()
item2 = sum2.people_total.sort_values()
plt.xticks(x)
print(item)
print(item1)
print(item2)

plt.plot(x,item,marker='o',markersize=2,linewidth=0.8)
plt.plot(x,item1,marker='o',markersize=2,linewidth=1)
plt.plot(x,item2,marker='x',markersize=2,linewidth=0.5)
plt.title("3 years plot sort people")
plt.ylabel("total people")
plt.xlabel("106-108 year county")
plt.grid(True)
plt.legend()
plt.show()
