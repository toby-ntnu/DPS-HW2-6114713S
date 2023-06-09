import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df6 = pd.read_csv('population_density_106.csv')
sum=df6.groupby(['county']).sum()
#print(sum['people_total'])
x = np.arange(1, 23, 1)
plt.xticks(x)
plt.plot(x,sum['people_total'],marker='o',markersize=2,linewidth=0.8)
plt.title("106 plot people")
plt.ylabel("total people")
plt.xlabel("county")
plt.grid(True)
plt.legend()
plt.show()
