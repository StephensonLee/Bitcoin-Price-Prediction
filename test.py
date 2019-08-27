import  pandas as pd
import seaborn as sns
from matplotlib import pyplot
from numpy import concatenate

data = pd.read_csv("D2.csv", index_col="Timestamp")
values1 = data[['Weighted_Price']+['Volume_(Currency)']].values
values2 = data[['Volume_(Currency)']].values
print(values1)
# print(values2)
print(values1[:,1]-values2[:,0])
