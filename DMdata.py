import pandas as pd
import numpy as np

data = pd.read_csv("D1.csv", index_col="Timestamp")
data['Weighted_Price'].replace(0, np.nan, inplace=True)
data['Weighted_Price'].fillna(method='ffill', inplace=True)

# print(data)
weighted=data['Weighted_Price'].values
open = data['Open'].values
close = data['Close'].values
high = data['High'].values
low = data['Low'].values
volume = data['Volume_(BTC)'].values
dif1 = close-open
dif2 = high-low
gap =open- np.roll(close,1)
gap[0]=0
diff = dif1-np.roll(dif1,1)
diff[0]=0

data['Weighted']=weighted
data['Dif(OC)']=dif1
data['Dif(HL)']=dif2
data['Gap']=gap
data['Diff']=diff
data.to_csv('D2.csv')
print(data)
