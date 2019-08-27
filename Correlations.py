import  pandas as pd
import seaborn as sns
from matplotlib import pyplot
from sklearn.preprocessing import MinMaxScaler

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0 , -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ..... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i==0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

look_back =2
data = pd.read_csv("D2.csv", index_col="Timestamp")
values = data[['Weighted_Price']+['Volume_(Currency)']+ ['Dif(HL)'] + ['Dif(HO)'] + ['Dif(LO)']].values
# values = data.values
features = values.shape[1]
values_back = series_to_supervised(values, look_back, 1)
values_back = values_back.values
# for i in range(features):
#     for j in range(look_back):
#         values_back[:,j*features+i]=values_back[:,j*features+i]-values_back[:,(look_back*features+i)]

scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values_back)

pyplot.figure(figsize=(16, 9))
sns.heatmap(pd.DataFrame(values_back).corr().round(2), annot=True, cmap='RdYlGn', linewidths=0.5, vmin=0)
pyplot.show()