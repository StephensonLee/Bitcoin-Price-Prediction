#!/usr/bin/env python
# coding: utf-8

# In[103]:


# Import necessary library needed for the model training
from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import plotly.offline as py
import plotly.graph_objs as go
import numpy as np
import seaborn as sns
# py.init_notebook_mode(connected=True)
# get_ipython().run_line_magic('matplotlib', 'inline')


# In[104]:


# Read the data set
data = pd.read_csv("D1.csv", index_col="Timestamp")


# In[105]:


# View data info
data.info()


# In[106]:


# View data rows
data.head()


# In[107]:


# Check latest date of data
data.tail()


# In[108]:


# Plot line graph base on Weighted Price
btc_trace=go.Scatter(x=data.index, y=data['Weighted_Price'], name='Price')
py.iplot([btc_trace])


# In[109]:


# Fill value 0 data points on Weighted Price with NaN and then use ffill method to fill values
data['Weighted_Price'].replace(0, np.nan, inplace=True)
data['Weighted_Price'].fillna(method='ffill', inplace=True)


# In[110]:


# Plot new line graph again on Weighted Price with newly filled values
btc_trace = go.Scatter(x=data.index, y=data['Weighted_Price'], name='Price')
py.plot([btc_trace])


# In[111]:


# Using Weighted Price as a feature to train the LSTM model
# Use MinMaxScaler to normalize Weighted Price to range from 0 to 1
values = data['Weighted_Price'].values.reshape(-1,1)
values = values.astype('float32')
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)


# In[112]:


# Split 70% of data for training and 30% for testing
train_size = int(len(scaled) * 0.7)
test_size = len(scaled) - train_size
train, test = scaled[0:train_size,:], scaled[train_size:len(scaled),:]
print(len(train), len(test))


# In[113]:


# Create function for creating daaset with look back
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    print(len(dataY))
    return np.array(dataX), np.array(dataY)


# In[114]:


# Generate dataset for trainX, trainY, testX, testY
look_back = 1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)


# In[142]:


# Reshape X for model training
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
print(trainX)


# In[118]:


# Running LSTM model with 300 epochs
model = Sequential()
model.add(LSTM(100, input_shape=(trainX.shape[1], trainX.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
history = model.fit(trainX, trainY, epochs=300, batch_size=100, validation_data=(testX, testY), verbose=0, shuffle=False)


# 

# In[119]:


# Plot line graph to show amount loss according to the epoch
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()


# In[120]:


# Make prediction using textX and plotting line graph against testY
yhat = model.predict(testX)
pyplot.plot(yhat, label='predict')
pyplot.plot(testY, label='true')
pyplot.legend()
pyplot.show()


# In[121]:


# Scaler Inverse Y back to normal value
yhat_inverse = scaler.inverse_transform(yhat.reshape(-1, 1))
testY_inverse = scaler.inverse_transform(testY.reshape(-1, 1))


# In[122]:


# RMSE
rmse = sqrt(mean_squared_error(testY_inverse, yhat_inverse))
print('Test RMSE: %.3f' % rmse)


# In[123]:


# Plot line graph with Y as USD
pyplot.plot(yhat_inverse, label='predict')
pyplot.plot(testY_inverse, label='actual', alpha=0.5)
pyplot.legend()
pyplot.show()


# In[124]:


# Convert X to dates
predictDates = data.tail(len(testX)).index


# In[125]:


# Reshape testY and yhat for plotly
testY_reshape = testY_inverse.reshape(len(testY_inverse))
yhat_reshape = yhat_inverse.reshape(len(yhat_inverse))


# In[126]:


# Plot predicted and actual line graph with X=dates, Y=USD
actual_chart = go.Scatter(x=predictDates, y=testY_reshape, name='Actual Price')
predict_chart = go.Scatter(x=predictDates, y=yhat_reshape, name='Predict Price')
py.plot([predict_chart, actual_chart])


# In[127]:


# Using additional features for model training
# Find correlation in features to Weighted Price
sns.heatmap(data.corr(), annot=True, cmap='RdYlGn', linewidths=0.1, vmin=0)
pyplot.show()

# In[128]:


# Function to convert series to supervised learning
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
            


# In[129]:


# Get all data values
values = data[['Weighted_Price'] + ['Volume_(BTC)'] + ['Volume_(Currency)']].values
values = values.astype('float32')
# Normalise features to range from 0 to 1
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)


# In[130]:


# Frame as supervised learning
reframed = series_to_supervised(scaled, 1, 1)
reframed.head()


# In[131]:


# Drop unnecessary columns
reframed.drop(reframed.columns[[4,5]], axis=1, inplace=True)
print(reframed.head())


# In[132]:


# Split data to 70% training, 30% testing
values = reframed.values
n_train_hours = int(len(values) * 0.7)
train = values[:n_train_hours, :]
test = values[n_train_hours:, :]
# split into input and outputs
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)


# In[133]:


# Training the LSTM with 300 epochs
multi_model = Sequential()
multi_model.add(LSTM(100, input_shape=(train_X.shape[1], train_X.shape[2])))
multi_model.add(Dense(1))
multi_model.compile(loss='mae', optimizer='adam')
multi_history = multi_model.fit(train_X, train_y, epochs=300, batch_size=100, validation_data=(test_X, test_y), verbose=0, shuffle=False)


# In[134]:


# Plot line graph to show amount loss according to the epoch
pyplot.plot(multi_history.history['loss'], label='multi_train')
pyplot.plot(multi_history.history['val_loss'], label='multi_test')
pyplot.legend()
pyplot.show()


# In[135]:


# Make prediction using testX and plotting the graph against testY
yhat = multi_model.predict(test_X)
pyplot.plot(yhat, label='predict')
pyplot.plot(test_y, label='true')
pyplot.legend()
pyplot.show()


# In[136]:


# Sclaer Inverse Y back to normal value
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
# invert scaling for forecast
print(len(yhat))
print(len(test_X[:, 1:]))
inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]
# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_y, test_X[:, 1:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]


# In[137]:


# RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)


# In[138]:


# Plot line graph with actual price, predicted price with feature weightedprice, redicted price with features Volume and weightedprice
actual_chart = go.Scatter(x=predictDates, y=inv_y, name= 'Actual Price')
multi_predict_chart = go.Scatter(x=predictDates, y=inv_yhat, name= 'Multi Predict Price')
predict_chart = go.Scatter(x=predictDates, y=yhat_reshape, name= 'Predict Price')
py.plot([predict_chart, multi_predict_chart, actual_chart])


# In[ ]:




