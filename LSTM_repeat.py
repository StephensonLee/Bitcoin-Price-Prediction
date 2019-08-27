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
import pickle
py.init_notebook_mode(connected=True)

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

for look_back in range(1,2):
    for time_steps in range(1,2):
        for nerons in range(80,81,1):
            # Read the data set
            data = pd.read_csv("D2.csv", index_col="Timestamp")
            # look_back = 5
            # time_steps= 1 #step
            # Fill value 0 data points on Weighted Price with NaN and then use ffill method to fill values
            data['Weighted_Price'].replace(0, np.nan, inplace=True)
            data['Weighted_Price'].fillna(method='ffill', inplace=True)

            # Get all data values
            values = data[['Weighted_Price'] + ['Volume_(Currency)']+['Dif(OC)']+['Dif(HL)']+['Gap(CO)']+['Diff(OC)']+['Dif(HO)']+['Dif(LO)']+['Open']+['High']+['Low']+['Close']+['Volume_(BTC)']].values
            # values = data[['Weighted_Price'] + ['Volume_(Currency)']+['Dif(HL)']+['Dif(OC)']].values
            # values = data[['Weighted_Price']].values
            values = values.astype('float32')
            features=values.shape[1]
            # print(features)
            # Normalise features to range from 0 to 1
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled = scaler.fit_transform(values)
            # Frame as supervised learning
            reframed = series_to_supervised(scaled, look_back, 1)
            # Drop unnecessary columns
            for i in range(features-1):
                reframed.drop(reframed.columns[[len(reframed.columns) - 1]], axis=1, inplace=True)
            print(reframed.head())

            # Split data to 70% training, 30% testing
            values = reframed.values
            n_train_hours = int(len(values) * 0.7/time_steps)*time_steps
            train = values[:n_train_hours, :]
            test = values[n_train_hours:, :]

            #label
            label = []
            label.append(0)
            print(test[:,0])
            for i,num in enumerate(test[1:,0]):
                if num>test[i-1,0]: label.append(1)
                else: label.append(0)
            # print(label)

            # split into input and outputs
            train_X, train_y = train[:, :-1], train[:, -1]
            test_X, test_y = test[:, :-1], test[:, -1]
            # reshape input to be 3D [samples, timesteps, features]
            train_X = train_X.reshape((train_X.shape[0], time_steps, -1))
            # print(train_X)
            test_X = test_X.reshape((test_X.shape[0], time_steps, -1))
            print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

            # Training the LSTM with 300 epochs
            multi_model = Sequential()
            # multi_model.add(LSTM(nerons, input_shape=(time_steps, train_X.shape[2])))
            multi_model.add(LSTM(nerons, input_shape=(train_X.shape[1], train_X.shape[2]),return_sequences=False))
            # multi_model.add(LSTM(nerons,return_sequences=False))
            multi_model.add(Dense(1))
            multi_model.compile(loss='mae', optimizer='adam')
            start=datetime.now()
            multi_history = multi_model.fit(train_X, train_y, epochs=300, batch_size=100, validation_data=(test_X, test_y), verbose=0, shuffle=False)
            end=datetime.now()
            print('time',(end-start).seconds)
            # multi_model = pickle.load(open('multi_model.sav', 'rb'))
            # Make prediction using testX and plotting the graph against testY
            yhat = multi_model.predict(test_X)

            # Sclaer Inverse Y back to normal value
            test_X = test_X.reshape((test_X.shape[0], -1))
            test_X=test_X[:,:features]
            # print(test_X.shape)
            # invert scaling for forecast
            inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)
            # print(inv_yhat.shape)
            inv_yhat = scaler.inverse_transform(inv_yhat)
            inv_yhat = inv_yhat[:,0]
            # invert scaling for actual
            test_y = test_y.reshape((len(test_y), 1))
            inv_y = concatenate((test_y, test_X[:, 1:]), axis=1)
            inv_y = scaler.inverse_transform(inv_y)
            inv_y = inv_y[:,0]

            # RMSE
            rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
            print('look_back',look_back,'time_step',time_steps,'neurons',nerons)
            print('Test RMSE: %.1f' % rmse)

            labelpdt = []
            print(inv_yhat)
            for i,num in enumerate(inv_yhat):
                if i==0: labelpdt.append(0)
                elif num>inv_yhat[i-1]: labelpdt.append(1)
                else: labelpdt.append(0)
            print(labelpdt)

            num0=0
            num1=0
            for i in label:
                if i==0: num0+=1
                else: num1+=1
            print(num0,num1)

            recall0=0
            recall1=1
            for i,lab in enumerate(label):
                if lab==0 and labelpdt[i]==0: recall0+=1
                if lab==1 and  labelpdt[i]==1: recall1+=1
            print('recall of decrease',recall0/num0)
            print('recall of increase',recall1/num1)