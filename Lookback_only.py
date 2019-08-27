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
from keras.layers import Activation
from keras.layers import LSTM
from keras.layers import Dropout
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
time_out=[]
RMSE_out=[]
recall0_out=[]
recall1_out=[]
recall01_out=[]
recall10_out=[]
hmean=[]
hmean_elbow=[]
loback=[]
neuron=[]

layers = 2
for look_back in range(2,3):
    for time_steps in range(1,2):
        for nerons in range(60,61,1):
            for iteration in range(1):
                # Read the data set
                data = pd.read_csv("D2.csv", index_col="Timestamp")

                # Fill value 0 data points on Weighted Price with NaN and then use ffill method to fill values
                data['Weighted_Price'].replace(0, np.nan, inplace=True)
                data['Weighted_Price'].fillna(method='ffill', inplace=True)

                # pyplot.figure(figsize=(16, 9))
                # sns.heatmap(data.corr().round(2), annot=True, cmap='RdYlGn', linewidths=0.5, vmin=0)
                # pyplot.show()

                # Get all data values
                values = data[['Weighted_Price']+['Volume_(Currency)']+ ['Dif(HL)']+['Dif(HO)'] + ['Dif(LO)']].values
                # values = data[['Weighted_Price']+['Volume_(Currency)']+ ['Dif(HL)'] + ['Dif(HO)'] + ['Dif(LO)']].values
                values1 = data[['Weighted_D1']+['Weighted_D2']+['Dif(OC)']].values

                # values = data[['Weighted_Price'] + ['Volume_(Currency)']+['Dif(OC)']+['Gap(CO)']+['Diff(OC)']+['Dif(HO)']+['Dif(LO)']+['Open']+['High']+['Low']+['Close']+['Volume_(BTC)']].values
                values = values.astype('float32')
                values1 = values1.astype('float32')
                features=values.shape[1]

                # Normalise features to range from 0 to 1
                scaler = MinMaxScaler(feature_range=(0, 1))
                scaled = scaler.fit_transform(values)
                scaler1 = MinMaxScaler(feature_range=(0, 1))
                scaled1 = scaler1.fit_transform(values1)

                print(scaled.shape)
                print(scaled1.shape)
                # Frame as supervised learning
                reframed = series_to_supervised(scaled, look_back, 1)

                # Drop unnecessary columns
                for i in range(features-1):
                    reframed.drop(reframed.columns[[len(reframed.columns) - 1]], axis=1, inplace=True)
                print(reframed.head())

                # Split data to 66% training, 34% testing
                values = reframed.values
                values1 = scaled1[:-look_back]
                values = concatenate((values1, values), axis=1)
                print(pd.DataFrame(values).head())

                # corre = pd.DataFrame(values)
                # pyplot.figure(figsize=(16, 9))
                # sns.heatmap(corre.corr().round(2), annot=True, cmap='RdYlGn', linewidths=0.5, vmin=0)
                # pyplot.show()

                n_train_hours = int(len(values) * 0.66/time_steps)*time_steps
                train = values[:n_train_hours, :]
                test = values[n_train_hours:, :]

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
                if layers>1:
                    multi_model.add(LSTM(nerons, input_shape=(train_X.shape[1], train_X.shape[2]),return_sequences=True))
                    multi_model.add(Dropout(0.2))
                else:
                    multi_model.add(LSTM(nerons, input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences=False))
                for i in range(layers-1):
                    multi_model.add(LSTM(int(nerons),input_shape=(train_X.shape[1], train_X.shape[2]),return_sequences=False))
                multi_model.add(Dropout(0.2))
                multi_model.add(Dense(1))
                # multi_model.add(Activation('tanh'))
                multi_model.compile(loss='mae', optimizer='adam')
                start=datetime.now()
                multi_history = multi_model.fit(train_X, train_y, epochs=50, batch_size=100, validation_data=(test_X, test_y), verbose=0, shuffle=False)
                end=datetime.now()
                print('time',(end-start).seconds)

                pyplot.plot(multi_history.history['loss'], label='multi_train')
                pyplot.plot(multi_history.history['val_loss'], label='multi_test')
                pyplot.legend()
                pyplot.show()
                pickle.dump(multi_model, open('test.sav', 'wb'))

                multi_model = pickle.load(open('test.sav', 'rb'))
                # Make prediction using testX and plotting the graph against testY
                yhat = multi_model.predict(test_X)


                # Scaler Inverse Y back to normal value
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

                #label
                label = []
                for i,num in enumerate(inv_y):
                    if i==0: label.append(0)
                    elif num>inv_y[i-1]: label.append(1)
                    else: label.append(0)

                # predit the label
                labelpdt = []
                for i,num in enumerate(inv_yhat):
                    if i==0: labelpdt.append(0)
                    elif num>inv_yhat[i-1]: labelpdt.append(1)
                    else: labelpdt.append(0)

                # calculate the recall
                Fall=0
                Raise=0
                RElbow=0
                FElbow = 0
                for i in range(1,len(label)):
                    if label[i]==0:
                        Fall+=1
                        if label[i-1]==1: FElbow+=1
                    else:
                        Raise+=1
                        if label[i-1]==0: RElbow+=1
                print(Fall,Raise,FElbow,RElbow)
                Fall_cor=0
                Raise_cor=0
                FElbow_cor=0
                RElbow_cor=0
                for i,lab in enumerate(label):
                    if i!=0 and lab==0 and labelpdt[i]==0:
                        Fall_cor+=1
                        if label[i-1]==1: FElbow_cor+=1
                    if i!=0 and lab==1 and labelpdt[i]==1:
                        Raise_cor+=1
                        if label[i - 1] == 0: RElbow_cor += 1
                FElbow_recall= FElbow_cor/FElbow
                RElbow_recall = RElbow_cor / RElbow
                Fall_recall= Fall_cor/Fall
                Raise_recall=Raise_cor/Raise
                Hmean=2*Fall_recall*Raise_recall/(Fall_recall+Raise_recall)
                Hmean_Elbow=2*FElbow_recall*RElbow_recall/(FElbow_recall+RElbow_recall)
                print('recall of decrease',Fall_recall)
                print('recall of increase',Raise_recall)
                print('recall of fall elbow', FElbow_recall)
                print('recall of rise elbow', RElbow_recall)
                print('harmony recall', Hmean)
                print('elbow harmony recall', Hmean_Elbow)

                time_out.append((end-start).seconds)
                RMSE_out.append(rmse)
                recall0_out.append(Fall_recall)
                recall1_out.append(Raise_recall)
                recall10_out.append(FElbow_recall)
                recall01_out.append(RElbow_recall)
                hmean.append(Hmean)
                hmean_elbow.append(Hmean_Elbow)
                loback.append(look_back)
                neuron.append(nerons)

results = pd.DataFrame()
results['Neurons']=neuron
results['Look back']=loback
results['Time']=time_out
results['RMSE']=RMSE_out
results['Sensitive']= recall1_out
results['Alarm']=recall0_out
results['Fall Elbow'] = recall10_out
results['Rise Elbow'] = recall01_out
results['Harmonic Mean'] = hmean
results['Elbow Harmonic Mean'] = hmean_elbow
results.to_csv('LSTM_results.csv')


# Plot line graph with actual price, predicted price with feature weightedprice, redicted price with features Volume and weightedprice
predictDates = data.tail(len(test_X)).index
actual_chart = go.Scatter(x=predictDates, y=inv_y, name= 'Actual Price')
multi_predict_chart = go.Scatter(x=predictDates, y=inv_yhat, name= 'Multi Predict Price')
py.plot([multi_predict_chart, actual_chart],filename='muti_predict.html')