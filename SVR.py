import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from math import sqrt
from numpy import concatenate
import plotly.offline as py
import plotly.graph_objs as go
from datetime import datetime
# #############################################################################
# Create function for creating daaset with look back
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

# Generate sample data
data = pd.read_csv("D2.csv", index_col="Timestamp")
values = data[['Weighted_Price']+['Volume_(Currency)']+ ['Dif(HL)']+['Dif(HO)'] + ['Dif(LO)']+['Weighted_D1']+['Weighted_D2']+['Dif(OC)']].values
values = values.astype('float32')
features=values.shape[1]

# Normalise features to range from 0 to 1
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
reframed = series_to_supervised(scaled, 1, 1)
# reframed.drop(reframed.columns[[4,5]], axis=1, inplace=True)

for i in range(features - 1):
    reframed.drop(reframed.columns[[len(reframed.columns) - 1]], axis=1, inplace=True)
print(reframed.head())

# Split data to 70% training, 30% testing
values = reframed.values
n_train_hours = int(len(values) * 0.66)
train = values[:n_train_hours, :]
test = values[n_train_hours:, :]
# split into input and outputs
trainX, trainY = train[:, :-1], train[:, -1]
testX, testY = test[:, :-1], test[:, -1]
print(trainX.shape, trainY.shape, testX.shape, testY.shape)

RMSE_out=[]
recall0_out=[]
recall1_out=[]
recall01_out=[]
recall10_out=[]
hmean=[]
hmean_elbow=[]
parameter_C=[]
parameter_gamma=[]
parameter_epsilon=[]
for C in range(80,81,1):
    for gamma in range(20,21,1):
        for epsilon in range (5,6,1):
            #SVR
            start= datetime.now()
            svr = SVR(kernel='rbf', C=C, gamma=gamma/10000, epsilon=epsilon/10000)
            svr.fit(trainX,trainY)
            predictY=svr.predict(testX)
            end= datetime.now()
            print('time', (end - start).seconds)

            # invert scaling for forecast
            predictY = predictY.reshape((len(predictY), 1))
            inv_yhat = concatenate((predictY, testX[:, 1:]), axis=1)
            inv_yhat = scaler.inverse_transform(inv_yhat)
            inv_yhat = inv_yhat[:,0]

            # invert scaling for actual
            test_y = testY.reshape((len(testY), 1))
            inv_y = concatenate((test_y, testX[:, 1:]), axis=1)
            inv_y = scaler.inverse_transform(inv_y)
            inv_y = inv_y[:,0]

            # RMSE
            rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
            print('Test RMSE: %.3f' % rmse)

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
            parameter_C.append(C)
            parameter_gamma.append(gamma)
            parameter_epsilon.append(epsilon)
            RMSE_out.append(rmse)
            recall0_out.append(Fall_recall)
            recall1_out.append(Raise_recall)
            recall10_out.append(FElbow_recall)
            recall01_out.append(RElbow_recall)
            hmean.append(Hmean)
            hmean_elbow.append(Hmean_Elbow)

results = pd.DataFrame()
results['C']=parameter_C
results['Gamma']=parameter_gamma
results['Epsilon']=parameter_epsilon
results['RMSE']=RMSE_out
results['Sensitive']= recall1_out
results['Alarm']=recall0_out
results['Fall Elbow'] = recall10_out
results['Rise Elbow'] = recall01_out
results['Harmonic Mean'] = hmean
results['Elbow Harmonic Mean'] = hmean_elbow
results.to_csv('SVR_results.csv')


predictDates = data.tail(len(testX)).index
actual_chart = go.Scatter(x=predictDates, y=inv_y, name= 'Actual Price')
predict_chart = go.Scatter(x=predictDates, y=inv_yhat, name= 'Predict Price')
py.plot([predict_chart, actual_chart])