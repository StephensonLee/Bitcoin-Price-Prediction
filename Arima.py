import numpy as np
import pandas as pd
from pandas import read_csv
from pandas import datetime
from matplotlib import pyplot
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt
import plotly.offline as py
import plotly.graph_objs as go

time_out=[]
RMSE_out=[]
recall0_out=[]
recall1_out=[]
recall01_out=[]
recall10_out=[]
hmean=[]
hmean_elbow=[]
# import the data
data = read_csv("D1.csv", index_col="Timestamp")
data['Weighted_Price'].replace(0, np.nan, inplace=True)
data['Weighted_Price'].fillna(method='ffill', inplace=True)
X = data[['Weighted_Price']].values
# ['Weighted_Price']+['Weighted_D1']+['Weighted_D2']+['Volume_(BTC)']+['Volume_(Currency)']+['Dif(HL)']+['Dif(OC)']

# train and predict using ARIMA
size = int(len(X) * 0.66)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()
start = datetime.now()
for t in range(len(test)):
	model = ARIMA(history, order=(8,2,0))
	model_fit = model.fit(disp=0)
	output = model_fit.forecast()
	yhat = output[0]
	predictions.append(yhat)
	obs = test[t]
	history.append(obs)
	print('predicted=%f, expected=%f' % (yhat, obs))
end = datetime.now()
print('time', (end - start).seconds)

rmse = sqrt(mean_squared_error(test, predictions))
print('Test RMSE: %.3f' % rmse)

# plot
pyplot.plot(test, color= 'blue')
pyplot.plot(predictions, color='red')
pyplot.show()

# label
label = []
for i, num in enumerate(test):
	if i == 0:
		label.append(0)
	elif num > test[i - 1]:
		label.append(1)
	else:
		label.append(0)

# predit the label
labelpdt = []
for i,num in enumerate(predictions):
	if i==0: labelpdt.append(0)
	elif num>predictions[i-1]: labelpdt.append(1)
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

results = pd.DataFrame()
results['Time']=time_out
results['RMSE']=RMSE_out
results['Sensitive']= recall1_out
results['Alarm']=recall0_out
results['Fall Elbow'] = recall10_out
results['Rise Elbow'] = recall01_out
results['Harmonic Mean'] = hmean
results['Elbow Harmonic Mean'] = hmean_elbow
results.to_csv('ARIMA_results.csv')

predictDates = data.tail(len(test)).index
test= [x for [x] in test]
predictions= [x for [x] in predictions]
actual_chart = go.Scatter(x=predictDates, y=test, name= 'Actual Price')
multi_predict_chart = go.Scatter(x=predictDates, y=predictions, name= 'ARIMA Predict Price')
py.plot([multi_predict_chart, actual_chart],filename='ARIMA_Predict.html')