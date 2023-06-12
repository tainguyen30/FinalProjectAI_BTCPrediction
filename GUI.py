# First we will import the necessary Library
import pandas as pd
import numpy as np
import math

# For Evalution we will use these library
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score
from sklearn.metrics import mean_poisson_deviance, mean_gamma_deviance, accuracy_score

from keras.models import load_model

# For PLotting we will use these library
import matplotlib.pyplot as plt
from itertools import cycle
import plotly.express as px
import streamlit as st
from sklearn.preprocessing import MinMaxScaler

# Load our dataset
# Note it should be in same dir
df = pd.read_csv(r'C:\Users\nguye\PycharmProjects\BTC Prediction Price_Final\BTC-USD 2014-2023.csv')
df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
print('Total number of days present in the dataset: ', df.shape[0])
print('Total number of fields present in the dataset: ', df.shape[1])

# Visualizations
st.subheader("Closing price vs time chart")
fig = plt.figure(figsize=(12, 6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader("Closing price vs time chart with MA100")
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)

st.subheader("Closing price vs time chart with MA100 & MA200")
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(ma100)
plt.plot(ma200)
plt.plot(df.Close)
st.pyplot(fig)

# ____________________________________________________________________________________________________________

# Building LSTM Model
# Steps:
# 1st step: Preparing Data for training and testing
# 1year Data (2022)

# First Take all the Close Price
closedf = df[['Date', 'Close']]
print("Shape of close dataframe:", closedf.shape)

# Take data of 1 year
closedf = closedf[closedf['Date'] >= '5/16/2022']
close_BTC = closedf.copy()
print("Total data for prediction: ", closedf.shape[0])

# Normalizing Data (Chuẩn hoá dữ liệu)
# Goal: change the values of numeric columns in the dataset to use a common
# scale, without distorting differences in the ranges of values or losing information MinMaxScaler: For each value in
# a feature, (MinMaxScaler - min)/range range: the difference between the original maximum and original minimum
# Preserve: the shape of the original distribution

# Deleting date column and normalizing using MinMax Scaler
del closedf['Date']
scaler = MinMaxScaler(feature_range=(0, 1))
closedf = scaler.fit_transform(np.array(closedf).reshape(-1, 1))
print(closedf.shape)

# We keep the training set as 60% and 40% testing set
training_size = int(len(closedf) * 0.60)
test_size = len(closedf) - training_size
train_data, test_data = closedf[0:training_size, :], closedf[training_size:len(closedf), :1]
print("train_data: ", train_data.shape)
print("test_data: ", test_data.shape)


# Transform the close price based on Time-series-analysis forecasting requirement
# Take 15
# Convert an array of values into a dataset matrix
def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), 0]  ###i=0, 0,1,2,3-----99   100
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)


time_step = 15
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

print("X_train: ", X_train.shape)
print("y_train: ", y_train.shape)
print("X_test: ", X_test.shape)
print("y_test", y_test.shape)

# Reshape input to be [samples, time steps, features] which is required for LSTM
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
print("X_train: ", X_train.shape)
print("X_test: ", X_test.shape)

# Load model
model = load_model(r"C:\Users\nguye\PycharmProjects\BTC Prediction Price_Final\BTC Prediction Price.h5")

# Make the prediction and check performance metrics
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)
print(train_predict.shape, test_predict.shape)

# Model Evaluation
# Transform back to original form
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)
original_ytrain = scaler.inverse_transform(y_train.reshape(-1, 1))
original_ytest = scaler.inverse_transform(y_test.reshape(-1, 1))

# Evaluation metrices RMSE, MSE and MAE
print("Train data RMSE: ", math.sqrt(mean_squared_error(original_ytrain, train_predict)))
print("Train data MSE: ", mean_squared_error(original_ytrain, train_predict))
print("Train data MAE: ", mean_absolute_error(original_ytrain, train_predict))
print("-------------------------------------------------------------------------------------")
print("Test data RMSE: ", math.sqrt(mean_squared_error(original_ytest, test_predict)))
print("Test data MSE: ", mean_squared_error(original_ytest, test_predict))
print("Test data MAE: ", mean_absolute_error(original_ytest, test_predict))

# Variance Regression Score
print("Train data explained variance regression score:",
      explained_variance_score(original_ytrain, train_predict))
print("Test data explained variance regression score:",
      explained_variance_score(original_ytest, test_predict))

# R square score for regression
print("Train data R2 score:", r2_score(original_ytrain, train_predict))
print("Test data R2 score:", r2_score(original_ytest, test_predict))

# Regression Loss Mean Gamma deviance regression loss (MGD) and Mean Poisson deviance regression loss (MPD)
print("Train data MGD: ", mean_gamma_deviance(original_ytrain, train_predict))
print("Test data MGD: ", mean_gamma_deviance(original_ytest, test_predict))
print("----------------------------------------------------------------------")
print("Train data MPD: ", mean_poisson_deviance(original_ytrain, train_predict))
print("Test data MPD: ", mean_poisson_deviance(original_ytest, test_predict))

# __________________________________________________________________________________________________________

# Comparision of original Bitcoin close price and predicted close price
# shift train predictions for plotting
look_back = time_step
trainPredictPlot = np.empty_like(closedf)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict) + look_back, :] = train_predict
print("Train predicted data: ", trainPredictPlot.shape)

# Shift test predictions for plotting
testPredictPlot = np.empty_like(closedf)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(train_predict) + (look_back * 2) + 1:len(closedf) - 1, :] = test_predict
print("Test predicted data: ", testPredictPlot.shape)

names = cycle(['Original close price', 'Train predicted close price', 'Test predicted close price'])

plotdf = pd.DataFrame({'date': close_BTC['Date'],
                       'original_close': close_BTC['Close'],
                       'train_predicted_close': trainPredictPlot.reshape(1, -1)[0].tolist(),
                       'test_predicted_close': testPredictPlot.reshape(1, -1)[0].tolist()})

fig = px.line(plotdf, x=plotdf['date'], y=[plotdf['original_close'], plotdf['train_predicted_close'],
                                           plotdf['test_predicted_close']],
              labels={'value': 'BTC price', 'date': 'Date'})
fig.update_layout(title_text='Comparision between original close price vs predicted close price',
                  plot_bgcolor='white', font_size=15, font_color='black', legend_title_text='Close Price')
fig.for_each_trace(lambda t: t.update(name=next(names)))

fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=False)
fig.show()

# Predicting next 30 days
x_input = test_data[len(test_data) - time_step:].reshape(1, -1)
temp_input = list(x_input)
temp_input = temp_input[0].tolist()

lst_output = []
n_steps = time_step
i = 0
pred_days = 30
while i < pred_days:
    if len(temp_input) > time_step:
        x_input = np.array(temp_input[1:])
        # print("{} day input {}".format(i,x_input))
        x_input = x_input.reshape(1, -1)
        x_input = x_input.reshape((1, n_steps, 1))

        yhat = model.predict(x_input, verbose=0)
        # print("{} day output {}".format(i,yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input = temp_input[1:]
        # print(temp_input)

        lst_output.extend(yhat.tolist())
        i = i + 1
    else:
        x_input = x_input.reshape((1, n_steps, 1))
        yhat = model.predict(x_input, verbose=0)
        temp_input.extend(yhat[0].tolist())

        lst_output.extend(yhat.tolist())
        i = i + 1
print("Output of predicted next days: ", len(lst_output))

# Plotting last 15 days of dataset and next predicted 30 days
last_days = np.arange(1, time_step + 1)
day_pred = np.arange(time_step + 1, time_step + pred_days + 1)
print(last_days)
print(day_pred)

predicted = np.empty((len(last_days) + pred_days + 1, 1))
predicted[:] = np.nan
predicted = predicted.reshape(1, -1).tolist()[0]

last_original_days_value = predicted
next_predicted_days_value = predicted

last_original_days_value[1:time_step + 1] = \
    scaler.inverse_transform(closedf[len(closedf) - time_step - 1:]).reshape(1, -1).tolist()[0]
next_predicted_days_value[time_step + 1:] = \
    scaler.inverse_transform(np.array(lst_output).reshape(-1, 1)).reshape(1, -1).tolist()[0]

# print(last_original_days_value)
# print(next_predicted_days_value)

new_pred_plot = pd.DataFrame({
    'last_original_days_value': last_original_days_value,
    'next_predicted_days_value': next_predicted_days_value
})

names = cycle(['Last 15 days close price', 'Predicted next 30 days close price'])

fig = px.line(new_pred_plot, x=new_pred_plot.index, y=[new_pred_plot['last_original_days_value'],
                                                       new_pred_plot['next_predicted_days_value']],
              labels={'value': 'BTC price', 'index': 'Timestamp'})
fig.update_layout(title_text='Compare last 15 days vs next 30 days',
                  plot_bgcolor='white', font_size=15, font_color='black', legend_title_text='Close Price')

fig.for_each_trace(lambda t: t.update(name=next(names)))
fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=False)
fig.show()

# Plotting entire closing BTC price with next 30 days period of prediction
lstmdf = closedf.tolist()
lstmdf.extend((np.array(lst_output).reshape(-1, 1)).tolist())
lstmdf = scaler.inverse_transform(lstmdf).reshape(1, -1).tolist()[0]

names = cycle(['Close price'])

fig = px.line(lstmdf, labels={'value': 'BTC price', 'index': 'Timestamp'})
fig.update_layout(title_text='Plotting 2023 whole closing BTC price in 30 days with prediction',
                  plot_bgcolor='white', font_size=15, font_color='black', legend_title_text='BTC')

fig.for_each_trace(lambda t: t.update(name=next(names)))

fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=False)
fig.show()

# Final Graph
st.subheader("Predictions vs Original",)
fig2 = plt.figure(figsize=(12, 6))
plt.plot(original_ytest, "b", label="Original price")
plt.plot(test_predict, "r", label="Predicted price")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
st.pyplot(fig2)
plt.show()
