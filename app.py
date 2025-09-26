#stock market price prediction using ML
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from keras.models import load_model
#using data from keras file(where our model is trained)
import streamlit as st
#hosting the app.

#Loading the model.....
model = load_model('C:\STOCK\Stock Predictions Model.keras')

#header of the page
st.header('Stock Market Predictor')

st.subheader('STOCK SYMBOLS:\n\n1.Nifty50->^NSEI 2.Tesla->TSLA 3.Facebook,Insta->META. 4.Bitcoin->BTC-USD.')
stock = st.text_input('Enter the stock symbol','GOOG') #asking user for stock symbol
start = '2012-01-01'#Setting date from where to where
end = '2022-12-31'

data = yf.download(stock,start,end) #Downloading the dataset from yahoo finance.

st.subheader('Stock Data')
st.write(data)#table of data is created
#80% train 20% test
data_train = pd.DataFrame(data.Close[0: int(len(data)*0.80)])
data_test = pd.DataFrame(data.Close[int(len(data)*0.80): len(data)])
#scaling(0,1)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))


#fitting the data
pas_100_days = data_train.tail(100)#for past 100 days
data_train = pd.concat([pas_100_days,data_test],ignore_index=True)
data_test_scaler = scaler.fit_transform(data_test)

#Creating graph price moving average 50
st.subheader('Price VS MA50')
ma_50_days = data.Close.rolling(50).mean()
fig1 = plt.figure(figsize=(8,6))
plt.plot(ma_50_days,'r',label="MA_50_days")
plt.plot(data.Close,'g',label="PRICE")
plt.show()
plt.legend()#for showing lablels
st.pyplot(fig1)#plotting graph on web

st.subheader('Price VS MA50 VS MA100')
ma_100_days = data.Close.rolling(100).mean()
fig2 = plt.figure(figsize=(8,6))
plt.plot(ma_50_days,'r',label="MA_50_days")
plt.plot(ma_100_days,'b',label="MA_100_days")
plt.plot(data.Close,'g',label="PRICE")
plt.legend()
plt.show()
st.pyplot(fig2)

st.subheader('Price VS MA100 VS MA200')
ma_200_days = data.Close.rolling(200).mean()
fig3 = plt.figure(figsize=(8,6))
plt.plot(ma_100_days,'r',label="MA_100_days")
plt.plot(ma_200_days,'b',label="MA_200_days")
plt.plot(data.Close,'g',label="PRICE")
plt.legend()
plt.show()
st.pyplot(fig3)

#Array slicing
x = []
y = []

for i in range(100, data_test_scaler.shape[0]):
    x.append(data_test_scaler[i-100:i])
    y.append(data_test_scaler[i,0])
    

x,y =np.array(x),np.array(y)

predict = model.predict(x)

scale = 1/scaler.scale_

predict = predict*scale
y = y*scale


#drawing charts
st.subheader('Original Price VS Predicted Price')
fig4 = plt.figure(figsize=(8,6))
plt.plot(predict,'r',label="PREDICTED PRICE")
plt.plot(y,'g',label='ORIGINAL PRICE')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
st.pyplot(fig4)


