#!/usr/bin/env python3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense

"""
Explanation:
-Concatinating the two datasets to get
a whole big training dataset
-Converting the timestamps to unix timestamps(
which is mainly the count of seconds since the unix
epoch(1 jan 1970 00:00)
-Spliting into input and labels
-Train test split based on 0.8 training and
0.2 of the dataset for validation
-Matching shapes as needed
-Building a GRU based model to avoid the vanishing gradiants
problem and to get faster training
-Using mean sqaured error loss
-Compiling the model
-Testing it on validation examples
)
"""


data_bitstamp = pd.read_csv("./bitstamp_processed.csv")
data_coinbase = pd.read_csv("./coinbase_processed.csv")
data = pd.concat([data_bitstamp, data_coinbase])
data['Timestamp'] = pd.to_datetime(data['Timestamp'])
data['Timestamp'] = (data['Timestamp'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
print(data.head())
X = data.drop(columns=['Close']).values
y = data['Close'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_reshaped = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test_reshaped = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
model = Sequential()
model.add(GRU(50, input_shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2])))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.fit(X_train_reshaped, y_train, epochs=5, batch_size=32, verbose=1)
loss = model.evaluate(X_test_reshaped, y_test, verbose=0)
print("loss:", loss)
model.save("btc_forecasting_model.h5")
