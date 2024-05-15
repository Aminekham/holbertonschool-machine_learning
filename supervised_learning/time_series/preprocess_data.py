#!/usr/bin/env python3
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

"""
Keeping only close(final price) and timestamp which are the only relevent
features for our dataset than dropping null values (they appear often in the dataset)
and finally using minmax scaling
minmax scaling equation: X = X - Xmin / Xmax - Xmin
"""


raw_coinbase_data = pd.read_csv("./coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv")
raw_bitstamp_data = pd.read_csv("./bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv")
coinbase_data = raw_coinbase_data[['Timestamp', 'Close']]
bitstamp_data = raw_bitstamp_data[['Timestamp', 'Close']]
coinbase_data['Timestamp'] = pd.to_datetime(coinbase_data['Timestamp'], unit='s')
bitstamp_data['Timestamp'] = pd.to_datetime(bitstamp_data['Timestamp'], unit='s')
coinbase_data.set_index('Timestamp', inplace=True)
bitstamp_data.set_index('Timestamp', inplace=True)
coinbase_data.dropna(inplace=True)
bitstamp_data.dropna(inplace=True)
scaler = MinMaxScaler()
scaled_coinbase_data = pd.DataFrame(scaler.fit_transform(coinbase_data), columns=coinbase_data.columns, index=coinbase_data.index)
scaled_bitstamp_data = pd.DataFrame(scaler.fit_transform(bitstamp_data), columns=bitstamp_data.columns, index=bitstamp_data.index)
scaled_coinbase_data.to_csv("coinbase_processed.csv")
scaled_bitstamp_data.to_csv("bitstamp_processed.csv")
