import numpy as np
import pandas as pd
import os
from collections import deque
from io import StringIO
import cbpro  # coinbase pro api

client = cbpro.PublicClient()
path = 'csv/'
symbols = ['DOGE', 'SHIB']


def get_historical_prices():
    date = pd.Timestamp.now().round(freq='s')

    # create and populate needed csvs
    for asset in symbols:
        asset_path = path+asset+'.csv'
        # if no csv file exists pull minute data from the last 30 days
        if not os.path.exists(asset_path):
            end = (pd.Timestamp.now()).round(
                freq='T') - pd.Timedelta(minutes=1)
            start = end - pd.Timedelta(days=30)
            until = start + pd.Timedelta(minutes=300)
            prices = pd.DataFrame()
            while until <= end:
                data = np.array(client.get_product_historic_rates(asset+'-USD', granularity=60, start=start, end=until))
                if data.size == 1:
                    break
                data = data[::-1]
                df = pd.DataFrame(data)
                prices = pd.concat([prices, df], ignore_index=True, axis=0)
                start += pd.Timedelta(minutes=300)
                until += pd.Timedelta(minutes=300)
            prices.columns = ['time', 'low', 'high', 'open', 'close', 'volume']
            prices['time'] = pd.to_datetime(prices['time'], unit='s')
            prices.drop_duplicates(subset='time', keep='last', inplace=True)
            prices[['time', 'close']].to_csv(asset_path, mode='a', header=False, index=False)

def update_prices():
    # update any gaps in the csv file
    for asset in symbols:
        asset_path = path+asset+'.csv'
        # load in csv
        with open(asset_path, 'r') as f:
            N = 5
            q = deque(f, N)
        prices = pd.read_csv(StringIO(''.join(q)), header=None)
        prices.columns = ['time', 'close']
        start = pd.to_datetime(prices['time'].iloc[-1]) + pd.Timedelta(minutes=1)
        end = (pd.Timestamp.now() - pd.Timedelta(minutes=1)).round(freq='T')
        until = start + min(pd.Timedelta(minutes=300), end-start)

        if start <= end:
            while until <= end:
                data = np.array(client.get_product_historic_rates(asset+'-USD', granularity=60, start=start, end=until))
                if data.size == 1:
                    break
                data = data[::-1]
                df = pd.DataFrame(data)
                
                df.columns = ['time', 'low', 'high', 'open', 'close', 'volume']
                df.drop(['low', 'high', 'open', 'volume'],axis=1, inplace=True)
                df['time'] = pd.to_datetime(df['time'], unit='s')
                prices = pd.concat([prices, df], ignore_index=True, axis=0)
                start += pd.Timedelta(minutes=300)
                until += pd.Timedelta(minutes=300)
            prices.drop_duplicates(subset='time', keep='last', inplace=True)
            prices[['time', 'close']].to_csv(asset_path, mode='a', header=False, index=False)

get_historical_prices()
update_prices()
