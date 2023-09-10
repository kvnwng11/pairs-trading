import numpy as np
import pandas as pd
import datetime as dt
import os
from collections import deque
from io import StringIO
from binance.client import Client
import warnings
warnings.filterwarnings("ignore")

api_key = ''
api_secret = ''
client = Client(api_key, api_secret)

path = ''
symbols = []


def GetHistoricalData(coin, start_time, end_time):
    # Gets minute data in the range [start_time, end_time]
    untilThisDate = end_time
    sinceThisDate = start_time
    candle = client.get_historical_klines(
        f'{coin}USDT', Client.KLINE_INTERVAL_1MINUTE, str(sinceThisDate), str(untilThisDate))

    df = pd.DataFrame(candle, columns=['time', 'open', 'high', 'low', 'close', 'volume', 'closeTime',
                                       'quoteAssetVolume', 'numberOfTrades', 'takerBuyBaseVol', 'takerBuyQuoteVol', 'ignore'])
    df['time'] = pd.to_datetime(
        df['time'], unit='ms').dt.strftime("%Y-%m-%d %H:%M:%S")
    df = df.drop(['closeTime', 'quoteAssetVolume', 'numberOfTrades',
                  'takerBuyBaseVol', 'takerBuyQuoteVol', 'ignore'], axis=1)

    return df[['time', 'close']]


def get_historical_prices():
    # Gets minute data for all pairs
    for asset in symbols:
        asset_path = path+asset+'.csv'
        # if no csv file exists, pull minute data from the last 30 days
        if not os.path.exists(asset_path):
            end = dt.datetime.utcnow()
            end = end - \
                dt.timedelta(minutes=1, seconds=end.second,
                             microseconds=end.microsecond)
            start = end - dt.timedelta(days=32)

            prices = pd.DataFrame(GetHistoricalData(asset, start, end))
            prices[['time', 'close']].to_csv(
                asset_path, mode='w', header=False, index=False)


def update_prices():
    # Updates any gaps in the data
    for asset in symbols:
        asset_path = path+asset+'.csv'
        # load in csv
        with open(asset_path, 'r') as f:
            N = 2
            q = deque(f, N)
        prices = pd.read_csv(StringIO(''.join(q)), header=None)
        prices.columns = ['time', 'close']

        start = dt.datetime.strptime(
            prices['time'].iloc[-1], "%Y-%m-%d %H:%M:%S") + dt.timedelta(minutes=1)
        end = dt.datetime.utcnow()
        end = end - dt.timedelta(seconds=end.second,
                                 microseconds=end.microsecond)
        #print("start: ", start)
        #print("end: ", end)
        prices = pd.DataFrame(GetHistoricalData(asset, start, end))
        prices[['time', 'close']].to_csv(
            asset_path, mode='a', header=False, index=False)


get_historical_prices()
update_prices()
