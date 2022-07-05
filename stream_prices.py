import numpy as np
import pandas as pd
import datetime as dt
import os
from collections import deque
from io import StringIO
from binance.client import Client

api_key = 'XMkgHm97q4ScccpiRb1Lf0Ta8BYanEPHBjIgwZatDqaNkPcgfnuSUFaWLdOL97IF'
api_secret = '5E2PnhoRqx0nPD3QZ7cJaGvKgnlHuiGpd15x94eXKKaWvk2DtO8qNylrC63otSIj'
client = Client(api_key, api_secret)

path = 'csv/'
symbols = ['DOGE', 'SHIB', 'BTC', 'LTC', 'MATIC', 'XRP']


def get_historical_prices():
    def GetHistoricalData(coin, start_time, end_time):
        # Calculate the timestamps for the binance api function
        untilThisDate = end_time
        sinceThisDate = start_time
        # Execute the query from binance - timestamps must be converted to strings !
        candle = client.get_historical_klines(f'{coin}USDT', Client.KLINE_INTERVAL_1MINUTE, str(sinceThisDate), str(untilThisDate))

        # Create a dataframe to label all the columns returned by binance so we work with them later.
        df = pd.DataFrame(candle, columns=['time', 'open', 'high', 'low', 'close', 'volume', 'closeTime',
                          'quoteAssetVolume', 'numberOfTrades', 'takerBuyBaseVol', 'takerBuyQuoteVol', 'ignore'])
        # as timestamp is returned in ms, let us convert this back to proper timestamps.
        df['time'] = pd.to_datetime(df['time'], unit='ms').dt.strftime("%Y-%m-%d %H:%M:%S")

        # Get rid of columns we do not need
        df = df.drop(['closeTime', 'quoteAssetVolume', 'numberOfTrades',
                     'takerBuyBaseVol', 'takerBuyQuoteVol', 'ignore'], axis=1)

        return df[['time','close']]

    # create and populate needed csvs
    for asset in symbols:
        asset_path = path+asset+'.csv'
        # if no csv file exists, pull minute data from the last 30 days
        if not os.path.exists(asset_path):
            end = dt.datetime.utcnow()
            end = end - dt.timedelta(minutes=1, seconds=end.second, microseconds=end.microsecond)
            start = end - dt.timedelta(days=30)

            prices = pd.DataFrame(GetHistoricalData(asset, start, end))
            #print(prices)
            #prices.drop_duplicates(subset='time', keep='last', inplace=True)
            prices[['time', 'close']].to_csv(asset_path, mode='w', header=False, index=False)

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
        start = pd.to_datetime(
            prices['time'].iloc[-1]) + pd.Timedelta(minutes=1)
        end = (pd.Timestamp.now() - pd.Timedelta(minutes=1)).round(freq='T')
        until = start + min(pd.Timedelta(minutes=300), end-start)

        if start <= end:
            while until <= end:
                data = np.array(client.get_product_historic_rates(
                    asset+'-USD', granularity=60, start=start, end=until))
                if data.size == 1:
                    break
                data = data[::-1]
                df = pd.DataFrame(data)

                df.columns = ['time', 'low', 'high', 'open', 'close', 'volume']
                df.drop(['low', 'high', 'open', 'volume'],
                        axis=1, inplace=True)
                df['time'] = pd.to_datetime(df['time'], unit='s')
                prices = pd.concat([prices, df], ignore_index=True, axis=0)
                start += pd.Timedelta(minutes=300)
                until += pd.Timedelta(minutes=300)
            prices.drop_duplicates(subset='time', keep='last', inplace=True)
            prices[['time', 'close']].to_csv(
                asset_path, mode='a', header=False, index=False)


get_historical_prices()
#update_prices()
