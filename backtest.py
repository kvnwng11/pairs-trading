import pandas as pd
import yfinance as yf
from Trader import Trader

coin1 = 'BTC-USD'
coin2 = 'ETH-USD'

# Get data
start = '2023-01-01'
end = pd.to_datetime('today')
BTC = list(yf.download(coin1, start, end)['Open'])
ETH = list(yf.download(coin2, start, end)['Open'])

# Initialize
trader = Trader(coin1, coin2, window=30)

# Add prices to Trader object
for BTC_price, ETH_price in zip(BTC, ETH):
    # Add new price
    updates = {
        coin1: BTC_price,
        coin2: ETH_price
    }
    trader.update_data(updates)

    # Check for a trade
    trader.trade()

trader.results() # Plot gross returns
trader.plot_zscores() # Plot trade signals