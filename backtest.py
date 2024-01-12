import pandas as pd
import yfinance as yf
from Trader import Trader

# Get data
start = '2023-01-01'
end = pd.to_datetime('today')
btc = list(yf.download('BTC-USD', start, end)['Open'])
eth = list(yf.download('ETH-USD', start, end)['Open'])

# Initialize
trader = Trader("BTC-USD", "ETH-USD", window=30)

# Add prices to Trader object
for btc_price, eth_price in zip(btc, eth):
    # Add new price
    updates = {
        'BTC-USD': btc_price,
        'ETH-USD': eth_price
    }
    trader.update_data(updates)

    # Check for a trade
    trader.trade()

trader.results() # Plot gross returns
trader.plot_zscores() # Plot trade signals