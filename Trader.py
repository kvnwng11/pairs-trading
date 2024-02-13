import numpy as np
import pandas as pd
import datetime as dt
import math
import yfinance as yf
from enum import Enum
import matplotlib.pyplot as plt

class PositionType(Enum):
    NONE = 0
    BUY = 1
    SELL = 2

class Trader:

    def __init__(self, coin1: str, coin2: str, window: int):
        """  Constructor  """
        self.coin1 = coin1
        self.coin2 = coin2
        self.window = window + 1
        self.state = PositionType.NONE
        self.weight1 = 0
        self.weight2 = 0
        self.data = {}

        # Trading parameters
        self.stop_loss = -0.05
        self.commission = 0.001
        self.tax = 0.20
        self.entry_zscore = 1
        self.exit_zscore = 0.5

        # Results of strategy
        self.returns = []
        self.taxed_returns = []
        self.zscores = []
        self.buy_signals = []
        self.sell_signals = []
        self.exit_signals = []
        
    def initialize_prices(self):
        """  Get prices for the last self.window days  """

        end = pd.to_datetime('today')
        start = end - dt.timedelta(days=self.window)

        # Get one month of prices
        self.data[self.coin1] = list(yf.download(self.coin1, start, end, progress=False)['Close'])
        self.data[self.coin2] = list(yf.download(self.coin2, start, end, progress=False)['Close'])

    def update_data(self, updates: dict):
        """  Drops oldest price point and adds one  """

        for coin, price in updates.items():
            # Not stored
            if coin not in self.data:
                self.data[coin] = []

            # Remove oldest
            if len(self.data[coin]) >= self.window:
                self.data[coin] = self.data[coin][1:]

            self.data[coin].append(price) # Add new price

    def trade(self):
        """  Check for trades to place  """

        # Not enough data
        if len(self.data[self.coin1]) < self.window or len(self.data[self.coin2]) < self.window:
            return
        
        # TODO: OLS to find beta
        beta = 1
        
        # Initilaize 30 days of prices
        past_x = self.data[self.coin1][:-1]
        past_y = self.data[self.coin2][:-1]

        # Initialize current price
        curr_x = self.data[self.coin1][-1]
        curr_y = self.data[self.coin2][-1]

        # Find zscore
        past_spread = np.log(past_y) - beta * np.log(past_x)
        curr_spread = np.log(curr_y) - beta * np.log(curr_x)
        z_score = (curr_spread - np.average(past_spread)) / np.std(past_spread)

        buy = None
        sell = None
        exit = None

        # Daily return
        current_return = self.weight1*(curr_x/self.data[self.coin1][-2] - 1) + self.weight2*(curr_y/self.data[self.coin2][-2] - 1)
        taxed_return = 0

        # Stop loss
        if current_return < self.stop_loss:
            self.weight1 = self.weight2 = 0
            self.state = PositionType.NONE
            current_return -= self.commission
            # TODO: Use API to place trade

            exit = z_score

        # Exit position
        elif self.state != PositionType.NONE and z_score >= -self.exit_zscore and z_score <= self.exit_zscore:
            self.weight1 = self.weight2 = 0
            self.state = PositionType.NONE

            if current_return > 0:
                taxed_return -= self.tax

            current_return -= self.commission
            # TODO: Use API to place trade


            exit = z_score

        # Sell the spread
        elif self.state == PositionType.NONE and z_score > self.entry_zscore:
            signal = 1
            self.weight1 = signal
            self.weight2 = -beta * signal
            self.state = PositionType.SELL
            current_return -= self.commission
            # TODO: Use API to place trade

            sell = z_score

        # Buy the spread
        elif self.state == PositionType.NONE and z_score < -self.entry_zscore:
            signal = -1
            self.weight1 = signal
            self.weight2 = -beta * signal
            self.state = PositionType.BUY
            current_return -= self.commission
            # TODO: Use API to place trade

            buy = z_score

        taxed_return += current_return

        self.returns.append(current_return)
        self.taxed_returns.append(taxed_return)
        self.zscores.append(z_score)
        self.buy_signals.append(buy)
        self.sell_signals.append(sell)
        self.exit_signals.append(exit)

    def results(self):
        """  Plot returns  """
        plt.figure(figsize=(20,12))
        plt.plot(np.append(1,np.cumprod(1+np.array(self.returns))))
        plt.plot(np.append(1,np.cumprod(1+np.array(self.taxed_returns))))
        plt.legend(['Net return', 'Taxed returns'])
        plt.show()

    def plot_zscores(self):
        """  Plot zscores  """
        plt.figure(figsize=(20,10))
        plt.plot(self.zscores, color = 'blue')
        plt.plot(self.buy_signals, color='green', marker='^', linestyle='dashed', markersize=12)
        plt.plot(self.sell_signals, color='red', marker='^', linestyle='dashed', markersize=12)
        plt.plot(self.exit_signals, color='yellow', marker='^', linestyle='dashed', markersize=12)
        plt.show()