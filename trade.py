import pandas as pd
import numpy as np
import statsmodels.api as sm
from collections import deque
from io import StringIO
import os
import shutil

data_path = '/home/kvnwng11/pairs-trading/csv/'
state_path = '/home/kvnwng11/pairs-trading/state/'
pairs = [
    ['DOGE', 'SHIB'],
    ['BTC', 'MATIC'],
    ['BTC', 'XRP'],
    ['ETH', 'ADA'],
    ['ETH', 'SOL'],
    ['ETH', 'XRP'],
    ['XRP', 'ADA']
]

window = 30 * 1440
stop_loss = -0.05
commission = 0.001
entry_zscore = 1
exit_zscore = 0

today = pd.to_datetime("today")  # get current timestamp
N = 31*1440 # number of prices to read in

def zscore(data, curr):
    data = np.asarray(data)
    return (curr - np.average(data))/np.std(data)

# trade function
def trade(pair):
    # initialize
    statefile = pair[0]+'-'+pair[1]+'.csv'
    x_label = pair[0]
    y_label = pair[1]
    x_position = 0
    y_position = 0
    signal = 0
    current_return = 0

    # create statefile if non-existent
    if not os.path.exists(state_path+statefile):
        src = state_path+'template.csv'
        dst = state_path+statefile
        shutil.copy(src, dst)

    # read in last state
    last_state = pd.read_csv(state_path+statefile)
    last_state.columns = ['timestamp', 'balance', 'returns', 'x_position', 'x_price', 'y_position', 'y_price', 'signal',  'numtrades', 'zscore']
    last_state = last_state.tail(1)
    signal = last_state['signal'].iloc[-1]
    x_old_position = last_state['x_position'].iloc[-1]
    y_old_position = last_state['y_position'].iloc[-1]
    numtrades = last_state['numtrades'].iloc[-1]
    balance = last_state['balance'].iloc[-1]

    # read in price data
    raw_data = pd.DataFrame()
    for symbol in pair:
        p = data_path+symbol+'.csv'
        with open(p, 'r') as f:
            q = deque(f, N)
        df = pd.read_csv(StringIO(''.join(q)), header=None)
        df = df.drop(df.columns[[0]], axis=1)
        d = (df.iloc[1:, :]).iloc[:, 0].to_numpy().astype(float)
        raw_data[symbol] = d

    # initialize variables
    t = len(raw_data)-1
    old_signal = signal
    past_data = raw_data[[x_label,y_label]][t-window-1:t-1]
    x = np.array(past_data[x_label])
    y = np.array(past_data[y_label])
    curr_x = raw_data[x_label][t]
    curr_y = raw_data[y_label][t]

    # simple beta
    reg = sm.OLS(np.log(y), sm.add_constant(np.log(x)))
    reg = reg.fit()
    b0 = reg.params[1]
    hedge_ratio = b0

    # find current zscore
    past_spread = np.log(y) - hedge_ratio*np.log(x)
    curr_spread = np.log(curr_y) - hedge_ratio*np.log(curr_x)
    curr_zscore = zscore(past_spread, curr_spread)

    current_return = x_old_position*(curr_x/raw_data[x_label][t-1] - 1) + y_old_position*(curr_y/raw_data[y_label][t-1] - 1)

    # check for stop loss
    if current_return < stop_loss:
        signal = 0
    # check if still in trade
    else:
        # decide to exit
        if curr_zscore >= -exit_zscore-0.1 and curr_zscore <= exit_zscore+0.1:
            signal = old_signal = 0
            x_position = y_position = 0
            current_return = (1-commission) * current_return
        elif signal == 1:
            signal = old_signal = 0
            x_position = y_position = 0
            current_return = (1-commission) * current_return
        elif signal == -1:
            signal = old_signal = 0
            x_position = y_position = 0
            current_return = (1-commission) * current_return
        
        # decide to trade
        if np.sign(curr_zscore) == old_signal:
            signal = old_signal
            x_position = x_old_position
            y_position = y_old_position
        # sell signal
        elif curr_zscore > entry_zscore:
            signal = 1
            x_position = signal
            y_position = -hedge_ratio*signal
        # buy signal
        elif curr_zscore < -entry_zscore:
            signal = -1
            x_position = signal
            y_position = -hedge_ratio*signal
        # do nothing
        else:
            signal = 0
            x_position = y_position = 0

    # commission calculation
    if signal != old_signal:
        current_return = (1-commission) * current_return
    if signal == 0:
        current_return = 0

    # calculate returns
    balance *= (1+current_return)

    # update state file
    new_state = {'timestamp': [today],
                'balance': [balance],
                'returns': [current_return],
                'x_position': [x_position],
                'x_price': [curr_x],
                'y_position': [y_position],
                'y_price': [curr_y],
                'signal': [signal],
                'numtrades': [numtrades],
                'zscore': [curr_zscore]}

    update = pd.DataFrame(new_state)
    update.to_csv(state_path+statefile, mode='a', header=False, index=False)

# driver function
def execute():
    for pair in pairs:
        trade(pair)

execute()