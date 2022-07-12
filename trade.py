import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from io import StringIO

data_path = '/home/kvnwng11/code/pairs-trading/csv/'
state_path = '/home/kvnwng11/code/pairs-trading/state/'
statefile = ''

pair = ['', '']

window = 30 * 1440
stop_loss = -0.05
commission = 0.001
entry_zscore = 1
exit_zscore = 0

signal = 0
current_return = 0
position0 = 0
position1 = 0
today = pd.to_datetime("today")  # get current timestamp
N = 31*1440


def zscore(data, curr):
    data = np.asarray(data)
    return (curr - np.average(data))/np.std(data)

# downloading price data for stocks and the market index
raw_data = pd.DataFrame()
for symbol in pair:
    p = data_path+symbol+'.csv'
    with open(p, 'r') as f:
        q = deque(f, N)
    df = pd.read_csv(StringIO(''.join(q)), header=None)
    df = df.drop(df.columns[[0]], axis=1)
    d = (df.iloc[1:, :]).iloc[:, 0].to_numpy().astype(float)
    raw_data[symbol] = d

# read in last state
last_state = pd.read_csv(state_path+statefile)
last_state.columns = ['timestamp', 'balance', 'current_return', 'gross_returns', 'net_returns','position0', 'price0', 'position1', 'price1', 'signal',  'numtrades', 'zscore']
last_state = last_state.tail(1)
signal = last_state['signal'].iloc[-1]
position0 = last_state['position0'].iloc[-1]
position1 = last_state['position1'].iloc[-1]
numtrades = last_state['numtrades'].iloc[-1]
balance = last_state['balance'].iloc[-1]
current_return = last_state['current_return'].iloc[-1]

# initialize variables
t = len(raw_data)-1
old_signal = signal
old_position0 = position0
old_position1 = position1
past_data = raw_data[[pair[0], pair[1]]][t-window-1:t-1]
past_price0 = np.asarray(past_data[pair[0]])
past_price1 = np.asarray(past_data[pair[1]])
curr_price0 = raw_data[pair[0]][t]
curr_price1 = raw_data[pair[1]][t]
past_spread = np.log(past_price0) - np.log(past_price1)
curr_spread = np.log(curr_price0) - np.log(curr_price1)
curr_zscore = zscore(past_spread, curr_spread)

"""
TODO: implement hedge ratio and fix return logic bug

# Johansen test for hedge ratio
res = coint_johansen(raw_data[[symbols[0],symbols[1]]], det_order=0, k_ar_diff=1)
print("Eigenvalue critical values:\n", res.cvm)
print("Eigenvalue statistic:\n", res.max_eig_stat)
print("Eigenvector:\n", res.evec)
johansen_stat = res.max_eig_stat[1]
b0 = res.evec[0][0]
b1 = res.evec[1][0]
"""
hedge_ratio = 1 #abs(b1/b0)

# check for stop loss
if current_return < stop_loss:
    signal = 0
# check if still in trade
else:
    # exit signal
    if curr_zscore >= exit_zscore-0.1 and curr_zscore <= exit_zscore+0.1:
        signal = 0
    # still in position
    elif np.sign(curr_zscore) == old_signal:
        signal = old_signal
        position0 = old_position0
        position1 = old_position1
    # trade signal if above 1 std
    elif curr_zscore > entry_zscore:
        signal = 1
        position0 = signal
        position1 = -hedge_ratio*signal
    elif curr_zscore < -entry_zscore:
        signal = -1
        position0 = signal
        position1 = -hedge_ratio*signal
    else:
        signal = old_signal

# calculate returns
gross = position0*(curr_price0/raw_data[pair[0]][t-1] - 1) + \
    position1*(curr_price1/raw_data[pair[1]][t-1] - 1)
net = gross - commission * \
    (abs(position0 - old_position0) + abs(position1 - old_position1))
if signal == old_signal:
    current_return = (1+current_return)*(1+gross)-1
else:
    current_return = gross
    numtrades += 1

# calculate returns
balance *= (1+net)
p0 = raw_data[pair[0]][t]
p1 = raw_data[pair[1]][t]
new_state = {'timestamp': [today],
             'balance': [balance],
             'current_return': [current_return],
             'gross_returns': [gross],
             'net_returns': [net],
             'position0': [position0],
             'price0': [p0],
             'position1': [position1],
             'price1': [p1],
             'signal': [signal],
             'numtrades': [numtrades],
             'zscore': [curr_zscore]}

update = pd.DataFrame(new_state)
update.to_csv(state_path+statefile, mode='a', header=False, index=False)
