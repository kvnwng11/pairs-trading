import pandas as pd
import numpy as np
import statsmodels.api as sm
from collections import deque
from io import StringIO

data_path = '/home/kvnwng11/code/pairs-trading/csv/'
state_path = '/home/kvnwng11/code/pairs-trading/state/'
statefile = ''

pair = ['', '']
x_label = pair[0]
y_label = pair[1]

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
x = np.array(past_data[x_label])
y = np.array(past_data[y_label])
curr_x = raw_data[x_label][t]
curr_y = raw_data[y_label][t]

# simple beta
reg = sm.OLS(np.log(y), sm.add_constant(np.log(x)))
reg = reg.fit()
a0 = reg.params[0]
b0 = reg.params[1]
hedge_ratio = b0

# find current zscore
past_spread = np.log(y) - hedge_ratio*np.log(x)
curr_spread = np.log(curr_y) - hedge_ratio*np.log(curr_x)
curr_zscore = zscore(past_spread, curr_spread)

# find derivatives
first_d = np.gradient((past_spread))
second_d = np.gradient(first_d)
first_d = first_d[-1]
second_d = second_d[-1]

# check for stop loss
if current_return < stop_loss:
    signal = 0
# check if still in trade
else:
    # decide to exit
    if curr_zscore >= exit_zscore-0.1 and curr_zscore <= exit_zscore+0.1:
        signal = old_signal = 0
    elif signal == 1 and first_d > 0 and second_d > 0:
        signal = old_signal = 0
    elif signal == -1 and first_d < 0 and second_d < 0:
        signal = old_signal = 0
    
    # decide to trade
    if np.sign(curr_zscore) == old_signal:
        signal = old_signal
        position0 = old_position0
        position1 = old_position1
    # sell signal
    elif curr_zscore > entry_zscore and first_d < 0 and second_d < 0:
        signal = 1
        position1 = signal
        position0 = -hedge_ratio*signal
    # buy signal
    elif curr_zscore < -entry_zscore and first_d > 0 and second_d > 0:
        signal = -1
        position1 = signal
        position0 = -hedge_ratio*signal
    # default: do nothing
    else:
        signal = 0

# calculate returns
gross = position0*(curr_x/raw_data[x_label][t-1] - 1) + position1*(curr_y/raw_data[y_label][t-1] - 1)
net = gross - commission * \
    (abs(position0 - old_position0) + abs(position1 - old_position1))
if signal == old_signal:
    current_return = (1+current_return)*(1+gross)-1
else:
    current_return = gross
    numtrades += 1

if signal == 0:
    current_return = 0

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
