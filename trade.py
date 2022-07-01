"""
Strategy adapted from https://www.youtube.com/c/NEDLeducation/featured
"""

# importing packages
import numpy as np
import pandas as pd
import statsmodels.api as sm
import scipy.optimize as spop
import matplotlib.pyplot as plt
from collections import deque  # for fast csv reading
from io import StringIO

symbols = ['DOGEUSDT', 'SHIBUSDT']  # pair to trade
statefile = 'balance.csv'

path = 'csv/'
N = 31*1440

window = 30*1440  # rolling window length
KPSS_max = 0.463  # maximum KPSS statistic (95% critical value)
unbiased = 1  # specify KPSS test (one-parameter unbiased or two-parameter)

# strategy parameters - trading fee, optimal entry (divergence), and stop-loss
fee = 0.001
entry = 0.02
stop_loss = -0.05

# initially start in cash
signal = 0
current_return = 0
position0 = 0
position1 = 0

# initialising arrays
gross_returns = np.array([])
net_returns = np.array([])
market_returns = np.array([])
signals = np.array([])
KPSS_stats = np.array([])
raw_data = pd.DataFrame()
today = pd.to_datetime("today")  # get current timestamp

# downloading price data for stocks and the market index
for symbol in symbols:
    p = path+symbol+'.csv'
    with open(p, 'r') as f:
        q = deque(f, N)
    df = pd.read_csv(StringIO(''.join(q)), header=None)
    df = df.drop(df.columns[[0]], axis=1)
    d = (df.iloc[1:, :]).iloc[:, 0].to_numpy().astype(float)
    raw_data[symbol] = d

if len(raw_data) < N:
    exit()

# read in last state
last_state = pd.read_csv(path+statefile)
last_state.columns = ['timestamp', 'balance', 'current_return', 'gross_returns', 'net_returns',
                      'position0', 'price0', 'position1', 'price1', 'signal',  'numtrades', 'KPSS_stats']
last_state = last_state.tail(1)
signal = last_state['signal'].iloc[-1]
position0 = last_state['position0'].iloc[-1]
position1 = last_state['position1'].iloc[-1]
numtrades = last_state['numtrades'].iloc[-1]
balance = last_state['balance'].iloc[-1]
current_return = last_state['current_return'].iloc[-1]

# initialize
t = len(raw_data)-2
old_signal = signal
old_position0 = position0
old_position1 = position1

# current data window
data = raw_data[t-window:t:1440]

# Function to optimize: stock2 = a + b*stock1
# OLS parameters as starting values
reg = sm.OLS(np.array(data[symbols[1]]),
             sm.add_constant(np.array(data[symbols[0]])))
res = reg.fit()
a0 = res.params[0]
b0 = res.params[1]
if unbiased == 1:
    # unbiased one-parameter KPSS
    def KPSS(b):
        a = np.average(data[symbols[1]] - b*data[symbols[0]])
        resid = np.array(data[symbols[1]] - (a + b*data[symbols[0]]))
        cumulative_resid = np.cumsum(resid)
        st_error = (np.sum(resid**2)/(len(resid)-2))**(1/2)
        KPSS = np.sum(cumulative_resid**2)/(len(resid)**2*st_error**2)
        return KPSS
    # minimizing the KPSS function (maximising the stationarity)
    res = spop.minimize(KPSS, b0, method='Nelder-Mead')
    KPSS_opt = res.fun
    # retrieving optimal parameters
    b_opt = float(res.x)
    a_opt = np.average(data[symbols[1]] - b_opt*data[symbols[0]])
else:
    # biased two-parameter KPSS
    def KPSS2(kpss_params):
        a = kpss_params[0]
        b = kpss_params[1]
        resid = np.array(data[symbols[1]] - (a + b*data[symbols[0]]))
        cumulative_resid = np.cumsum(resid)
        st_error = (np.sum(resid**2)/(len(resid)-2))**(1/2)
        KPSS = np.sum(cumulative_resid**2)/(len(resid)**2*st_error**2)
        return KPSS
    # minimizing the KPSS function (maximising the stationarity)
    res = spop.minimize(KPSS2, [a0, b0], method='Nelder-Mead')
    # retrieving optimal parameters
    KPSS_opt = res.fun
    a_opt = res.x[0]
    b_opt = res.x[1]

# simulate trading
if current_return < stop_loss:  # check if stop-loss is violated
    signal = 0
    #print('stop-loss triggered')
# if we are already in position, check whether the equilibrium is restored, continue in position if not
elif np.sign(raw_data[symbols[1]][t] - (a_opt + b_opt*raw_data[symbols[0]][t])) == old_signal:
    signal = old_signal
else:
    # only trade if the pair is cointegrated
    if KPSS_opt > KPSS_max:
        signal = 0
    # only trade if there are large enough profit opportunities (optimal entry)
    elif abs(raw_data[symbols[1]][t]/(a_opt + b_opt*raw_data[symbols[0]][t])-1) < entry:
        signal = 0
    else:
        signal = np.sign(raw_data[symbols[1]][t] -
                         (a_opt + b_opt*raw_data[symbols[0]][t]))

# update positions
position0 = signal
position1 = -signal

# calculate returns
gross = position0*(raw_data[symbols[0]][t+1]/raw_data[symbols[0]][t] - 1) + \
    position1*(raw_data[symbols[1]][t+1]/raw_data[symbols[1]][t] - 1)
net = gross - fee*(abs(position0 - old_position0) +
                   abs(position1 - old_position1))
balance *= (1+net)
if signal == old_signal:
    current_return = (1+current_return)*(1+gross)-1
else:
    current_return = gross

# update state file
p0 = raw_data[symbols[0]][t+1]
p1 = raw_data[symbols[1]][t+1]
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
             'KPSS_stats': [KPSS_opt]}
update = pd.DataFrame(new_state)
update.to_csv(path+statefile, mode='a', header=False, index=False)


# populating arrays
# KPSS_stats = np.append(KPSS_stats, KPSS_opt)
# signals = np.append(signals, signal)
# gross_returns = np.append(gross_returns, gross)
# net_returns = np.append(net_returns, net)
# market_returns = np.append(market_returns, market)


# building the output dataframe
# output = pd.DataFrame()
# output['KPSS'] = KPSS_stats
# output['signal'] = signals
# output['gross'] = gross_returns
# output['net'] = net_returns
# output['market'] = market_returns
# visualising the results
# plt.figure(figsize=(20, 12))
# plt.plot(np.append(1, np.cumprod(1+gross_returns)))
# plt.plot(np.append(1, np.cumprod(1+net_returns)))
# plt.plot(np.append(1, np.cumprod(1+market_returns)))
# plt.legend(['Gross returns', 'Net returns', 'Market returns'])
