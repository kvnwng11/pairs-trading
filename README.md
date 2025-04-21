# Pairs Trading

A lightweight pairs trading bot. Made to quickly and accurately backtest strategies. 

Pairs trading models the logarithmic prices of two assets as such:

$$\log(y_t) = \alpha + \beta \log(x_t) + \epsilon_t$$

where $\epsilon_t$ is a stationary process with mean zero. 

The spread $\log(y_t) - \beta \log(x_t)$ oscillates around some equilibrum. We go long one unit of $y_t$ and short $\beta$ units of $x_t$ when the spread is below some threshold. Conversely, we long $\beta$ units of $x_t$ and short one unit of $y_t$ when the spread is above some threshold. $\beta$ is estimated via rolling OLS and we bet on a return to the equilibrium. 

---

## Parameters

For risk management, trader uses a stop loss of 5% and assumes 0.1% commissions.

The bot buys the spread when the z-score is below -1, and shorts the spread when the z-score is above 1. Positions are exited when -0.5 $\leq$ z-score $\leq$ 0.5. 

---

## Example Backtests

Below are two backtests on daily prices at 9:00am from Jan 1, 2023 - Feb 19, 2024. 

Bitcoin and Etherem:

<img src="img/btc-eth.png" width="600">


Two meme coins:

<img src="img/meme-coins.png" width="600">
