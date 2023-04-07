import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from var import VaR
from scipy.stats import norm
import yfinance as yf

plt.rcParams['figure.figsize'] = (8, 5)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
sns.set()


def get_data(stock_list, start, end):
    stock_data = yf.download(tickers=stock_list, start=start, end=end,
                             progress=False)['Close']
    return stock_data


def calc_hist_VaR(returns, alpha=5):
    if isinstance(returns, pd.Series):
        return np.percentile(returns, alpha)
    elif isinstance(returns, pd.DataFrame):
        return returns.aggregate(calc_hist_VaR, alpha=alpha)
    else:
        raise TypeError('Returns must be a Pandas Series or Pandas DataFrame')


def calc_hist_CVaR(returns, alpha=5):
    if isinstance(returns, pd.Series):
        belowVaR = returns <= calc_hist_VaR(returns, alpha=alpha)
        return returns[belowVaR].mean()
    elif isinstance(returns, pd.DataFrame):
        return returns.aggregate(calc_hist_CVaR, alpha=alpha)
    else:
        raise TypeError('Returns must be a Pandas Series or Pandas DataFrame')


def plot_VaR_CVaR(rets, VaR, CVaR, alpha=5):
    plt.figure(figsize=(6, 3))
    plt.plot(rets)
    plt.axhline(y=var, color='r', linestyle='--',
                label=f'VaR ({100-alpha}% confidence)')
    plt.axhline(y=cvar, color='g', linestyle='--',
                label=f'CVaR ({100-alpha}% confidence)')
    plt.legend()
    plt.xlabel('Date')
    plt.ylabel('Daily Returns')
    plt.title('Historical Daily Returns')
    plt.show()


def analyze_portfolio(data, weights, initial_investment, period=1, alpha=5):
    # Asset level
    daily_returns = np.log(data / data.shift(1)).dropna()
    cov_matrix = daily_returns.cov()
    asset_mean_returns = daily_returns.mean()
    
    # Portfolio level; remember we use this to say something about the future
    daily_returns['portfolio'] = np.dot(daily_returns, weights)
    pf_return = np.sum(weights * asset_mean_returns)
    pf_std = np.sqrt(weights.T.dot(cov_matrix).dot(weights))
    
    # Historical VaR & CVaR
    VaR = calc_hist_VaR(daily_returns, alpha=5)
    CVaR = calc_hist_CVaR(daily_returns, alpha=5)
    
    # Projections
    period = np.arange(1, period+1) # use array for graphing; access -1 for actual period
    pf_exp_return = pf_return * period
    pf_exp_std = pf_std * np.sqrt(period)
    VaR_exp = VaR * np.sqrt(period)
    CVaR_exp = CVaR * np.sqrt(period)
    
    # Investment level
    return_inv = np.round(initial_investment * period, 2)
    VaR_inv = np.round(initial_investment * -VaR_exp, 2)
    CVaR_inv = np.round(initial_investment * -CVaR_exp, 2)
    
    # Reporting
    print('Expected Portfolio Return:     ', return_inv[-1])
    print('Value at Risk 95th CI:         ', VaR_inv[-1])
    print('Conditional VaR 95th CI:       ', CVaR_inv[-1])
    
    # Plotting
    plot_VaR_CVaR()
    
    return VaR_inv, CVaR_inv


stocks = ['AAPL', 'MSFT', 'C', 'DIS']
start_date = pd.to_datetime('today') - pd.DateOffset(years=5)
end_date = pd.to_datetime('today')
weights = np.array([.3, .3, .2, .2])
period = 1  # days
initial_investment = 1000
data = get_data(stocks, start_date, end_date)
rets, var, cvar = analyze_portfolio(data, weights, initial_investment, period)