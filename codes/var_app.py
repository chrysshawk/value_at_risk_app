import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# from var import VaR
# from scipy.stats import norm
import yfinance as yf

plt.rcParams['figure.figsize'] = (6, 4)
plt.rcParams['figure.dpi'] = 200
plt.rcParams['savefig.dpi'] = 200
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


def plot_VaR_CVaR(daily_returns, VaR, CVaR, alpha=.05):
    plt.figure(figsize=(6, 3))
    plt.plot(daily_returns)
    plt.axhline(y=VaR, color='r', linestyle='--',
                label=f'VaR ({100-alpha*100}% confidence)')
    plt.axhline(y=CVaR, color='g', linestyle='--',
                label=f'CVaR ({100-alpha*100}% confidence)')
    plt.legend()
    plt.xlabel('Date')
    plt.ylabel('Daily Returns')
    plt.title('Historical Daily Returns')
    plt.show()


def calc_VaR_CVaR_hist(returns, alpha):
    VaR = returns.quantile(alpha)
    CVaR = returns[returns <= VaR].mean()
    return VaR, CVaR

def analyze_portfolio(data, weights, initial_investment, period=1, alpha=.05):
    # Calctulate Returns
    daily_rets = np.log(data / data.shift(1)).dropna()
    daily_rets['PF'] = daily_rets.dot(weights)
    mean_rets = daily_rets.mean()
    
    # Metrics
    cov_matrix = daily_rets.iloc[:, 0:len(weights)].cov()
    pf_std = np.sqrt(weights.T.dot(cov_matrix).dot(weights))
    
    # Historical VaR & CVaR
    VaR, CVaR = calc_VaR_CVaR_hist(daily_rets, alpha)
    
    # Investment Adj VaR & CVaR
    inv_weighted = initial_investment * np.append(weights, 1)
    inv_rets = inv_weighted * daily_rets
    inv_VaR, inv_CVaR = calc_VaR_CVaR_hist(inv_rets, alpha)
    plot_VaR_CVaR(inv_rets['PF'], inv_VaR['PF'], inv_CVaR['PF'], alpha)
    
    # Projections
    period = np.arange(1, period+1)
    proj_rets = mean_rets * period
    proj_std = pf_std * np.sqrt(period)
    proj_VaR = inv_VaR * np.sqrt(period)
    proj_CVaR = inv_CVaR * np.sqrt(period)
    
    # Reporting
    print('Expected Portfolio Return:         ',
          initial_investment * proj_rets['PF'])
    print('Portfolio Value at Risk 95th CI:   ',
          initial_investment + proj_VaR['PF'])
    print('Portfolio Conditional VaR 95th CI: ',
          initial_investment + proj_CVaR['PF'])
    
    plot_VaR_CVaR(inv_rets['PF'], proj_VaR['PF'], proj_CVaR['PF'], alpha)

    return proj_VaR, proj_CVaR, proj_rets, proj_std


def plot_period_VaR_CVaR(inv_rets, inv_VaR, inv_CVaR, period):
    period = np.arange(1, period+1)
    
    plt.plot(var_horizon[:time_horizon], "o",
         c='blue', marker='*', label='IBM')
    

stocks = ['AAPL', 'MSFT', 'C', 'DIS']
start_date = pd.to_datetime('today') - pd.DateOffset(years=5)
end_date = pd.to_datetime('today')
weights = np.array([.3, .3, .2, .2])
period = 1  # days
alpha = .05
initial_investment = 1000

data = get_data(stocks, start_date, end_date)

analyze_portfolio(data, weights, initial_investment, period=1)