import quandl
import pandas as pd
from openbb_terminal.sdk import openbb
import yfinance as yf
from openbb_terminal import config_terminal as cfg
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
from credential import openbb_cred

openbb.login(email=openbb_cred()[0],
             password=openbb_cred()[1], 
             token=openbb_cred()[2])
data = openbb.stocks.screener.screener_data(preset_loaded='djia')

ticker = list(data['Ticker'])

price_close = openbb.stocks.ca.hist(ticker, start_date='2018-01-01', candle_type='c')
price_close = price_close.loc['2019-03-20':].dropna()

returns = price_close.pct_change().dropna()
final_date = price_close.index[-1] + timedelta(days = 1)
benchmark = yf.download('^DJI', start='2019-03-20', end=final_date).loc[:,'Close'].dropna()

def one_over_n_ptf(data, wealth, r, rf, benchmark):
    "data = DataFrame of assets"
    "wealth = Initial amount of money"
    "r = DataFrame/Array of returns"
    "rf = risk free rate"
    "benchmark = DataFrame of Benchmark returns"
    
    w = np.full(data.shape[1],1 / data.shape[1])
    stocks = np.floor((w * wealth)/data.iloc[0])
    if np.dot(stocks, data.iloc[0].T) == wealth:
        print("all wealth is invested in stocks")
    else:
        liquidity = wealth - np.dot(stocks, data.iloc[0].T)
        print('wealth invested in stock = ' + str(np.dot(stocks, data.iloc[0].T)))
        print('wealth in liquidity = ' + str(liquidity))

    portfolio_value = np.dot(stocks, data.T) + liquidity
    portfolio_returns = pd.DataFrame(portfolio_value).pct_change().dropna()
    portfolio_returns.index = data.iloc[1:].index
    mean = np.float(np.mean(portfolio_returns, axis = 0))
    
    benchmark_ret = pd.DataFrame(benchmark).pct_change().dropna()
    benchmark_mean = np.float(np.mean(benchmark_ret, axis = 0))
    beta = np.cov(benchmark_ret, portfolio_returns, rowvar = False)[1,0] / benchmark_ret.std()
    te = pd.DataFrame(portfolio_returns.to_numpy() - benchmark_ret.to_numpy())
    
    metrics = {
        'Cumulative_Returns': (portfolio_value[-1] / portfolio_value[0]) - 1,
        'Cumulative_Returns_TS': (1 + portfolio_returns).cumprod() * wealth,
        'Portfolio_mean': mean,
        'Benchmark_mean': benchmark_mean,
        'Benchmark_Returns': (benchmark.iloc[-1] / benchmark.iloc[0]) - 1,
        'Benchmark_Returns_TS': (1 + benchmark_ret).cumprod() * wealth,
        'Portfolio_std': np.float(portfolio_returns.std()),
        'Benchmark_std': np.float(benchmark_ret.std()),
        'Portfolio_VaR(95%)': np.quantile(portfolio_returns, q = 0.05),
        'Benchmark_VaR(95%)': np.quantile(benchmark_ret, q = 0.05),
        'Beta': np.float(beta),
        'Tracking_Error': te,
        'Sharpe_Ratio': np.float((mean - rf/252)/portfolio_returns.std()),
        'Information_Ratio': np.float((mean - benchmark_mean)/np.std(te, axis = 0)),
        'Treynor_Index': np.float((mean - rf / 252)/ beta),
        'Omega_Ratio': np.float(np.sum(te[te>0])/-np.sum(te[te<0])),
        'Correlation': np.corrcoef(benchmark_ret, portfolio_returns, rowvar = False)[1,0]
        }
    
    plot, ax = plt.subplots(figsize = (12,10))
    ax.plot(metrics['Cumulative_Returns_TS'])
    ax.plot(metrics['Benchmark_Returns_TS'])
    ax.legend(['1/N Returns', 'Benchmark Returns'])
    ax.set_title('1-over-N Portfolio Evolution')
    ax.set_ylabel('Wealth Evolution ($)')
    
    return w, stocks, portfolio_value, metrics, ax

import cvxpy as cvx
def sharpe_ptf(data,opt_sample, performance_sample, wealth, r, rf, benchmark, omega_rebelancing):
    "opt_sample = DataFrame containing the data we want to use for the first optimization"
    "performance_sample = DataFrame containing the data used to assess ptf performance"
    "wealth = Initial amount of money"
    "r = DataFrame/Array of returns"
    "rf = risk free rate"
    "benchmark = DataFrame of Benchmark returns"
    "omega rebelancing = lookback days to consider new allocation"
    
    returns_sample = opt_sample.pct_change().dropna()
    sigma_sample = np.cov(returns_sample, rowvar = False)
    mean_sample = returns_sample.mean().to_numpy()
    rf_d = rf /  252
    w_0 = cvx.Variable(opt_sample.shape[1])
    obj = cvx.Minimize(cvx.quad_form(w_0, sigma_sample))
    constraints = [w_0 >= 0,
                   mean_sample.T @ w_0 - rf_d == 1]  
    problem = cvx.Problem(obj, constraints)
    problem.solve()
    
    w_0 = w_0.value
    w_0 = w_0 /np.dot((np.ones(opt_sample.shape[1]).T), w_0)
    
    sharpe_0 = (np.dot(mean_sample.T, w_0) - rf_d) / np.sqrt(np.dot(w_0, np.dot(sigma_sample, w_0.T)))
    stocks_0 = np.floor((w_0 * wealth) / performance_sample.iloc[0])
    
    stocks = np.zeros((performance_sample.shape[0], performance_sample.shape[1]))
    w = np.zeros((performance_sample.shape[0], performance_sample.shape[1]))
    omega = np.zeros((performance_sample.shape[0] - omega_rebelancing))
    sharpe = np.zeros((performance_sample.shape[0] - omega_rebelancing))
    te = np.zeros((performance_sample.shape[0] - 1))
    
    stocks[0:omega_rebelancing] = stocks_0
    w[0:omega_rebelancing] = w_0
    
    returns_perf = performance_sample.pct_change().dropna()
    benchmark_ret = pd.DataFrame(benchmark).pct_change().dropna()
    
    portfolio_value = np.zeros(performance_sample.shape[0])
    portfolio_value[0:omega_rebelancing] = np.sum(stocks[0:omega_rebelancing] * performance_sample.iloc[0: omega_rebelancing,:], axis = 1)
    
    liquidity = np.zeros(performance_sample.shape[0])
    liquidity[0:omega_rebelancing] = wealth - portfolio_value[0]
    portfolio_value[0:omega_rebelancing] = portfolio_value[0:omega_rebelancing] + liquidity[0:omega_rebelancing]
    
    sharpe[0] = np.float((pd.DataFrame(portfolio_value[0:omega_rebelancing]).pct_change().dropna().mean() - rf_d)\
                        /pd.DataFrame(portfolio_value[0:omega_rebelancing]).pct_change().dropna().std())
    
    te[0:omega_rebelancing - 1] = (pd.DataFrame(portfolio_value[0:omega_rebelancing]).pct_change().dropna().to_numpy().flatten()\
                                        - benchmark_ret.iloc[1:omega_rebelancing].to_numpy().flatten())
    
    omega[0] = np.float(np.sum(te[te>0])/-np.sum(te[te<0]))
    
    for n in range(0, omega.shape[0]):
        
        stocks[omega_rebelancing + n] = stocks_0
        w[omega_rebelancing + n] = w_0
        portfolio_value[omega_rebelancing + n] = np.sum(stocks[omega_rebelancing + n] * performance_sample.iloc[omega_rebelancing + n,:])
    
    ptf_ret = pd.DataFrame(portfolio_value).pct_change().dropna()
    
    print(omega.shape[0] + omega_rebelancing)
    for n in range(0, omega.shape[0]):
        te[omega_rebelancing - 1 + n] = np.float(ptf_ret.iloc[omega_rebelancing - 1 + n,:]) - np.float(benchmark_ret.iloc[omega_rebelancing - 1+ n,:])
        #te = pd.DataFrame(te)
        te_i = te[0: omega_rebelancing + n]
        omega[n] = (np.sum(te_i[te_i>0]) / -np.sum(te_i[te_i<0]))
    
    for n in range(0, omega.shape[0] - omega_rebelancing):
        if omega[n + omega_rebelancing] <= np.quantile(omega[0: omega_rebelancing + n], q = 0.25):
            returns_n = returns_sample.append(returns_perf.iloc[0: omega_rebelancing + n])
            sigma = np.cov(returns_n, rowvar = False)
            mean_sample = returns_n.mean().to_numpy()
            w_0 = cvx.Variable(opt_sample.shape[1])
            obj = cvx.Minimize(cvx.quad_form(w_0, sigma))
            constraints = [w_0 >= 0,
                           mean_sample.T @ w_0 - rf_d == 1]  
            problem = cvx.Problem(obj, constraints)
            problem.solve()
            
            w_0 = w_0.value
            w_0 = w_0 /np.dot((np.ones(opt_sample.shape[1]).T), w_0)
            
            w[omega_rebelancing + n] = w_0
            
            stocks[omega_rebelancing + n] = np.floor((w[omega_rebelancing + n] * portfolio_value[omega_rebelancing + n - 1]) / performance_sample.iloc[omega_rebelancing + n])
            portfolio_value[omega_rebelancing + n] = np.sum(stocks[omega_rebelancing + n] * performance_sample.iloc[omega_rebelancing + n])
            
        else:
            w[omega_rebelancing + n] = w[omega_rebelancing + n - 1]
            stocks[omega_rebelancing + n] = stocks[omega_rebelancing + n - 1]
            portfolio_value[omega_rebelancing + n] = np.sum(stocks[omega_rebelancing + n] * performance_sample.iloc[omega_rebelancing + n])
        
    portfolio_returns = pd.DataFrame(portfolio_value).pct_change().dropna()
    portfolio_returns.index = performance_sample.iloc[1:].index
    mean = np.float(np.mean(portfolio_returns, axis = 0))
    
    benchmark_ret = pd.DataFrame(benchmark).pct_change().dropna()
    benchmark_mean = np.float(np.mean(benchmark_ret, axis = 0))
    beta = np.cov(benchmark_ret, portfolio_returns, rowvar = False)[1,0] / benchmark_ret.std()
    te = ptf_ret.to_numpy() - benchmark_ret.to_numpy()
    
    metrics = {
        'Cumulative_Returns': (portfolio_value[-1] / portfolio_value[0]) - 1,
        'Cumulative_Returns_TS': (1 + portfolio_returns).cumprod() * wealth,
        'Portfolio_mean': mean,
        'Benchmark_mean': benchmark_mean,
        'Benchmark_Returns': (benchmark.iloc[-1] / benchmark.iloc[0]) - 1,
        'Benchmark_Returns_TS': (1 + benchmark_ret).cumprod() * wealth,
        'Portfolio_std': np.float(portfolio_returns.std()),
        'Benchmark_std': np.float(benchmark_ret.std()),
        'Portfolio_VaR(95%)': np.quantile(portfolio_returns, q = 0.05),
        'Benchmark_VaR(95%)': np.quantile(benchmark_ret, q = 0.05),
        'Beta': np.float(beta),
        'Tracking_Error': te,
        'Sharpe_Ratio': np.float((mean - rf/252)/portfolio_returns.std()),
        'Information_Ratio': np.float((mean - benchmark_mean)/np.std(te, axis = 0)),
        'Treynor_Index': np.float((mean - rf / 252)/ beta),
        'Omega_Ratio': np.float(np.sum(te[te>0])/-np.sum(te[te<0])),
        'Correlation': np.corrcoef(benchmark_ret, portfolio_returns, rowvar = False)[1,0]
        }
    
    plot, ax = plt.subplots(figsize = (12,10))
    ax.plot(metrics['Cumulative_Returns_TS'])
    ax.plot(metrics['Benchmark_Returns_TS'])
    ax.legend(['Max Shape Returns', 'Benchmark Returns'])
    ax.set_title('Max Sharpe Portfolio Evolution')
    ax.set_ylabel('Wealth Evolution ($)')
    
    return w_0, sharpe_0, w, stocks, sharpe, omega, portfolio_value, te, metrics

equal_weight = one_over_n_ptf(price_close.loc['2021-01-05':], 1000000, returns, rf = 0.02, benchmark = benchmark.loc['2021-01-05':])
sharpe = sharpe_ptf(data = price_close,
                    opt_sample = price_close.loc[:'2021-01-04'], 
                    performance_sample = price_close.loc['2021-01-05':], 
                    wealth = 1000000, 
                    r = returns.loc['2021-01-05':,:], 
                    rf = 0.02, 
                    benchmark = benchmark.loc['2021-01-05':], 
                    omega_rebelancing = 5)