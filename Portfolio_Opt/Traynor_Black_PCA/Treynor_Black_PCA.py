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

data = openbb.stocks.screener.screener_data(preset_loaded='sdk_sp500')

ticker = list(data['Ticker'])

price = openbb.stocks.ca.hist(ticker, start_date ='2022-06-30', candle_type='c')
price = price.drop(['KVUE', 'GEHC', 'VLTO'], axis = 1)

returns = np.log(price.shift(1) / price).dropna().to_numpy()

spy = openbb.etf.load('SPY', start_date = '2022-06-30').loc[:,'Close']
sp500 = yf.download('^GSPC', start = '2022-06-30').loc[:,'Close']

spy_ret = np.log(spy.shift(1) / spy).dropna().to_numpy()
sp500_ret = np.log(sp500.shift(1) / sp500).dropna().to_numpy()

fig, ax = plt.subplots(1,2, figsize=(12, 6))
ax[0].hist(sp500_ret, density = True)
ax[0].set_title('S&P 500 returns distribution')
ax[0].set_ylabel('Probability density')
ax[0].set_xlabel('Value')
ax[1].hist(spy_ret, density = True)
ax[1].set_title('SPY ETF returns distribution')
ax[1].set_ylabel('Probability density')
ax[0].set_xlabel('Value')
fig.suptitle('Returns distribution')

plt.figure(figsize = (12,10))
plt.plot((sp500 / sp500.iloc[0]) * 100)
plt.plot((spy / spy.iloc[0]) * 100)
plt.legend(['S&P 500', 'SPY ETF'])
plt.xlabel('Date')
plt.ylabel('Evolution')
plt.title('S&P 500 vs SPDR SPY ETF')

train = returns[0:200]
test = returns[200:300]
oos = returns[300:]

test_date = price.index[301:]

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def PCA_train(train_array, comp = 1):
    scaler = StandardScaler()
    
    standard_data = scaler.fit_transform(train_array)
    
    pca = PCA(comp)
    
    analysis = pca.fit(standard_data)
    
    eigenvalues = pca.explained_variance_
    eigenvectors = pca.components_
    pc= pca.transform(standard_data)
    var = pca.explained_variance_ratio_
    
    return eigenvalues, eigenvectors, pc, np.sum(var)

pca_1 = PCA_train(train)
pca_2 = PCA_train(train, 2)
pca_3 = PCA_train(train, 3)
pca_4 = PCA_train(train, 4)
pca_5 = PCA_train(train, 5)
pca_10 = PCA_train(train, 10)
pca_30 = PCA_train(train, 30)
pca_50 = PCA_train(train, 50)

def PCA_test(test_array, comp = 1):
    scaler = StandardScaler()
    
    standard_data = scaler.fit_transform(test_array)
    
    pca = PCA(comp)
    
    analysis = pca.fit(standard_data)
    
    eigenvalues = pca.explained_variance_
    eigenvectors = pca.components_
    pc= pca.transform(standard_data)
    var = pca.explained_variance_ratio_
    
    return eigenvalues, eigenvectors, pc, np.sum(var)

pca_1_t = PCA_test(test)
pca_2_t = PCA_test(test, 2)
pca_3_t = PCA_test(test, 3)
pca_4_t = PCA_test(test, 4)
pca_5_t = PCA_test(test, 5)
pca_10_t = PCA_test(test, 10)
pca_30_t = PCA_test(test, 30)
pca_50_t = PCA_test(test, 50)

from sklearn.linear_model import LinearRegression

def factor_model_noise(endog, exog, exog_test, endog_test, n_comp):
    prediction = np.zeros((test.shape[0], test.shape[1]))
    x = exog
    intercept = np.zeros(test.shape[1])
    beta = np.zeros((test.shape[1], n_comp))
    for n in range(0, train.shape[1]):
        y = endog[:, n]
        
        reg = LinearRegression().fit(x, y)
        prediction[:,n] = reg.predict(exog_test)
        intercept[n] = reg.intercept_
        beta[n] = reg.coef_
        
    noise = endog_test - prediction
    
    return noise, prediction, intercept, beta

noise_1 = factor_model_noise(train, pca_1[2], pca_1_t[2], test, 1)
noise_2 = factor_model_noise(train, pca_2[2], pca_2_t[2], test, 2)
noise_3 = factor_model_noise(train, pca_3[2], pca_3_t[2], test, 3)
noise_4 = factor_model_noise(train, pca_4[2], pca_4_t[2], test, 4)
noise_5 = factor_model_noise(train, pca_5[2], pca_5_t[2], test, 5)
noise_10 = factor_model_noise(train, pca_10[2], pca_10_t[2], test, 10)
noise_30 = factor_model_noise(train, pca_30[2], pca_30_t[2], test, 30)
noise_50 = factor_model_noise(train, pca_50[2], pca_50_t[2], test, 50)

market = np.mean(sp500_ret[200:300])
var_market = np.var(sp500_ret[200:300])
risk_free = openbb.economy.treasury(maturity = '10y', start_date = '2022-06-30')
risk_free = risk_free[risk_free != '-'].dropna()/25200
risk_free = risk_free.iloc[198:298].to_numpy()

def Traynor_Black(alpha, noise, beta, rm, rf):
    noise_var = np.var(noise[0], axis = 0)
    
    sum_app = np.sum(alpha / noise_var)
    
    wi = (alpha / noise_var) / sum_app
    
    alpha_active = np.dot(wi.T, alpha)
    betas = np.dot(wi.T, beta)
    beta_active = np.sum(betas)
    var_active = np.sum(wi ** 2 * noise_var)
    
    w0 = (alpha_active / var_active) * (var_market / (rm - rf))
    
    wA = w0 / (1 + (1 - beta_active) * w0)
    wP = 1 - wA
    
    wi_A = wi * wA
    
    return wA, wP, wi_A

treynor_1 = Traynor_Black(noise_1[2], noise_1, noise_1[3], market, np.mean(risk_free))
treynor_2 = Traynor_Black(noise_2[2], noise_2, noise_2[3], market, np.mean(risk_free))
treynor_3 = Traynor_Black(noise_3[2], noise_3, noise_3[3], market, np.mean(risk_free))
treynor_4 = Traynor_Black(noise_4[2], noise_4, noise_4[3], market, np.mean(risk_free))
treynor_5 = Traynor_Black(noise_5[2], noise_5, noise_5[3], market, np.mean(risk_free))
treynor_10 = Traynor_Black(noise_10[2], noise_10, noise_10[3], market, np.mean(risk_free))
treynor_30 = Traynor_Black(noise_30[2], noise_30, noise_30[3], market, np.mean(risk_free))
treynor_50 = Traynor_Black(noise_50[2], noise_50, noise_50[3], market, np.mean(risk_free))

ptf_1 = (np.dot(treynor_1[2], oos.T) + treynor_1[1] * spy_ret[300:]) + 1
ptf_2 = (np.dot(treynor_2[2], oos.T) + treynor_2[1] * spy_ret[300:]) + 1
ptf_3 = (np.dot(treynor_3[2], oos.T) + treynor_3[1] * spy_ret[300:]) + 1
ptf_4 = (np.dot(treynor_4[2], oos.T) + treynor_4[1] * spy_ret[300:]) + 1
ptf_5 = (np.dot(treynor_5[2], oos.T) + treynor_5[1] * spy_ret[300:]) + 1
ptf_10 = (np.dot(treynor_10[2], oos.T) + treynor_10[1] * spy_ret[300:]) + 1
ptf_30 = (np.dot(treynor_30[2], oos.T) + treynor_30[1] * spy_ret[300:]) + 1
ptf_50 = (np.dot(treynor_50[2], oos.T) + treynor_50[1] * spy_ret[300:]) + 1

plt.figure(figsize = (12,10))
plt.plot(test_date, np.cumprod(ptf_1))
plt.plot(test_date, np.cumprod(ptf_2))
plt.plot(test_date, np.cumprod(ptf_3))
plt.plot(test_date, np.cumprod(ptf_4))
plt.plot(test_date, np.cumprod(ptf_5))
plt.plot(test_date, np.cumprod(ptf_10))
plt.plot(test_date, np.cumprod(ptf_30))
plt.plot(test_date, np.cumprod(ptf_50))
plt.plot(test_date, np.cumprod(spy_ret[300:] + 1))
plt.plot(test_date, np.cumprod(sp500_ret[300:] + 1))
plt.xlabel('Dates')
plt.ylabel('Portfolio Evolution')
plt.title('Traynor-Black with PCA vs SPY')
plt.legend(['PCA_1', 'PCA_2', 'PCA_3', 'PCA_4', 'PCA_5', 'PCA_10', 'PCA_30', 'PCA_50', 'SPY ETF', 'S&P 500'], loc = 'upper right')
