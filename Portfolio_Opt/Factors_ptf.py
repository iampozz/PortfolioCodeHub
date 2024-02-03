import pandas as pd
import numpy as np

stock_data = pd.read_excel('C:\\Users\\pozzo\\Dropbox\\UniBG Works\\Ortobelli Paper\\Dow Jones excel.xlsx',
                           index_col = 'Dates')


stock_price = stock_data.iloc[:,0:30].fillna(method = 'ffill').loc['2013-12-05':]
pe_ratio =  stock_data.iloc[:,60:90].fillna(method = 'ffill').loc['2013-12-05':]
mkt_cap = stock_data.iloc[:,150:180].fillna(method = 'ffill').loc['2013-12-05':]
pb_ratio = stock_data.iloc[:,180:210].fillna(method = 'ffill').loc['2013-12-05':]

djia = pd.read_excel('C:\\Users\\pozzo\Dropbox\\Cincinelli Paper\\Dow Jones data.xlsx', 
                     index_col = 'dates').fillna(method = 'ffill')

dataset = pd.concat((stock_price, djia), ignore_index=True, axis = 1)
asset = list(stock_price.columns)
asset.append('DJIA')
dataset.columns = asset

returns = dataset.iloc[:,0:30].pct_change()

returns_index = dataset.iloc[:,30].pct_change()

rf = 0.02/252

cov = returns.rolling(2).cov(returns_index)
cov = cov.iloc[2:]
var_index = returns_index.rolling(2).var().dropna()

b_2 = pd.concat([var_index] * 30, ignore_index=True, axis = 1).dropna()
beta = cov.to_numpy() / b_2.to_numpy()
beta = pd.DataFrame(beta).shift(1)
beta = beta.iloc[1:]

returns_2 = returns.iloc[3:len(returns)]
returns_index_2 = returns_index.iloc[3:len(returns_index)]
risk_free = np.full((len(returns_2),30), rf)
index = pd.concat([returns_index_2] * 30, ignore_index=True, axis = 1)
daily_capm = risk_free + beta.to_numpy()*(index.to_numpy() - risk_free)


noise = returns_2 - daily_capm

def factor_portfolio(n, factor, returns, inv_proportion = False):
    "n = fragmentation into deciles, quartiles, ecc."
    "factor = dataframe that contains the factor which is used to build the portfolio (e.g. P/E, P/B, ecc.)"
    "returns = array of returns"
    "inv_prop = if FALSE the asset with the highest metric get the highest weight in the portfolio"
    phi = np.arange(n, 100 + n, n)
    frag = []
    for x in phi:
        frag.append(np.nanpercentile(factor, q = x, axis = 1))
    
    ptf = []
    for x in phi:
        ptf.append(np.zeros((factor.shape[0], factor.shape[1])))
    
    for l in range(0, factor.shape[0]):
        for c in range(0, factor.shape[1]):
            if factor.iloc[l,c] < frag[0][l]:
                ptf[0][l,c] = factor.iloc[l,c]
            else:
                ptf[0][l,c] = 0
                
                for p in range(1, len(ptf)):
                    if frag[p - 1][l]  < factor.iloc[l,c] <= frag[p][l]:
                        ptf[p][l,c] = factor.iloc[l,c]
                    else:
                        ptf[p][l,c] = 0
    
    w = []
    
    if inv_proportion == False:
        for p in range(0, len(ptf)):
            w_df  = ((ptf[p].T / np.sum(ptf[p].T, axis = 0)).T)
            print(np.sum(w_df, axis = 1))
            
            w.append(pd.DataFrame(w_df, columns = factor.columns, index = factor.index))
            
            
        ptf_returns = []
        
        for q in range(0, len(w)):
            j = str(q + 1)
            ptf_returns_df = (np.sum(w[q] * returns, axis = 1))
            ptf_returns.append(pd.DataFrame(ptf_returns_df, columns = ['Ptf ' + j], index = factor.index))
            
        ptf_returns = pd.concat(ptf_returns, axis = 1)
            
    if inv_proportion == True:
        for p in range(0, len(ptf)):
            b = 1/ptf[p]
            b[np.isinf(b)] = 0
            a = np.sum(b, axis = 1)
            w.append(b.T / a)
            print(np.sum(w[p], axis = 0))
    
        ptf_returns = []
        
        for q in range(0, len(w)):
            ptf_returns.append(np.sum(w[q].T * returns, axis = 1))
    
    return frag, w, ptf_returns

pe_ratio = pe_ratio.loc['2013-12-10':]
mkt_cap = mkt_cap.loc['2013-12-10':]
pb_ratio = pb_ratio.loc['2013-12-10':]

ret_array = returns_2.to_numpy()
pe = factor_portfolio(10, pe_ratio, ret_array, inv_proportion=False)
mkt = factor_portfolio(10, mkt_cap, ret_array, inv_proportion=False)
pb = factor_portfolio(10, pb_ratio, ret_array, inv_proportion=False)

pe_price = (pe[2] + 1).cumprod()
mkt_price = (mkt[2] + 1).cumprod()
pb_price = (pb[2] + 1).cumprod()

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

fig1, axes1 = plt.subplots(3,1, figsize = (18,10))
plt.suptitle('Portfolio evolution')
sns.lineplot(ax = axes1[0], data = pe_price)
axes1[0].set_title('P/E Portfolio')
axes1[0].set_ylabel('Price')

sns.lineplot(ax = axes1[1], data = mkt_price)
axes1[1].set_title('Mkt Cap. Portfolio')
axes1[1].set_ylabel('Price')

sns.lineplot(ax = axes1[2], data = pb_price)
axes1[1].set_title('P/B Portfolio')
axes1[1].set_ylabel('Price')
plt.tight_layout()

pe_ret = pe[2]
mkt_ret = mkt[2]
pb_ret = pb[2]

factor_corr = []

for t in range(0, pe[2].shape[1]):
    label = str(t + 1)
    factor_corr.append(pd.concat((pe_ret.iloc[:,t], mkt_ret.iloc[:,t], pb_ret.iloc[:,t]), axis = 1))
    factor_corr[t].columns = ['P/E ' + label, 'Mkt ' + label, 'P/B '+ label]
    factor_corr[t] = factor_corr[t].corr()
    
fig3, axes3 = plt.subplots(10,1, figsize = (15,20))
plt.suptitle('Factor Returns Correlation')
for t in range(0, pe[2].shape[1]):
            s = str(t + 1)
            sns.heatmap(ax =axes3[t], data = factor_corr[t], annot = True, cmap="crest", linewidths=.5, linecolor='black')
            axes3[t].set_title('Decile ' + s)
            axes3[t].set_yticklabels(axes3[t].get_yticklabels(),rotation = 45)
plt.tight_layout()

spread = []

for s in range(0,pe[2].shape[1] - 1):
    spread.append(pd.DataFrame(pe[2].iloc[:,s] - pe[2].iloc[:,s + 1]))
    spread.append(pd.DataFrame(mkt[2].iloc[:,s] - mkt[2].iloc[:,s + 1]))
    spread.append(pd.DataFrame(pb[2].iloc[:,s] - pb[2].iloc[:,s + 1]))
    
spread = pd.concat(spread, axis = 1, ignore_index=True)

variables = []
for h in range(0, noise.shape[1]):
    variables.append(pd.concat((noise.iloc[:,h], spread, pe_ratio, mkt_cap, pb_ratio), axis = 1, ignore_index=False).fillna(0))

def buildLaggedFeatures(s,lag=20,dropna=True):

    if type(s) is pd.DataFrame:
        new_dict={}
        for col_name in s:
            new_dict[col_name]=s[col_name]
            # create lagged Series
            for l in range(1,lag+1):
                new_dict['%s_lag%d' %(col_name,l)]=s[col_name].shift(l)
        res=pd.DataFrame(new_dict,index=s.index)

    elif type(s) is pd.Series:
        the_range=range(lag+1)
        res=pd.concat([s.shift(i) for i in the_range],axis=1)
        res.columns=['lag %d' %i for i in the_range]
    else:
        print ('Only works for DataFrame or Series')
        return None
    if dropna:
        return res.dropna()
    else:
        return res
    
lagged_variables = []

for g in range(0, len(variables)):
    lagged_variables.append(buildLaggedFeatures(variables[g], lag = 20))
