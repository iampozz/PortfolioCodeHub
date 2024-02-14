import pandas as pd
import numpy as np
import scipy.stats as stats

matrix = pd.read_excel('C:\\Users\\pozzo\\OneDrive\\Documenti\\CFA\\Matrice Pozz.xlsx', sheet_name = 'Definitivo_2',\
                       index_col = 'year')

change = matrix.pct_change().dropna()
mean_matrix = change.mean()
cov_matrix = change.cov()

drivers = stats.multivariate_normal(mean_matrix, cov_matrix, allow_singular=True)
n = 50000
years = 6
fcfo_f = np.zeros((years, n))
fcfo_f[0] = matrix['FCFO'].iloc[-1]
pfn_f = np.zeros((years, n))
pfn_f[0] = matrix['PFN'].iloc[-1]

rate = np.zeros(n * years)
rate_pfn = np.zeros(n * years)

for i in range(0, n * years):
    rate[i] = drivers.rvs()[4]
    rate_pfn[i] = drivers.rvs()[5]
    
rate = np.reshape(rate, (years, n))
rate_pfn = np.reshape(rate_pfn, (years, n))

for i in range(1, years):
    for n in range(0, n):
        fcfo_f[i,n] = fcfo_f[i-1,n] * (1 + rate[i - 1, n])
        pfn_f[i,n] = pfn_f[i-1,n] * (1 + rate_pfn[i - 1, n])
        
mean = np.mean(fcfo_f, axis = 1)
mean_pfn = np.mean(pfn_f, axis = 1)

pfn_f = pfn_f.T
