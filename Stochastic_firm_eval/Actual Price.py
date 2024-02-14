import pandas as pd
import numpy as np
import scipy.stats as stats

fcfo = pd.read_excel('C:\\Users\\pozzo\\OneDrive\\Documenti\\CFA\\Matrice Pozz.xlsx', sheet_name = 'FCFO',\
                       index_col = 'N Path').iloc[:,1:].to_numpy()

pfn = pd.read_excel('C:\\Users\\pozzo\\OneDrive\\Documenti\\CFA\\Matrice Pozz.xlsx', sheet_name = 'PFN',\
                       index_col = 'N Path').iloc[:,1].to_numpy()

    
wacc = np.array([0.05, 0.06,0.0655,0.07,0.08])
growth = np.array([0.005, 0.01, 0.0114,0.013,0.018])


def fair_price(fcfo, pfn, growth, wacc):
    
    prices = np.zeros(fcfo.shape[0])
    
    for n in range(0, fcfo.shape[0]):
        prices[n] = ((fcfo[n,0] * (1 + wacc)**-1 + fcfo[n,1] * (1 + wacc)**-2 + fcfo[n,2] * (1 + wacc)**-3 +\
                     fcfo[n,3] * (1 + wacc)**-4 + fcfo[n,4] * (1 + wacc)**-5) +\
                        ((fcfo[n,4] * (1 + growth)) / (wacc - growth)) - pfn[n]) / 175000000
    
    return prices

all_price = []
for w in wacc:
    for g in growth:
        all_price.append(fair_price(fcfo, pfn, g, w))


all_price_array = pd.DataFrame(all_price).T
mean_price = all_price_array.mean()

all_price_array_1 = all_price_array[all_price_array >=4.68]
all_price_array_1 = all_price_array_1[all_price_array <=10.36]
flat_1 = all_price_array_1.dropna().to_numpy().flatten()

import matplotlib.pyplot as plt
plt.figure()
plt.hist(flat_1, bins = 150, density = True, cumulative = True, label = 'Price CDF')
plt.scatter(6.654499394, 0.5, c = 'r', label = 'Expected Price')
plt.plot(np.full(150,6.654499394), np.full(150,np.linspace(0,0.5, 150)), c = 'r')
plt.plot(np.full(150, np.linspace(min(flat_1),6.654499394, 150)), np.full(150, 0.5), c = 'r')
plt.title('Distribution of Forecasted Prices')
plt.xlabel('Forecasted Price')
plt.ylabel('Frequency')
plt.xlim(min(flat_1))
plt.legend(loc = 'best')

all_price_array_3 = all_price_array[all_price_array >=4.68]
all_price_array_3 = all_price_array_3[all_price_array <=10.36]
flat_3 = all_price_array_3.dropna().to_numpy().flatten()

plt.figure()
plt.hist(flat_3, bins = 200)
plt.title('Prices Distribution')
plt.ylabel('Frequency')
plt.xlabel('Prices')

all_price_array_2 = all_price_array[all_price_array >=5.0]
all_price_array_2 = all_price_array_2[all_price_array <=8.50]
flat_2 = all_price_array_2.dropna().to_numpy().flatten()

prob = len(flat_2) / len(flat_3)