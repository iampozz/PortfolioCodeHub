import pandas as pd
from openbb_terminal.sdk import openbb
import yfinance as yf
from openbb_terminal import config_terminal as cfg
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from credential import openbb_cred

openbb.login(email=openbb_cred()[0],
             password=openbb_cred()[1], 
             token=openbb_cred()[2])

dow = openbb.stocks.options.load_options_chains("DIA")
dow = dow.chains

call = dow[dow['optionType'] == 'call']
put = dow[dow['optionType'] == 'put']

def m_currentday():
    now = datetime.now()
    midnight = datetime.combine(now.date(), datetime.min.time()) + timedelta(days=1)
    time_until_midnight = midnight - now
    minutes_until_midnight = time_until_midnight.total_seconds() / 60
    
    return round(minutes_until_midnight,0)

def m_tillexpiration():
    now = datetime.now()

    next_month = now.replace(day=1) + timedelta(days=32)

    third_friday = next_month.replace(day=1)

    while third_friday.weekday() != 4:
        third_friday += timedelta(days=1)

    third_friday += timedelta(weeks=2)

    time_until_third_friday = third_friday - now

    minutes_until_third_friday = time_until_third_friday.total_seconds() / 60

    return minutes_until_third_friday, third_friday.date()
    
m_1 = m_currentday()
m_standard = 510
m_weekly = 900
m_2 = m_tillexpiration()[0]
date_exp = m_tillexpiration()[1]

t_near = (m_1 + m_standard + m_2) / 525600
t_next = (m_1 + m_weekly + m_2 + 10080) / 525600

def near_next_option(option, exp, opt_type, next_option = 0):
    
    date = str(exp + timedelta(days = next_option))
    data = option[option['expiration'] <= date].groupby('strike')[['bid', 'ask']].mean()
    
    data = (data['bid'] + data['ask']) / 2
    data.name = opt_type
    
    return data

near_call = near_next_option(call, date_exp, 'call')
near_put = near_next_option(put, date_exp, 'put')
next_call = near_next_option(call, date_exp, 'call', 7)
next_put = near_next_option(put, date_exp,'put', 7)

near_diff = abs(near_call - near_put)
next_diff = abs(next_call - next_put)

near_strike = near_diff.idxmin()
next_strike = next_diff.idxmin()

ycrv_df = openbb.fixedincome.ycrv()

from scipy.interpolate import CubicSpline

cs = CubicSpline(ycrv_df['Maturity'], ycrv_df['Rate'] / 25200)

def day_diff():
    today = datetime.now()
    month_ahead = today + timedelta(days = 30)
    diff = (month_ahead - today).days
    
    data_range = pd.date_range(start=today, end = month_ahead, freq='D')
    
    return diff, data_range

x_fine = np.linspace(0, ycrv_df['Maturity'].iloc[0], day_diff()[0] + 1)
y_fine = cs(x_fine)
y_fine = pd.DataFrame(y_fine, index = day_diff()[1])

f_1 = round(float(near_strike + np.exp(t_near * y_fine.loc[str(date_exp)].to_numpy().flatten()) * min(near_diff)),2)
f_2 = round(float(next_strike + np.exp(t_next * y_fine.loc[str(date_exp + timedelta(days = 7))].to_numpy().flatten()) * min(next_diff)), 2)

print('near forward price = ' + str(f_1))
print('next forward price = ' + str(f_2))

k_0_near = float(pd.DataFrame(abs(near_diff.index - f_1), index = near_diff.index).idxmin().to_numpy().flatten())
k_0_next = float(pd.DataFrame(abs(next_diff.index - f_2), index = next_diff.index).idxmin().to_numpy().flatten())

def oom_option(dataset, k_0, call = True):
    if call == True:
        choice = dataset[dataset.index >= k_0]
        choice = choice.sort_index(ascending = False)
        
        for n in range(0, choice.shape[0]):
            if choice.iloc[n] == 0.00:
                if choice.iloc[n + 1] == 0.00:
                    choice = choice.iloc[0:n]
                    break
    
    if call == False:
        choice = dataset[dataset.index <= k_0]
        choice = choice.sort_index(ascending = False)
        
        for n in range(0, choice.shape[0]):
            if choice.iloc[n] == 0.00:
                if choice.iloc[n + 1] == 0.00:
                    choice = choice.iloc[0:n]
                    break
    
    return choice.sort_index(ascending = True)

choice_put = oom_option(near_put, k_0 = k_0_near, call = False)
choice_call = oom_option(near_call, k_0 = k_0_near)

choice_put_long = oom_option(next_put, k_0 = k_0_next, call = False)
choice_call_long = oom_option(next_call, k_0 = k_0_next)

near_merge = pd.concat([choice_put, choice_call])
near_merge[near_merge.index  ==  k_0_near] = (choice_call[choice_call.index == k_0_near] + 
                                              choice_put[choice_put.index == k_0_near]) / 2

next_merge = pd.concat([choice_put_long, choice_call_long])
next_merge[next_merge.index  ==  k_0_next] = (choice_call_long[choice_call_long.index == k_0_next] + 
                                              choice_put_long[choice_put_long.index == k_0_next]) / 2 

r_f1 = y_fine.loc[str(date_exp)].to_numpy().flatten()
r_f2 = y_fine.loc[str(date_exp + timedelta(days = 7))].to_numpy().flatten()

def variance(dataset, rf, t, f, k):
    c = np.zeros_like(dataset)
    c[0] = ((dataset.index[1] - dataset.index[0]) / dataset.index[0]**2) * np.exp(rf * t) * dataset.iloc[0]
    c[-1] = ((dataset.index[-1] - dataset.index[-2]) / dataset.index[-1]**2) * np.exp(rf * t) * dataset.iloc[-1]
    
    for n in range(1, len(c) - 1):
        c[n] = 2 /t * ((dataset.index[n + 1] - dataset.index[n - 1]) / dataset.index[n]**2) * np.exp(rf * t) * dataset.iloc[n]
    
   
    var = np.sum(c) - (1/t) * (f/k - 1)**2
    return var, c, np.sum(c)

var_1 = variance(near_merge, rf = r_f1, t = t_near, f = f_1, k = k_0_near)
var_2 = variance(next_merge, rf = r_f2, t = t_next, f = f_2, k = k_0_next)

def rate(t1,t2,var1,var2):
    minutes_1 = ((m_1 + m_weekly + m_2 + 10080) - 43200) / ((m_1 + m_weekly + m_2 + 10080) - (m_1 + m_standard + m_2))
    minutes_2 = (43200 - (m_1 + m_standard + m_2)) / ((m_1 + m_weekly + m_2 + 10080) - (m_1 + m_standard + m_2))
    minutes_3 = 525600 / 43200
    
    iv = 100 *  np.sqrt((t1 * var1 * minutes_1 + t2 * var2 * minutes_2) * minutes_3)
    
    return iv

implied_vol = rate(t_near, t_next, var_1[0], var_2[0])