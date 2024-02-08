import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

price = pd.read_excel('C:\\Users\\pozzo\\Dropbox\\Neural Networks\\Tesi forse\\Tesi EuroStoxx 50\\Dati NN.xlsx',\
                                    index_col = 'Dates', sheet_name = 'Price')
pe_ratio = pd.read_excel('C:\\Users\\pozzo\\Dropbox\\Neural Networks\\Tesi forse\\Tesi EuroStoxx 50\\Dati NN.xlsx',\
                                    index_col = 'Dates', sheet_name = 'PE Ratio')
esg = pd.read_excel('C:\\Users\\pozzo\\Dropbox\\Neural Networks\\Tesi forse\\Tesi EuroStoxx 50\\Dati NN.xlsx',\
                                    index_col = 'Dates', sheet_name = 'ESG Score')
p_c_ratio = pd.read_excel('C:\\Users\\pozzo\\Dropbox\\Neural Networks\\Tesi forse\\Tesi EuroStoxx 50\\Dati NN.xlsx',\
                                    index_col = 'Dates', sheet_name = 'PutCall ratio')
sp = pd.read_excel('C:\\Users\\pozzo\\Dropbox\\Neural Networks\\Tesi forse\\Tesi EuroStoxx 50\\Dati NN.xlsx',\
                                    index_col = 'Dates', sheet_name = 'S&P 500 price')
dow = pd.read_excel('C:\\Users\\pozzo\\Dropbox\\Neural Networks\\Tesi forse\\Tesi EuroStoxx 50\\Dati NN.xlsx',\
                                    index_col = 'Dates', sheet_name = 'Dow Jones')
oil = pd.read_excel('C:\\Users\\pozzo\\Dropbox\\Neural Networks\\Tesi forse\\Tesi EuroStoxx 50\\Dati NN.xlsx',\
                                    index_col = 'Dates', sheet_name = 'Oil')
gold = pd.read_excel('C:\\Users\\pozzo\\Dropbox\\Neural Networks\\Tesi forse\\Tesi EuroStoxx 50\\Dati NN.xlsx',\
                                    index_col = 'Dates', sheet_name = 'Gold')
vix = pd.read_excel('C:\\Users\\pozzo\\Dropbox\\Neural Networks\\Tesi forse\\Tesi EuroStoxx 50\\Dati NN.xlsx',\
                                    index_col = 'Dates', sheet_name = 'VIX Index')
rf = pd.read_excel('C:\\Users\\pozzo\\Dropbox\\Neural Networks\\Tesi forse\\Tesi EuroStoxx 50\\Dati NN.xlsx',\
                                    index_col = 'Date', sheet_name = 'Risk Free')
rf = rf / 252

name = price.columns.to_list()

# DATA MANAGEMENT
#Data cleaning, features engineering

lag = 10
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
daily_ret = price.pct_change()
lagged_ret = []
for n in range(len(name)):
    a = buildLaggedFeatures(price.iloc[:,n].pct_change(), lag = lag)
    lagged_ret.append(pd.concat([daily_ret.iloc[:,n],a], axis = 1, ignore_index = False).dropna())
    lagged_ret[n].columns = name[n] + lagged_ret[n].columns
    lagged_ret[n] = lagged_ret[n].drop([name[n]  + 'lag 0'], axis = 1)


frames = []

for n in range(len(name)):
    frames.append(pd.concat([pe_ratio.iloc[:,n], esg.iloc[:,n], p_c_ratio.iloc[:,n],
                             sp.iloc[:,0], dow.iloc[:,0], oil.iloc[:,0], gold.iloc[:,0], vix.iloc[:,0]
                             ], 
                             axis = 1, ignore_index = False,
                            keys = [name[n] + ' P/E', name[n] + ' ESG' ,name[n] + ' PutCallR',
                                    'S&P500','Dow Jones', 'WTI', 'Gold', 'VIX'
                                    ]))

data = []
for n in range(len(name)):
    data.append(frames[n].pct_change())

def buildLaggedFeatures_new(s,lag=20,dropna=True):

    if type(s) is pd.DataFrame:
        new_dict={}
        for col_name in s:
            new_dict[col_name]=s[col_name]
            # create lagged Series
            for l in range(1,lag+1):
                new_dict['%s_lag %d' %(col_name,l)]=s[col_name].shift(l)
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
data_lagged = []
for n in range(len(name)):
    b = buildLaggedFeatures_new(data[n], lag = lag)
    data_lagged.append(b)

raw_data = []
for n in range(len(name)):
    raw_data.append(pd.concat([lagged_ret[n], data_lagged[n]], 
                              axis = 1, ignore_index = False).dropna())


#PREDICTORS & TARGETS

predictors, target = [], []

for n in range(len(name)):
    target.append(raw_data[n].iloc[:,0])
    
    x = raw_data[n].drop([name[n]+name[n]], axis = 1)
    predictors.append(x)

corr = []

for n in range(len(name)):
    corr.append(predictors[n].corr())

#TRAIN-TEST SPLIT

predictors_train, predictors_test, target_train, target_test, oos_train, oos_test = [],[],[],[], [], []

for n in range(len(name)):
    
    predictors_train.append(predictors[n].loc[:'2020/12/31'].to_numpy())
    predictors_test.append(predictors[n].loc['2021/01/01':'2022/12/30'].to_numpy())
    target_train.append(target[n].loc[:'2020/12/31'].to_numpy())
    target_test.append(target[n].loc['2021/01/01':'2022/12/30'].to_numpy())
    oos_train.append(predictors[n].loc['2023/01/02':].to_numpy())
    oos_test.append(target[n].loc['2023/01/02':].to_numpy())
    
#Deep Learning Section

#Data Standardization
from sklearn.preprocessing import StandardScaler

norm = StandardScaler()
norm_t = StandardScaler()
for n in range(len(name)):
    # norm.fit(predictors_train[n])
    predictors_train[n] = norm.fit_transform(predictors_train[n])
    #norm_t.fit(target_train[n].reshape(len(target_train[n]),1))
    predictors_test[n] = norm.fit_transform(predictors_test[n])
    #target_train[n] = norm_t.fit_transform(target_train[n].reshape(len(target_train[n]),1))
    oos_train[n] = norm.fit_transform(oos_train[n])

#reshape arrays in 3D Dormat

for s in range(len(name)):
    predictors_train[s] = np.reshape(predictors_train[s],
                                      (predictors_train[s].shape[0],1, predictors_train[s].shape[1]))
    predictors_test[s] = np.reshape(predictors_test[s],
                                    (predictors_test[s].shape[0],1, predictors_test[s].shape[1]))
    target_train[s] = np.reshape(target_train[s],
                                  (target_train[s].shape[0]))
    target_test[s] = np.reshape(target_test[s],
                                (target_test[s].shape[0]))
    oos_train[s] = np.reshape(oos_train[n],
                              (oos_train[s].shape[0], 1, oos_train[s].shape[1]))
    oos_test[s] = np.reshape(oos_test[s],
                             (oos_test[s].shape[0]))
print(predictors_train[0].shape)
print(predictors_test[0].shape)
print(target_train[0].shape)
print(target_test[0].shape)
print(oos_train[0].shape)
print(oos_test[0].shape)

#LSTM Model

from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, Adagrad
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping, History
import tensorflow as tf
from keras_tuner import GridSearch
import tensorflow_addons as tfa

epochs = 100
input_shape = (1,predictors_train[0].shape[2])

log_space = (np.logspace(-3,0,10))
log_space = list(log_space)


# Definizione del modello di deep learning
def build_model(hp):
    
    hp_activation = hp.Choice('activation', values = ['linear', 'relu'])
    
    model = Sequential()
    model.add(layers.InputLayer(input_shape))
    
    model.add(layers.Conv1D(filters = 64, kernel_size = 3, padding = 'same', activation = hp_activation ))
    
    model.add(layers.MaxPooling1D(pool_size=1))
    
    model.add(layers.Reshape((1,-1)))
    

    hp_layer_1_l2 = hp.Choice('l2',  values = [0.001, 
                                               0.0021544346900318843,
                                               0.004641588833612777,
                                               0.01,
                                               0.021544346900318832,
                                               0.046415888336127774,
                                               0.1,
                                               0.21544346900318823,
                                               0.46415888336127775,
                                               1.0])
    model.add(layers.LSTM(units = 50,
                          activation = 'tanh',
                          return_sequences = True ,
                          dropout = 0.1,
                          kernel_regularizer=regularizers.l2(hp_layer_1_l2)))
    
    hp_layer_2_l2 = hp.Choice('l2_2',  values = [0.001, 
                                               0.0021544346900318843,
                                               0.004641588833612777,
                                               0.01,
                                               0.021544346900318832,
                                               0.046415888336127774,
                                               0.1,
                                               0.21544346900318823,
                                               0.46415888336127775,
                                               1.0])
    model.add(layers.LSTM(units = 25,
                          activation = 'tanh', 
                          return_sequences = False ,
                          dropout = 0.1,
                          kernel_regularizer=regularizers.l2(hp_layer_2_l2)))
    
    model.add(layers.Dense(units =25,
                           activation = hp_activation))
    model.add(layers.Dense(1, activation = 'linear'))
    
    model.compile(optimizer=Adagrad(hp.Choice('learning_rate', 
                                              values = [0.001,
                                         0.0021544346900318843,
                                         0.004641588833612777,
                                         0.01,
                                         0.021544346900318832,
                                         0.046415888336127774,
                                         0.1,
                                         0.21544346900318823,
                                         0.46415888336127775,
                                         1.0])),
                 loss='mse',
                 metrics = ['mean_absolute_error', 'mean_squared_error', tfa.metrics.RSquare()])
    return model
# Definizione della griglia di ricerca degli iperparametri

tuner_grid = []
for n in range(len(name)):
    tuner_grid.append(GridSearch(
        build_model,
        objective='loss',
        max_trials=10,
        project_name = name[n]
        )
            )
# Esecuzione della ricerca degli iperparametri
early_stop = EarlyStopping(monitor = 'loss', patience = 10)
grid_search = []
best_models = []
best_hps = []
forecast = []
history = []
loss = []
mae = []
mse = []
r_squared = []
for n in range(len(name)):
    grid_search.append(tuner_grid[n].search(predictors_train[n], target_train[n], epochs=epochs))
    # Stampare il miglior modello e la combinazione di iperparametri
    best_models.append(tuner_grid[n].get_best_models(num_models=1)[0])
    best_hps.append(tuner_grid[n].get_best_hyperparameters(num_trials=1)[0])
    # Fit del modello con gli iperparametri migliori
    history.append(best_models[n].fit(predictors_train[n], target_train[n], epochs=epochs, 
                                      callbacks = [early_stop, History()]))
    forecast.append(best_models[n].predict(predictors_test[n]))
    loss.append(history[n].history['loss'])
    mae.append(history[n].history['mean_absolute_error'])
    mse.append(history[n].history['mean_squared_error'])
    r_squared.append(history[n].history['r_square'])

#oos prediction

oos_forecast = []
for n in range(len(name)):
    oos_forecast.append(best_models[n].predict(oos_train[n]))
