# import terality as te
import pandas as pd
import numpy as np
import datetime
#%%
data_trade = pd.read_csv('test2108_2202_tick.csv')
#%%
data_book = pd.read_csv('book_preprocessor.csv')
#%%
data = data_book.iloc[:,1:]
#%%
data = data.fillna(method='ffill')
test_data = data.fillna(method='bfill')
test_data = test_data.replace(np.inf, 1)
test_data = test_data.replace(-np.inf, -1)
#%%
def get_vwap(data):
    v = data['volume'] - data['volume'].shift(1)
    p = data['last_price']
    data['last_price_vwap'] = np.sum(p*v) / np.sum(v)
    return data

time_group = data_time.set_index('datetime_').groupby(pd.Grouper(freq='1min')).apply(get_vwap)
#%%
test_data['timestamp'] = pd.to_datetime(test_data['timestamp'])
#%%
time_group = test_data.set_index('timestamp').groupby(pd.Grouper(freq='1min')).agg(np.mean)
# time_group = time_group[~time_group['wap1_shift2_mean'].isin([0])]
time_group = time_group.dropna(axis=0,how='all')
time_group = time_group.reset_index()
#%%
final_data.to_csv('test2209_03_31_04_08_2min_data_ru.csv')
#%%
data_2005 = pd.read_csv('test2005_11_26_03_18_2min_data_ru_last_price.csv')
data_2009 = pd.read_csv('test2009_03_19_07_30_2min_data_ru_last_price.csv')
data_2101 = pd.read_csv('test2101_07_31_11_24_2min_data_ru_last_price.csv')
data_2105 = pd.read_csv('test2105_11_25_03_29_2min_data_ru_last_price.csv')
data_2109 = pd.read_csv('test2109_03_30_08_03_2min_data_ru_last_price.csv')
# data_2110 = pd.read_csv('test2110_03_31_08_09_2min_data_rb.csv')
data_2201 = pd.read_csv('test2201_08_04_11_23_2min_data_ru_last_price.csv')
data_2205 = pd.read_csv('test2205_11_24_03_30_2min_data_ru_last_price.csv')
#%%
test_data = pd.concat([data_2005, data_2009, data_2101, data_2105, data_2109, data_2201, data_2205])
test_data = test_data.iloc[:,1:]
test_data = test_data.drop(['rolling_HR1_mean'], axis=1)
test_data['datetime'] = test_data['datetime'].astype('datetime64')
#%%
test_data['target'] = np.log(test_data['last_price']/test_data['last_price'].shift(1))*100
test_data['target'] = test_data['target'].shift(-1)
test_data = test_data.dropna(axis=0, how='any')
#%%
test_data = test_data.set_index('datetime')
#%%
from scipy import stats
import re
import warnings

def calcSpearman(data):

    ic_list_mean = []
    ic_list_std = []
    data = data.copy()
    # target = data['target']
    for column in list(data.columns[0:140]):

        ic = data[column].rolling(200).corr(data['target'])
        ic_mean = np.mean(ic)
        ic_std = np.std(ic)
        print('ic_mean',ic_mean)
        print('ic_std',ic_std)
        ic_list_mean.append(ic_mean)
        ic_list_std.append(ic_std)

        # print(ic_list)

    return ic_list_mean, ic_list_std

IC_mean, IC_std = calcSpearman(test_data)
#%%
IC = pd.DataFrame({'IC_mean':IC_mean,'IC_std':IC_std})
columns = pd.DataFrame(test_data.columns)

IC_columns = pd.concat([columns, IC], axis=1)
col = ['variable','IC_mean','IC_std']
IC_columns.columns = col
#%%
IC_columns['IR'] = np.abs(IC_columns['IC_mean'])/IC_columns['IC_std']
#%%
filter_value = 0.11
filter_value2 = -0.11
x_column_mean = IC_columns.variable[(IC_columns.IC_mean > filter_value) & (IC_columns.IR > 0.5)]
y_column_mean = IC_columns.variable[(IC_columns.IC_mean < filter_value2) & (IC_columns.IR > 0.5)]
#%%
x_column = x_column_mean.tolist()
y_column = y_column_mean.tolist()
final_col = x_column+y_column
#%%
final_data = test_data.reindex(columns=final_col)
#%%
final_data['volume'] = test_data['volume_x']
#%%
final_data = final_data.reset_index()
#%%
final_data.to_csv('test2001_2205_2min_p.csv')
#%%
test_data = test_data.reset_index()
#%%
test_data.to_csv('test2201_2205_min_ru_394var.csv')
#%%
final_data.to_csv('test2101_min_ru_withoutminfeature.csv')


