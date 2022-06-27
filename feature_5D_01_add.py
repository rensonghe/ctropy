from IPython.core.display import display, HTML
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#%%
data_orderflow_220201 = pd.read_csv('ETH_USDT-2022-01.csv')
col_1 = ['datetime','last_price','size','order_count']
data_orderflow_220201.columns = col_1
data_orderflow_220201['datetime'] = pd.to_datetime(data_orderflow_220201['datetime'])
time_group_1 = data_orderflow_220201.set_index('datetime').resample('1min').apply({'last_price':'last','size':'last'})
# time_group = time_group[~time_group['wap1_shift2_mean'].isin([0])]
time_group_1 = time_group_1.dropna(axis=0,how='all')
time_group_1 = time_group_1.reset_index()

data_orderflow_220202 = pd.read_csv('ETH_USDT-2022-02.csv')
col_2 = ['datetime','last_price','size','order_count']
data_orderflow_220202.columns = col_2
data_orderflow_220202['datetime'] = pd.to_datetime(data_orderflow_220202['datetime'])
time_group_2 = data_orderflow_220202.set_index('datetime').resample('1min').apply({'last_price':'last','size':'last'})
# time_group = time_group[~time_group['wap1_shift2_mean'].isin([0])]
time_group_2 = time_group_2.dropna(axis=0,how='all')
time_group_2 = time_group_2.reset_index()

data_orderflow_220203 = pd.read_csv('ETH_USDT-2022-03.csv')
col_3 = ['datetime','last_price','size','order_count']
data_orderflow_220203.columns = col_3
data_orderflow_220203['datetime'] = pd.to_datetime(data_orderflow_220203['datetime'])
time_group_3 = data_orderflow_220203.set_index('datetime').resample('1min').apply({'last_price':'last','size':'last'})
# time_group = time_group[~time_group['wap1_shift2_mean'].isin([0])]
time_group_3 = time_group_3.dropna(axis=0,how='all')
time_group_3 = time_group_3.reset_index()
#%%
data_orderflow_220201 = pd.read_csv('ETH_USDT-2022-01.csv')
col_1 = ['datetime','last_price','size','order_count']
data_orderflow_220201.columns = col_1
data_orderflow_220201['datetime'] = pd.to_datetime(data_orderflow_220201['datetime'])
#%%
time_group_1 = data_orderflow_220201.set_index('datetime').resample('1min').apply({'last_price':'last','size':'last'})
# time_group = time_group[~time_group['wap1_shift2_mean'].isin([0])]
time_group_1 = time_group_1.dropna(axis=0,how='all')
time_group_1 = time_group_1.reset_index()
#%%
import datetime
# data_kline_220201 = pd.read_csv('ETH_USDT-202201.csv')
data_kline_220201 = pd.read_csv('ETH_USDT-202201.csv')
col_2 = ['datetime', 'volume', 'close', 'high', 'low', 'open']
data_kline_220201.columns = col_2
#%%
convert = lambda x:datetime.datetime.fromtimestamp(x)
data_kline_220201['datetime'] = data_kline_220201['datetime'].apply(convert)
#%%
def calc_wap1(df):
    wap = (df['bid_price1'] * df['ask_size1'] + df['ask_price1'] * df['bid_size1']) / (df['bid_size1'] + df['ask_size1'])
    return wap

# Function to calculate second WAP
def calc_wap2(df):
    wap = (df['bid_price2'] * df['ask_size2'] + df['ask_price2'] * df['bid_size2']) / (df['bid_size2'] + df['ask_size2'])
    return wap

def calc_wap3(df):
    wap = (df['bid_price1'] * df['bid_size1'] + df['ask_price1'] * df['ask_size1']) / (df['bid_size1'] + df['ask_size1'])
    return wap

def calc_wap4(df):
    wap = (df['bid_price2'] * df['bid_size2'] + df['ask_price2'] * df['ask_size2']) / (df['bid_size2'] + df['ask_size2'])
    return wap


# Calculate the realized volatility
def realized_volatility(series):
    return np.sqrt(np.sum(series ** 2))

def realized_quarticity(series):
    return (np.sum(series**4)*series.shape[0]/3)

def reciprocal_transformation(series):
    return np.sqrt(1/series)*100000

def square_root_translation(series):
    return series**(1/2)


#%%
data = pd.read_csv('ETH-USDT-2021-08_2022-05-orderbooks.csv')
#%%
data['timestamp'] = pd.to_datetime(data['timestamp'])
#%%
data = data[data.timestampe<='2022-01-01']
#%%
def book_preprocessor(data):

    df = data

    rolling = 60

    # Calculate Wap
    df['wap1'] = calc_wap1(df)
    df['wap2'] = calc_wap2(df)
    df['wap3'] = calc_wap3(df)
    df['wap4'] = calc_wap4(df)
    df['wap1_quarticity']=realized_quarticity(df['wap1'])
    df['wap1_reciprocal'] = reciprocal_transformation(df['wap1'])
    df['wap1_square_root'] = square_root_translation(df['wap1'])
    df['wap2_quarticity'] = realized_quarticity(df['wap2'])
    df['wap2_reciprocal'] = reciprocal_transformation(df['wap2'])
    df['wap2_square_root'] = square_root_translation(df['wap2'])
    df['wap3_quarticity']=realized_quarticity(df['wap3'])
    df['wap3_reciprocal'] = reciprocal_transformation(df['wap3'])
    df['wap3_square_root'] = square_root_translation(df['wap3'])
    df['wap4_quarticity'] = realized_quarticity(df['wap4'])
    df['wap4_reciprocal'] = reciprocal_transformation(df['wap4'])
    df['wap4_square_root'] = square_root_translation(df['wap4'])

    df['wap1_shift2'] = df['wap1'].shift(1) - df['wap1'].shift(2)
    df['wap1_shift5'] = df['wap1'].shift(1) - df['wap1'].shift(5)
    df['wap1_shift10'] = df['wap1'].shift(1) - df['wap1'].shift(10)

    df['wap2_shift15'] = df['wap2'].shift(1) - df['wap2'].shift(2)
    df['wap2_shift20'] = df['wap2'].shift(1) - df['wap2'].shift(5)
    df['wap2_shift30'] = df['wap2'].shift(1) - df['wap2'].shift(10)


    df['wap3_shift2'] = df['wap3'].shift(1) - df['wap3'].shift(2)
    df['wap3_shift5'] = df['wap3'].shift(1) - df['wap3'].shift(5)
    df['wap3_shift10'] = df['wap3'].shift(1) - df['wap3'].shift(10)


    df['wap4_shift2'] = df['wap4'].shift(1) - df['wap4'].shift(2)
    df['wap4_shift5'] = df['wap4'].shift(1) - df['wap4'].shift(5)
    df['wap4_shift10'] = df['wap4'].shift(1) - df['wap4'].shift(10)


    df['mid_price1'] = (df['ask_price1']+df['bid_price1'])/2

    df['HR1'] = ((df['bid_price1']-df['bid_price1'].shift(1))-(df['ask_price1']-df['ask_price1'].shift(1)))/((df['bid_price1']-df['bid_price1'].shift(1))+(df['ask_price1']-df['ask_price1'].shift(1)))

    df['pre_vtA'] = np.where(df.ask_price1==df.ask_price1.shift(1),df.ask_size1-df.ask_size1.shift(1),0)
    df['vtA'] = np.where(df.ask_price1>df.ask_price1.shift(1),df.ask_size1,df.pre_vtA)
    df['pre_vtB'] = np.where(df.bid_price1==df.bid_price1.shift(1),df.bid_size1-df.bid_size1.shift(1),0)
    df['vtB'] = np.where(df.bid_price1>df.bid_price1.shift(1),df.bid_size1,df.pre_vtB)

    df['Oiab'] = df['vtB']-df['vtA']

    df['bid_ask_size1_minus'] = df['bid_size1']-df['ask_size1']
    df['bid_ask_size1_plus'] = df['bid_size1']+df['ask_size1']
    df['bid_ask_size2_minus'] = df['bid_size2'] - df['ask_size2']
    df['bid_ask_size2_plus'] = df['bid_size2'] + df['ask_size2']
    df['bid_ask_size3_minus'] = df['bid_size3'] - df['ask_size3']
    df['bid_ask_size3_plus'] = df['bid_size3'] + df['ask_size3']
    df['bid_ask_size4_minus'] = df['bid_size4'] - df['ask_size4']
    df['bid_ask_size4_plus'] = df['bid_size4'] + df['ask_size4']

    df['bid_size1_shift'] = df['bid_size1']-df['bid_size1'].shift()
    df['ask_size1_shift'] = df['ask_size1']-df['ask_size1'].shift()
    df['bid_size2_shift'] = df['bid_size2'] - df['bid_size2'].shift()
    df['ask_size2_shift'] = df['ask_size2'] - df['ask_size2'].shift()
    df['bid_size3_shift'] = df['bid_size3'] - df['bid_size3'].shift()
    df['ask_size3_shift'] = df['ask_size3'] - df['ask_size3'].shift()

    df['bid_ask_size1_spread'] = df['bid_ask_size1_minus']/df['bid_ask_size1_plus']
    df['bid_ask_size2_spread'] = df['bid_ask_size2_minus'] / df['bid_ask_size2_plus']
    df['bid_ask_size3_spread'] = df['bid_ask_size3_minus'] / df['bid_ask_size3_plus']
    df['bid_ask_size4_spread'] = df['bid_ask_size4_minus'] / df['bid_ask_size4_plus']

    # Calculate log returns


    df['roliing_mid_price1_mean'] = df['mid_price1'].rolling(rolling).mean()
    df['rolling_mid_price1_std'] = df['mid_price1'].rolling(rolling).std()

    df['rolling_HR1_mean'] = df['HR1'].rolling(rolling).mean()

    df['rolling_bid_ask_size1_minus_mean1'] = df['bid_ask_size1_minus'].rolling(rolling).mean()
    df['rolling_bid_ask_size2_minus_mean1'] = df['bid_ask_size2_minus'].rolling(rolling).mean()
    df['rolling_bid_ask_size3_minus_mean1'] = df['bid_ask_size3_minus'].rolling(rolling).mean()


    df['rolling_bid_size1_shift_mean1'] = df['bid_size1_shift'].rolling(rolling).mean()
    df['rolling_bid_size1_shift_mean3'] = df['bid_size1_shift'].rolling(5*rolling).mean()
    df['rolling_bid_size1_shift_mean5'] = df['bid_size1_shift'].rolling(10*rolling).mean()
    df['rolling_ask_size1_shift_mean1'] = df['ask_size1_shift'].rolling(rolling).mean()
    df['rolling_ask_size1_shift_mean3'] = df['ask_size1_shift'].rolling(5*rolling).mean()
    df['rolling_ask_size1_shift_mean5'] = df['ask_size1_shift'].rolling(10*rolling).mean()
    df['rolling_bid_ask_size1_spread_mean2'] = df['bid_ask_size1_spread'].rolling(rolling).mean()
    df['rolling_bid_ask_size1_spread_mean3'] = df['bid_ask_size1_spread'].rolling(5*rolling).mean()
    df['rolling_bid_ask_size1_spread_mean5'] = df['bid_ask_size1_spread'].rolling(10*rolling).mean()


    df['log_return1'] = np.log(df['wap1'].shift(1)/df['wap1'].shift(2))*100
    df['log_return2'] = np.log(df['wap2'].shift(1) / df['wap2'].shift(2)) * 100
    df['log_return3'] = np.log(df['wap3'].shift(1) / df['wap3'].shift(2)) * 100
    df['log_return4'] = np.log(df['wap4'].shift(1)/df['wap4'].shift(2))*100


    df['log_return_wap1_shift5'] = np.log(df['wap1'].shift(1)/df['wap1'].shift(5))*100
    df['log_return_wap2_shift5'] = np.log(df['wap2'].shift(1) / df['wap2'].shift(5)) * 100
    df['log_return_wap3_shift5'] = np.log(df['wap3'].shift(1) / df['wap3'].shift(5)) * 100
    df['log_return_wap4_shift5'] = np.log(df['wap4'].shift(1) / df['wap4'].shift(5)) * 100

    df['log_return_wap1_shift15'] = np.log(df['wap1'].shift(1)/df['wap1'].shift(15))*100
    df['log_return_wap2_shift15'] = np.log(df['wap2'].shift(1) / df['wap2'].shift(15)) * 100
    df['log_return_wap3_shift15'] = np.log(df['wap3'].shift(1) / df['wap3'].shift(15)) * 100
    df['log_return_wap4_shift15'] = np.log(df['wap4'].shift(1) / df['wap4'].shift(15)) * 100


    df['ewm_wap1_mean'] = pd.DataFrame.ewm(df['wap1'],span=rolling).mean()
    df['ewm_wap2_mean'] = pd.DataFrame.ewm(df['wap2'], span=rolling).mean()
    df['ewm_wap3_mean'] = pd.DataFrame.ewm(df['wap3'], span=rolling).mean()
    df['ewm_wap4_mean'] = pd.DataFrame.ewm(df['wap4'], span=rolling).mean()


    df['rolling_mean1'] = df['wap1'].rolling(rolling).mean()
    df['rolling_std1'] = df['wap1'].rolling(rolling).std()
    df['rolling_min1'] = df['wap1'].rolling(rolling).min()
    df['rolling_max1'] = df['wap1'].rolling(rolling).max()
    df['rolling_skew1'] = df['wap1'].rolling(rolling).skew()
    df['rolling_kurt1'] = df['wap1'].rolling(rolling).kurt()
    df['rolling_quantile1_25'] = df['wap1'].rolling(rolling).quantile(.25)
    df['rolling_quantile1_75'] = df['wap1'].rolling(rolling).quantile(.75)

    df['rolling_mean2'] = df['wap2'].rolling(rolling).mean()
    df['rolling_std2'] = df['wap2'].rolling(rolling).std()
    df['rolling_min2'] = df['wap2'].rolling(rolling).min()
    df['rolling_max2'] = df['wap2'].rolling(rolling).max()
    df['rolling_skew2'] = df['wap2'].rolling(rolling).skew()
    df['rolling_kurt2'] = df['wap2'].rolling(rolling).kurt()
    df['rolling_quantile2_25'] = df['wap2'].rolling(rolling).quantile(.25)
    df['rolling_quantile2_75'] = df['wap2'].rolling(rolling).quantile(.75)


    df['rolling_mean3'] = df['wap3'].rolling(rolling).mean()
    df['rolling_var3'] = df['wap3'].rolling(rolling).var()
    df['rolling_min3'] = df['wap3'].rolling(rolling).min()
    df['rolling_max3'] = df['wap3'].rolling(rolling).max()
    df['rolling_skew3'] = df['wap3'].rolling(rolling).skew()
    df['rolling_kurt3'] = df['wap3'].rolling(rolling).kurt()
    df['rolling_median3'] = df['wap3'].rolling(rolling).median()
    df['rolling_quantile3_25'] = df['wap3'].rolling(rolling).quantile(.25)
    df['rolling_quantile3_75'] = df['wap3'].rolling(rolling).quantile(.75)


    df['rolling_mean4'] = df['wap4'].rolling(rolling).mean()
    df['rolling_std4'] = df['wap4'].rolling(rolling).std()
    df['rolling_min4'] = df['wap4'].rolling(rolling).min()
    df['rolling_max4'] = df['wap4'].rolling(rolling).max()
    df['rolling_skew4'] = df['wap4'].rolling(rolling).skew()
    df['rolling_kurt4'] = df['wap4'].rolling(rolling).kurt()
    df['rolling_median4'] = df['wap4'].rolling(rolling).median()
    df['rolling_quantile4_25'] = df['wap4'].rolling(rolling).quantile(.25)
    df['rolling_quantile4_75'] = df['wap4'].rolling(rolling).quantile(.75)


    df['wap_balance1'] = abs(df['wap1'] - df['wap2'])
    df['wap_balance2'] = abs(df['wap1'] - df['wap3'])
    df['wap_balance3'] = abs(df['wap2'] - df['wap3'])
    df['wap_balance4'] = abs(df['wap3'] - df['wap4'])

    df['price_spread1'] = (df['ask_price1'] - df['bid_price1']) / ((df['ask_price1'] + df['bid_price1']) / 2)
    df['price_spread2'] = (df['ask_price2'] - df['bid_price2']) / ((df['ask_price2'] + df['bid_price2']) / 2)
    df['price_spread3'] = (df['ask_price3'] - df['bid_price3']) / ((df['ask_price3'] + df['bid_price3']) / 2)
    df['price_spread4'] = (df['ask_price4'] - df['bid_price4']) / ((df['ask_price4'] + df['bid_price4']) / 2)

    print(df.columns)
    return df

df1 = book_preprocessor(data)
df1.to_csv('book_preprocessor_2021_2022.csv')
#%%
data_orderflow_2021 = pd.read_csv('ETH_USDT-2021-08_2022-05-trades.csv')
data_orderflow_2021['timestamp'] = pd.to_datetime(data_orderflow_2021['timestamp'],unit='s')
col = ['datetime','id','last_price','size','side']
data_orderflow_2021.columns = col
data_orderflow_2021 = data_orderflow_2021.drop(['id','side'],axis=1)
data_orderflow_2021 = data_orderflow_2021.set_index('datetime').resample('1min').apply({'last_price':'last','size':'last'})
# data_orderflow_2021 = data_orderflow_2021.set_index('datetime').resample('1min').apply(np.mean)
#%%
def trade_preprocessor(data):
    df = data
    # df['log_return'] = np.log(df['last_price']).shift()

    df['amount'] = df['last_price'] * df['size']

    rolling = 60

    df['mid_price'] = np.where(df.size > 0, (df.amount - df.amount.shift(1)) / df.size, df.last_price)
    df['rolling_mid_price_mean'] = df['mid_price'].rolling(rolling).mean()
    df['rolling_mid_price_std'] = df['mid_price'].rolling(rolling).std()

    df['last_price_shift1'] = df['last_price'].shift(1) - df['last_price'].shift(1)
    df['last_price_shift2'] = df['last_price'].shift(1) - df['last_price'].shift(2)
    df['last_price_shift5'] = df['last_price'].shift(1) - df['last_price'].shift(5)
    df['last_price_shift10'] = df['last_price'].shift(1) - df['last_price'].shift(10)


    df['log_return_last_price_shift1'] = np.log(df['last_price'].shift(1) / df['last_price'].shift(1)) * 100
    df['log_return_last_price_shift2'] = np.log(df['last_price'].shift(1) / df['last_price'].shift(2)) * 100
    df['log_return_last_price_shift5'] = np.log(df['last_price'].shift(1) / df['last_price'].shift(5)) * 100
    df['log_return_last_price_shift10'] = np.log(df['last_price'].shift(1) / df['last_price'].shift(10)) * 100


    df['rolling_mean_size'] = df['size'].rolling(rolling).mean()
    df['rolling_var_size'] = df['size'].rolling(rolling).var()
    df['rolling_std_size'] = df['size'].rolling(rolling).std()
    df['rolling_sum_size'] = df['size'].rolling(rolling).sum()
    df['rolling_min_size'] = df['size'].rolling(rolling).min()
    df['rolling_max_size'] = df['size'].rolling(rolling).max()
    df['rolling_skew_size'] = df['size'].rolling(rolling).skew()
    df['rolling_kurt_size'] = df['size'].rolling(rolling).kurt()
    df['rolling_median_size'] = df['size'].rolling(rolling).median()

    df['ewm_mean_size'] = pd.DataFrame.ewm(df['size'], span=rolling).mean()
    df['ewm_std_size'] = pd.DataFrame.ewm(df['size'], span=rolling).std()

    df['size_percentile_25'] = df['size'].rolling(rolling).quantile(.25)
    df['size_percentile_75'] = df['size'].rolling(rolling).quantile(.75)
    df['size_percentile'] = df['size_percentile_75'] - df['size_percentile_25']


    df['price_percentile_25'] = df['last_price'].rolling(rolling).quantile(.25)
    df['price_percentile_75'] = df['last_price'].rolling(rolling).quantile(.75)
    df['price_percentile'] = df['price_percentile_75'] - df['price_percentile_25']



    df['rolling_mean_amount'] = df['amount'].rolling(rolling).mean()
    df['rolling_quantile_25_amount'] = df['amount'].rolling(rolling).quantile(.25)
    df['rolling_quantile_50_amount'] = df['amount'].rolling(rolling).quantile(.50)
    df['rolling_quantile_75_amount'] = df['amount'].rolling(rolling).quantile(.75)


    df['ewm_mean_amount'] = pd.DataFrame.ewm(df['amount'], span=rolling).mean()

    print (df.columns)
    return df

df1 = trade_preprocessor(data_orderflow_220201)
# df1.to_csv('trade_preprocessor_2021_new.csv')
#%%
data = pd.read_csv('trade_preprocessor_2021.csv')