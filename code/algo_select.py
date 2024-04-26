import pandas as pd
import numpy as np
import os
# import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

import time

def run_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Execution time of '{func.__name__}': {end_time - start_time} seconds")
        return result
    return wrapper

def get_index(index_name = '^GSPC', index_folder = '..\data'):
   
    df = pd.read_csv(f'{index_folder}\{index_name}.csv', parse_dates=['formatted_date'], na_values='None')
    df.sort_values(by=['formatted_date'], inplace=True)
    df.drop_duplicates(subset=['formatted_date'], inplace=True)
    
    df['mkt_return'] = df['close'] / df['close'].shift() - 1
    df.loc[0,'mkt_return'] = df.loc[0,'close']/df.loc[0,'open']-1
    df_index = df[['formatted_date', 'mkt_return']]
    
    return df_index

def get_rf(rf_name = 'DGS1', rf_folder = '..\data'):
    
    df = pd.read_csv(f'{rf_folder}\{rf_name}.csv', parse_dates=['DATE'])
    df.rename(columns = {'DATE':'formatted_date', f'{rf_name}':'rf_rate'}, inplace=True)
    df.sort_values(by=['formatted_date'], inplace=True)
    df.drop_duplicates(subset=['formatted_date'], inplace=True)
    
    df['rf_rate'] = df['rf_rate'].replace('.', np.nan)
    df['rf_rate'] = df['rf_rate'].ffill()
    df['rf_rate'] = df['rf_rate'].astype(float)
    df['rf_return'] = df['rf_rate'].apply(lambda x: (1+x/100) ** (1/252)-1)
    df_rf = df[['formatted_date','rf_return']]
    
    return df_rf

def get_stock(ticker, stock_folder = '..\data\selection'):
    
    df = pd.read_csv(f'{stock_folder}\{ticker}.csv', parse_dates=['formatted_date'], na_values='None')
    df.sort_values(by=['formatted_date'], inplace=True)
    df.drop_duplicates(subset=['formatted_date'], inplace=True)
    
    df_stock = df[['formatted_date','high','low','open','close','volume',
                   'adjclose', 'dividends','ticker','sector', ]]
    
    return df_stock


def get_merge(df_index, df_rf, df_stock):
    
    # handle stock missing value
    fill_list = ['high','low','open','close']
    df_stock[fill_list] = df_stock[fill_list].replace(0, np.nan)
    df_stock.replace('None', np.nan, inplace=True)
        
    # merge index and stock
    df = pd.merge(df_index, df_stock, on='formatted_date', how='left', sort=True, indicator=True)
    
    # remove pre ipo data
    df['ticker'] = df['ticker'].ffill()
    df = df[df['ticker'].notnull()]    
    
    # handling missing data
    df['trading'] = np.where(df['_merge'] == 'both', 1, 0)
    df['adjclose'].ffill(inplace=True)
    df['close'].ffill(inplace=True)
    
    fill_dict = {c: df['close'] for c in fill_list}
    df.fillna(value=fill_dict, inplace=True)
    
    fill_0_columns = ['volume']
    df.fillna(value={c: 0 for c in fill_0_columns}, inplace=True)
    
    #df.ffill(inplace=True) 
        
    df.drop(columns=['_merge'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    # append rf
    df = pd.merge(df, df_rf[['formatted_date','rf_return']], on='formatted_date', how='left')
    
    # calculate more data
    df['h2l_range'] =  df['high'] - df['low']
    df['o2c_range'] = df['close'] - df['open']
    df['range'] = df['high'] / df['low'] - 1
    df['o2c_return'] = df['close'] / df['open'] - 1
    df['c2c_return'] = (df['close'] + df['dividends'])/ df['close'].shift() - 1
    df['mkt-rf'] = df['mkt_return'] - df['rf_return']
    
    df_merge = df
    
    return df_merge


def get_factor(ticker, stock_folder = '..\data\yf\selection', 
               index_name = '^GSPC', index_folder = '..\data\yf\index',
               rf_name = 'DGS1', rf_folder = '..\data\FRED',
               start_date = '2004-01-01', end_date='2024-04-01'):
    
    df_stock = get_stock(ticker=ticker, stock_folder=stock_folder)
    df_index = get_index(index_name=index_name, index_folder=index_folder)
    df_rf = get_rf(rf_name=rf_name, rf_folder=rf_folder)
        
    df = get_merge(df_index, df_rf, df_stock)
    
    ## truncate date
    df = df.loc[df['formatted_date'] > pd.to_datetime(start_date)]
    df = df.loc[df['formatted_date'] < pd.to_datetime(end_date)]
    
    # construct factor
    dict_factor = {}

    df['price_volume'] = df['close']*df['volume']
    dict_factor['price_volume'] = 'sum'

    df['price_volume_pct'] = df['price_volume'].pct_change()
    dict_factor['price_volume_pct'] = lambda x: list(x)

    list_ma = [5, 10, 20]#,40,63,126,252]
    
    for n in list_ma: 
        
        df[f'price_volume_ma_{n}'] = df['price_volume'].rolling(n,min_periods=1).mean()
        dict_factor[f'price_volume_ma_{n}'] = 'last'
        
        df[f'volume_std_{n}'] = df['volume'].rolling(n,min_periods=1).std()
        dict_factor[f'volume_std_{n}'] = 'last'
        
        df[f'price_std_{n}'] = df['close'].rolling(n,min_periods=1).std()
        dict_factor[f'price_std_{n}'] = 'last'
        
        df[f'ma_{n}'] = df['close'].rolling(n,min_periods=1).mean()
        dict_factor[f'ma_{n}'] = 'last' 
    
        df[f'bias_{n}'] = df['close'] / df[f'ma_{n}'] - 1
        dict_factor[f'bias_{n}'] = 'last'      

        df[f'range_{n}'] = df['high'].rolling(n, min_periods=1).max() / df['low'].rolling(n, min_periods=1).min()-1
        dict_factor[f'range_{n}'] = 'last'
        
        df[f'c2c_std_{n}'] = df['c2c_return'].rolling(n,min_periods=1).std()
        dict_factor[f'c2c_std_{n}'] = 'last'
        
        df[f'period_c2c_{n}'] = (df['c2c_return']+1).rolling(n, min_periods=1).apply(lambda x: x.prod())
        dict_factor[f'period_c2c_{n}'] = 'last'

        df[f'price_volume_corr_{n}'] = df['close'].rolling(n,min_periods=1).corr(df['volume'])
        dict_factor[f'price_volume_corr_{n}'] = 'last'
        
        df[f'return_volume_corr_{n}'] = df['c2c_return'].rolling(n,min_periods=1).corr(df['price_volume_pct'])
        dict_factor[f'return_volume_corr_{n}'] = 'last'

    day_range = df['range'].values
    df[f'ideal_range_{n}'] = df['adjclose'].rolling(n,min_periods=1).apply(ideal_range_func, args=(day_range,n,0.25),raw=False)
    dict_factor[f'ideal_range_{n}'] = 'last'
    
    df[f'mkt_corr_{n}'] = (df['c2c_return']).rolling(n,min_periods=1).corr(df['mkt-rf'])
    dict_factor[f'mkt_corr_{n}'] = 'last'
    
    df[f'mkt-rf_std_{n}'] = df['mkt-rf'].rolling(n,min_periods=1).std()
    dict_factor[f'mkt-rf_std_{n}'] = 'last'
    
    df[f'mkt_beta_{n}'] = df[f'mkt_corr_{n}']*df[f'c2c_std_{n}']/df[f'mkt-rf_std_{n}']
    dict_factor[f'mkt_beta_{n}'] = 'last'
    
    
    
        
    df_factor = df
        
    return df_factor, dict_factor
 
def ideal_range_func(x, y, n, p):
    '''
    :param x: Series, adjclose
    :param y: ndarray, range
    :param n: int, rolling window
    :param p: float, percentile
    :return: float, range factor value
    '''
    s = int(n*p)
    v = y[x.sort_values().index.to_list()]
    v_low = v[:s].mean()
    v_high = v[-s:].mean()
    return v_high - v_low


def get_period(df_factor, dict_factor, freq='M'):
    
    df = df_factor.set_index('formatted_date', drop=False)
        
    agg_dict = {'formatted_date': 'last',
                'high': 'max',
                'low': 'min',
                'open': 'first',
                'close': 'last',
                'volume':'sum',
                'adjclose': 'last',
                'dividends': 'sum', # already accounted when calculate c2c return
                'ticker': 'last',
                'trading': 'last',
                'sector': 'last',
                
                'o2c_return': 'first',
                'c2c_return': lambda x: list(x),
                'mkt_return': lambda x: list(x),
                'rf_return': lambda x: list(x),
                
                **dict_factor              
                }

    df_period = df.resample(rule=freq).agg(agg_dict)
 
    # find returns in each period
    df_period['daily_returns'] = df_period.apply(lambda x: [x['o2c_return']] + x['c2c_return'][1:], axis=1)
    df_period['next_period_daily_returns'] = df_period['daily_returns'].shift(-1)
    df_period['daily_mkt_returns'] = df_period['mkt_return']
    df_period['daily_rf_returns'] = df_period['rf_return']
    
    df_period = df_period.reset_index(drop=True)
    
    df_period = df_period.drop(columns=['o2c_return', 'c2c_return','mkt_return','rf_return'])
    
    # drop unuseful period
    # df_period.drop([0], axis=0, inplace=True) # drop first period
    
    df_period = df_period[df_period['trading'] != 0] # drop if not trading
    #df_period = df_period[df_period['formatted_date'] > pd.to_datetime('2013')] # drop before 2013
    
    return df_period

@run_time
def get_pool(stock_folder='..\data\yf\selection', 
             index_name='^GSPC', index_folder='..\data\yf\index',
             rf_name = 'DGS1', rf_folder='..\data\FRED',
             start_date = '2004-01-01', end_date='2024-04-01', freq='M'):
    
    # find ticker list
    tickers = os.listdir(stock_folder)
    
    #loop through list
    pbar = tqdm(tickers, unit="ticker")
    list_df = []
    
    for ticker in pbar:
        if ticker.endswith('csv'):
            ticker=ticker.split('.')[0]            
               
            df_factor, dict_factor = get_factor(ticker, stock_folder=stock_folder, 
                                                index_name=index_name, index_folder=index_folder,
                                                rf_name=rf_name, rf_folder=rf_folder,
                                                start_date = start_date, end_date=end_date)
            df_period = get_period(df_factor, dict_factor, freq=freq)

            list_df.append(df_period)
            
            pbar.set_description(f"Processed {ticker}\t")
            

    df_pool = pd.concat(list_df, ignore_index=True)
    df_pool.sort_values(['formatted_date', 'ticker'], inplace=True)
    df_pool.reset_index(inplace=True, drop=True)
    
    return df_pool, dict_factor

@run_time
def get_select(df_pool, select_num = 10,
                       factor_list = ['price_volume'],                         
                       ascending = [False], 
                       weights = [0.1] * 10,
                       c_rate = 0.01, # 1% of total trade
                       simple = True,
                       window_width = 20):
    
    df=df_pool.copy()
    
    # it's possible that loaded df_pool file may read 'next_period_daily_returns'
    # as list of str rather than float, need to run the next line  
    #df['next_period_daily_returns'] = df['next_period_daily_returns'].apply(lambda x: json.loads(x))
    
    # drop last period where value is na
    df.dropna(subset=['next_period_daily_returns'], inplace=True)
    
    df['next_period_return_curve'] = df['next_period_daily_returns'].apply(lambda x: np.cumprod(np.array(x, ndmin=1)+1))
    df['next_period_total_return'] = df['next_period_return_curve'].apply(lambda x: x[-1] - 1)
    
    # store curve
    df_select = pd.DataFrame()
    
    if simple:
        # rank by sum of factor rank
        df['rank_sum'] = 0
        for fac, asc in zip(factor_list, ascending):
            df['rank_sum'] += df.groupby('formatted_date')[fac].rank(ascending=asc)
        
        df['rank'] = df.groupby('formatted_date')['rank_sum'].rank(ascending=True)
  
    else:
        df = multi_factor_regression(df, factor_list, window_width)

        df['rank'] = df.groupby('formatted_date')['factor'].rank(ascending=False)
    
    # select top stocks
    df = df[df['rank'] <= select_num]
    
    # calculate return
    
    
    # equal weighted
    #df_select['next_period_equity_curve_bc'] = df.loc[:,'next_period_return_curve'].groupby(df['formatted_date']).apply(lambda x: np.mean(x, axis=0))

    # unequal weighted
    df_select['next_period_portfolio_curve_bc'] = df['next_period_return_curve'].groupby(df['formatted_date']).apply(lambda x: np.average(x, weights=weights, axis=0))

    # transaction cost
    df_select['next_period_portfolio_curve'] = df_select['next_period_portfolio_curve_bc'] * (1 - c_rate)  
    df_select['next_period_portfolio_curve'] = df_select['next_period_portfolio_curve'].apply(lambda x: list(x[:-1]) + [x[-1] * (1 - c_rate)]) 
    
    # portfolio return
    df_select['next_period_portfolio_daily_returns'] = df_select['next_period_portfolio_curve'].apply(lambda x: list(pd.Series([1]+x).pct_change()[1:].values))
    df_select['next_period_portfolio_total_return'] = df_select['next_period_portfolio_curve'].apply(lambda x: x[-1] - 1)
    
    # selected stock
    df_select['selected_stocks'] = df['ticker'].groupby(df['formatted_date']).apply(lambda x: ','.join(x.values))
    
    
    return df_select

@run_time
def get_perform(df_index, df_rf):
       
    df = pd.merge(left=df_index, right=df_select, left_on=['formatted_date'],
                      right_index=True, how='left', sort=True)
    
    df_perform = df.copy()
    
    # shift to next period
    col = ['selected_stocks', 'next_period_portfolio_total_return', 'next_period_portfolio_daily_returns']
    new_col = ['holdings', 'period_portfolio_total_return', 'period_portfolio_daily_returns']
    df_perform[new_col] = df_perform[col].shift()
    
    # fill holdings in each period
    df_perform.loc[df_perform['holdings'].notnull(),'period_begin_date'] = df_perform['formatted_date']
    df_perform['period_begin_date'].ffill(inplace=True)
    df_perform['holdings'].ffill(inplace=True)
    df_perform['selected_stocks'].ffill(inplace=True)
    df_perform.dropna(subset=['selected_stocks'], inplace=True)
    
    # fill in daily return
    
    df_perform['portfolio_return'] = df_perform[['period_portfolio_daily_returns']].groupby(df_perform['period_begin_date'], group_keys=False).apply(fill_period)
    df_perform['portfolio_return'].fillna(value=0, inplace=True)
    
    df_perform['portfolio_curve'] = (df_perform['portfolio_return'] + 1).cumprod()
    df_perform['mkt_curve'] = (df_perform['mkt_return'] + 1).cumprod()
    df_perform.set_index('formatted_date', inplace=True)
    
    return df_perform[['holdings','portfolio_curve','mkt_curve','mkt_return']]


def fill_period(df):
    df["return"]=df.iloc[0, 0]
    return df['return']


def multi_factor_regression(df, factor_list, window_width, y='next_period_total_return', macro=False):

    # set train window (start date and end date for regression to calculate factors before trading)
    trade_dates_pair = pd.DataFrame()
    trade_dates_idx = df[['formatted_date']].drop_duplicates().reset_index(drop=True)
    trade_dates_pair['end_date'] = trade_dates_idx['formatted_date'].iloc[1:]

    # -1 all periods before trading, else, fixed rolling window
    if window_width == -1:
        trade_dates_pair['start_date'] = trade_dates_idx['formatted_date'].min()
    else:
        trade_dates_pair['start_date'] = trade_dates_pair['end_date'].shift(window_width)
    
    trade_dates_pair.dropna(inplace=True)
    
    # drop nan
    mask = df.isin([np.inf, -np.inf])
    df[mask] = np.nan
    df = df.dropna(subset=factor_list, how='any')

    # standardize value
    if not macro:
        df.loc[:,factor_list] = df.groupby('formatted_date')[factor_list].rank(pct=True)

    # train for each window
    para_df = pd.DataFrame(columns=factor_list)  
    for end_date, start_date in trade_dates_pair.values:
        
        # setup dataset
        local_df = df[(df['formatted_date'] >= start_date) & (df['formatted_date'] < end_date)][factor_list + [y, 'formatted_date']].copy()
        
        #train
        regr = LinearRegression().fit(local_df[factor_list], local_df[y])

        # save
        i = len(para_df)
        para_df.loc[i, factor_list] = regr.coef_  
        para_df.loc[i, 'intercept'] = regr.intercept_ 
        
        # train end date is the start date for trading of next period    
        para_df.loc[i, 'formatted_date'] = end_date  
    
    # calculate factor value
    df = pd.merge(df, para_df, on='formatted_date', how='right', suffixes=['', '_coef'])
    df['factor'] = 0
    for factor in factor_list:
        df['factor'] += df[factor] * df[factor + '_coef']

    df['factor'] += df['intercept']
    df['factor'] = df['factor'].astype(float)  # 

    return df





if __name__ == '__main__':
    
    index_name = '^GSPC'
    rf_name = 'DGS1'
    selection_pool = 'SP500'
    
    start_date = '2004-01-01'
    end_date='2024-04-01'    
    
    factor_list = ['price_volume', 'bias_5',  
                   'ideal_range_20', 'mkt_beta_20']
    ascending = [True] # False:largest, True: smallest
    simple = True
    freq = 'M'
    window_width = -1
    select_num = 10
    weights = [0.1]*10
    c_rate = 0.001 # 0.1% of total trade
    
    
    # navigate to find data folders
    current_dir = os.getcwd()
    parent_dir = os.path.dirname(current_dir)
    stock_folder = os.path.join(parent_dir, f'data\yf\{selection_pool}') # change here
    index_folder = os.path.join(parent_dir, 'data\yf\index')
    rf_folder = os.path.join(parent_dir, 'data\FRED')
    
    
    # get index, rf data
    df_index = get_index(index_name=index_name, index_folder=index_folder)
    df_rf = get_rf(rf_name = rf_name, rf_folder = rf_folder)
    
    # testing one stock in pool only    
    '''
    ticker = 'AAPL'
    
    df, dict_factor = get_factor(ticker, index=index)
    df_period = get_period(df, dict_factor, freq=freq)
    '''    
    
    '''
    # stock pool to select from
    df_pool, dict_factor = get_pool(stock_folder=stock_folder, 
                              index_name=index_name, index_folder=index_folder,
                              rf_name=rf_name , rf_folder=rf_folder,
                              start_date=start_date, end_date=end_date, freq=freq)
  
    df_pool.to_pickle(f'df_pool_{selection_pool}_{freq}.pkl')
    '''
    
    
    df_pool = pd.read_pickle(f'df_pool_{selection_pool}_{freq}.pkl')
    
    df_select = get_select(df_pool, 
                            factor_list=factor_list, 
                            select_num = select_num,
                            ascending = ascending, 
                            weights = weights,
                            c_rate = c_rate,
                            simple=simple,
                            window_width=window_width)
    
    df_perform = get_perform(df_index, df_select)
    df_perform.to_csv(f'df_perform_{selection_pool}_{freq}_{select_num}_{simple}_{factor_list}.csv')
    
    
    # Plot
    
    fig = plt.figure(figsize=(16, 9))
    
    plt.plot(df_perform['portfolio_curve'], label='Portfolio')
    plt.plot(df_perform['mkt_curve'],label='Index')
    plt.legend(loc='best')
    plt.title(f'{selection_pool}_{freq}_{select_num}_{simple}_{factor_list}', fontsize=20)
    plt.savefig(f'{selection_pool}_{freq}_{select_num}_{simple}_{factor_list}.png')
    
    # prep for evaluation
    '''
    require the following columns: 'formatted_date', 'equity_curve'.
    optional columns: 'equity_curve_base', 'rf_return', 'mkt_return', 'signal','position'.
    '''   
    df_eva = df_perform[['portfolio_curve','mkt_return']]
    df_eva = pd.merge(left=df_eva, right=df_rf, left_on='formatted_date',
                      right_on='formatted_date',how='left')
    df_eva = df_eva.rename(columns = {'portfolio_curve':'equity_curve'})
    df_eva.to_csv(f'df_eva_{selection_pool}_{freq}_{select_num}_{simple}_{factor_list}.csv',index=False)
    
    print("\nend of code")
    
    
