import pandas as pd
from datetime import datetime, timedelta


def strategy_evaluate(df):
    '''
    Parameters
    ----------
    df : DataFrame of daily data
        require the following columns: 'formatted_date', 'equity_curve'.
        optional columns: 'equity_curve_base', 'rf_return', 'mkt_return', 'signal','position'.

    Returns
    -------
    metrics : Series
        Performance metricss.
    year_return : DataFrame
        Yearly return.
    '''
      
    # create new to save results
    results = {}
    df = df.copy()
    #df = df.rename(columns={'formatted_date': 'Date'})
    df.sort_values(by='formatted_date', inplace=True,  ascending=True)
    
    if 'rf_return' not in df.columns:
        df['rf_return']=0
        
    # Date
    start = df['formatted_date'].iloc[0]
    end = df['formatted_date'].iloc[-1]
    calendar_days = (end - start).days
    trading_days = len(df)
    
    results['Start'] = str(start.date())
    results['End'] = str(end.date())
    results['Calendar days'] = f'{calendar_days} days'
    results['Trading days'] = f'{trading_days} days'
    
    if 'position' in df.columns:
        exposure_days = (df['position']==1).sum()
        results['Exposure days'] = f'{exposure_days} days'
        results['Exposure %'] = f'{exposure_days/trading_days:.2%}'
    
   
    # Values and returns
    initial_value = df['equity_curve'].iloc[0]
    terminal_value = df['equity_curve'].iloc[-1]
    df['total_return'] = df['equity_curve']/initial_value - 1 
    
    results['Initial value'] = round(initial_value, 2)
    results['Terminal value'] = round(terminal_value, 2)
    results['Peak value'] = round(df['equity_curve'].max(), 2)
    
    total_return = df['total_return'].iloc[-1]   
    
    for year in [1, 2, 3, 5, 10]:
        if calendar_days > year*365:  
            since_year = end - timedelta(days=year*365)
            df_since = df[df['formatted_date'] >= since_year]['equity_curve']
            since_return = df_since.iloc[-1]/df_since.iloc[0] - 1 
            results[f'Last {year}-year return'] = f'{since_return :.2%}' 
            
    results['Return since inception'] = f'{total_return :.2%}'         

    # Average annual return and volatility
    annual_return_g = (total_return + 1) ** ('365 days 00:00:00' / (end - start)) - 1
    results['Annual return (geo)'] = f'{annual_return_g:.2%}'
    
    df['daily_return'] = df['equity_curve']/df['equity_curve'].shift() - 1
    ann_return = df['daily_return'].mean()*252
    ann_return_vol = df['daily_return'].std()*(252**0.5)
    
    results['Annual return'] = f"{ann_return:.2%}" 
    results['Annual volatility'] =  f"{ann_return_vol:.2%}"
    
    # Ratios
    rf_rate = df['rf_return'].apply(lambda x: ((1+x)**(252)-1)).mean()
    results['Risk free rate'] = f'{rf_rate:.2%}'
    
    ann_ex_return_vol = (df['daily_return'] - df['rf_return']).std()*(252**0.5)
    #results['Annual excess return vol.'] = f'{ann_ex_return_vol:.2%}'
    
    downside_returns = df['daily_return'][df['daily_return'] < 0]
    results['Sharpe ratio'] = (ann_return - rf_rate)/ann_ex_return_vol
    results['Sortino ratio'] = (ann_return - rf_rate)/(downside_returns.std()*(252**0.5))
        
    if 'mkt_return' in df.columns:
        df['mkt-rf'] = df['mkt_return'] - df['rf_return']
        df['mkt_curve'] = (1 + df['mkt_return']).cumprod()
        
        mkt_return = df['mkt_curve'].iloc[-1] - 1
        ann_mkt_return = df['mkt_return'].mean()*252
        ann_mkt_vol = df['mkt_return'].std()*(252**0.5)
        ann_mkt_ex_return_vol = df['mkt-rf'].std()*(252**0.5)
        beta = df['mkt-rf'].cov(df['daily_return'])/df['mkt-rf'].var() 
        alpha = ann_return - rf_rate - beta*ann_mkt_return
        
        results['Market return'] =  f"{mkt_return:.2%}"        
        results['Annual market return'] = f"{ann_mkt_return:.2%}"
        results['Annual market volatility'] = f"{ann_mkt_vol:.2%}"
        results['Market Sharpe ratio'] = (ann_mkt_return - rf_rate)/ann_mkt_ex_return_vol
        results['Beta'] = beta           
        results['Treynor ratio'] = (ann_return - rf_rate)/beta
        results['Alpha'] = f"{alpha:.2%}"


    # Max Drawdown
    #max_drawdown = (df['equity_curve']/df['equity_curve'].expanding(min_periods=0).max()).min() -1
    
    # find current historical max
    
    df.set_index('formatted_date', inplace=True)
    
    df['max2here'] = df['equity_curve'].expanding(min_periods=0).max()
    # find current drowdwon from historical max
    df['dd2here'] = df['equity_curve'] / df['max2here'] - 1
    # find start and end date of MDD
    mdd_end = df['dd2here'].idxmin()
    max_drawdown = df.loc[mdd_end, 'dd2here']
    mdd_start = df.loc[:mdd_end, 'equity_curve'].idxmax()
    mdd_length = (mdd_end - mdd_start).days
    # drop columns for mid steps
    df.drop(['max2here', 'dd2here'], axis=1, inplace=True)
    
    df.reset_index(inplace=True)
    
    # Max Drawdown
    results['Max drawdown'] = f'{max_drawdown:.2%}'
    results['MDD start date'] = str(mdd_start.date())
    results['MDD end date'] = str(mdd_end.date())
    results['MDD duration'] = f'{mdd_length} days'
    results['Calmar Ratio'] = ann_return/max_drawdown
    
    # Trade stats require signal and position data
    if 'signal' in df.columns:        
        num_trades = (df['signal'].iloc[:-1]==1).sum()        
        if num_trades !=0:            
            results['# of trades'] = num_trades
            
            if 'position' in df.columns:               
                df.loc[df.index[-1], 'position'] = 0 # make sure to liquidate in the last day 
                num_group = (df['position'] != df['position'].shift()).cumsum()
                grouped = df.groupby(num_group, group_keys=False)
               
                trade_records = []
                
                for _, frame in grouped:                    
                    if (frame["position"] == 1).all():                        
                        begin = df.at[frame.index[0]-1,'equity_curve']
                        over = df.at[frame.index[-1]+1,'equity_curve']
                        period_return = over/begin - 1 
                        
                        begin_date = df.at[frame.index[0],'formatted_date']
                        over_date = df.at[frame.index[-1]+1,'formatted_date']
                        period_length = (over_date - begin_date).days 
                        
                        trade_record = [period_return,period_length]
                        trade_records.append(trade_record)
                        
                df_trade = pd.DataFrame(trade_records,columns=['return','length'])
                
                best_trade = df_trade['return'].max()
                worst_trade = df_trade['return'].min()
                avg_return = df_trade['return'].mean()
                
                results['Best trade'] = f'{best_trade:.2%}'
                results['Worst trade'] = f'{worst_trade:.2%}'
                results['Average trade return'] = f'{avg_return:.2%}'
                
                longest_trade = df_trade['length'].max()
                shorst_trade = df_trade['length'].min()
                avg_length = int(df_trade['length'].mean())
                
                results['Longest trade'] = f'{longest_trade} days'
                results['Shorst trade'] = f'{shorst_trade} days'
                results['Average trade length'] = f'{avg_length} days'
                
                win_rate = len(df_trade[df_trade['return'] > 0])/len(df_trade)
                results['Win rate'] = f'{win_rate:.2%}'

    # Yearly Return, Monthly Return
    year_return = pd.DataFrame()
    month_return = pd.DataFrame()    
    
    df.set_index('formatted_date', inplace=True)
    year_return['return'] = df['equity_curve'].resample(rule='A').apply(lambda x: x.iloc[-1]/x.iloc[0] - 1)  
    month_return['return'] = df['equity_curve'].resample(rule='M').apply(lambda x: x.iloc[-1]/x.iloc[0] - 1)  
    df.reset_index(inplace=True)
    
    results['Max mth gain'] = format(month_return['return'].max(), '.2%')  
    results['Max mth loss'] = format(month_return['return'].min(), '.2%') 
    
    if 'equity_curve_base' in df.columns: 
              
        return_base = df['equity_curve_base'].iloc[-1]/df['equity_curve'].iloc[0] - 1
        results['Total base return'] = f'{return_base :.2%}'
        
        
        df['daily_return_base'] = df['equity_curve_base'].pct_change()
        ann_return_base = df['daily_return_base'].mean()*252
        ann_return_vol_base = df['daily_return_base'].std()*(252**0.5)
        
        results['Annual base return'] = f'{ann_return_base:.2%}'
        results['Annual base volatility'] = f'{ann_return_vol_base:.2%}'    
        
        tracking_error = (df['daily_return'] - df['daily_return_base']).std()*(252**0.5)
        IR = (ann_return - ann_return_base)/tracking_error
        
        results['Tracking error'] = f'{tracking_error:.2%}'
        results['Information Ratio'] = IR        
        
        df.set_index('formatted_date', inplace=True)
        year_return['return_base'] = df['equity_curve_base'].resample(rule='A').apply(lambda x: x.iloc[-1]/x.iloc[0] - 1)  
        month_return['return_base'] = df['equity_curve_base'].resample(rule='M').apply(lambda x: x.iloc[-1]/x.iloc[0] - 1)  
        df.reset_index(inplace=True)
        
        year_return['ex_return'] = year_return['return'] - year_return['return_base']
        month_return['ex_return'] = month_return['return'] - month_return['return_base']
        
        # Monthly stats
        results['#Mth outperform base'] = len(month_return.loc[month_return['ex_return'] > 0])
        results['#Mth underperform base'] = len(month_return.loc[month_return['ex_return'] <= 0])
        results['%Mth outperform base'] = format(results['#Mth outperform base'] / len(month_return), '.2%') 
    
        results['Max mth ex base gain'] = format(month_return['ex_return'].max(), '.2%')  
        results['Max mth ex base loss'] = format(month_return['ex_return'].min(), '.2%') 
        results['Avg Mth ex base return'] = format(month_return['ex_return'].mean(), '.2%')  
    
        # return/base return
        results['Return/Base return'] = total_return/return_base
        
    
    
    metrics = pd.Series(results)
    
    year_return.reset_index(inplace=True)
    year_return.rename(columns={'formatted_date': 'Year end'}, inplace=True)
    year_return.set_index('Year end',inplace=True)
    
    
    return metrics, year_return

if __name__ == '__main__':
    
    '''
    require the following columns: 'formatted_date', 'equity_curve'.
    optional columns: 'equity_curve_base', 'rf_return', 'mkt_return', 'signal','position'.
    '''
    
    selection_pool = 'SP500'
    freq = 'M'
    select_num = 10
    simple = True
    factor_list = ['price_volume', 'bias_5',  
                   'ideal_range_20', 'mkt_beta_20']
    
    
    
    df_eva=pd.read_csv(f'df_eva_{selection_pool}_{freq}_{select_num}_{simple}_{factor_list}.csv', parse_dates=['formatted_date'])
    
    
    metrics, year_return = strategy_evaluate(df_eva)
    
    print(metrics)
    print(year_return)
    
    print("\nend of code")

