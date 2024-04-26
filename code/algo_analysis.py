import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
#import statsmodels.formula.api as sml

#pd.set_option('display.max_columns', 100)

def mad_cut(x, inclusive=True):

    upper_limit = x.median() + abs(x - x.mean()).median() * 3
    lower_limit = x.median() - abs(x - x.mean()).median() * 3 
    
    if inclusive:
        x.loc[x < lower_limit] = lower_limit
        x.loc[x > upper_limit] = upper_limit
        x.clip(lower_limit, upper_limit, inplace=True)
    else:
        x.loc[x < lower_limit] = np.nan
        x.loc[x > upper_limit] = np.nan
    
    return x


def winsor(x, inclusive=True, tail=5):
    
    if x.dropna().empty:
        x = x.fillna(0)
    
    else:
        if inclusive:
            x.loc[x < np.percentile(x.dropna(), tail)] = np.percentile(x.dropna(), tail)
            x.loc[x > np.percentile(x.dropna(), 100 - tail)] = np.percentile(x.dropna(), 100 - tail)
        else:
            x.loc[x < np.percentile(x.dropna(), tail)] = np.nan
            x.loc[x > np.percentile(x.dropna(), 100 - tail)] = np.nan
   
    return x


def zscore(x):
    return (x - x.mean()) / x.std()


def factor_dist_plot(df, factor, trim=True):
    
    df_factor_dist = df.copy()
    df_factor_dist['year'] = df_factor_dist['formatted_date'].dt.year
    
    if trim:
        sub = df_factor_dist.groupby('formatted_date')        
        sub = sub[factor].apply(mad_cut)        
        sub = sub.reset_index()        
        sub.columns = [ 'formatted_date','Index', factor]        
        sub.set_index('Index', inplace=True)
        df_factor_dist[f'{factor}_cut'] = sub[factor]
        
    years = df_factor_dist.year.unique()
    n_year = len(years)    
    n_col = math.ceil(n_year/3)
    
    # Plotting
    fig, axes = plt.subplots(3, n_col, figsize=(16, 9), sharex=False, sharey=False)

    for i in range(n_year):
        year_data = df_factor_dist.loc[df_factor_dist.year == years[i],factor]
        sns.kdeplot(year_data.to_list(), fill=True, ax=axes[int(i/n_col), i%n_col]).set(title=f'{years[i]}') 
        #year_data.plot.kde(ax=axes[int(i/n_col), i%n_col], title=f'{years[i]}', xlim=(plot_data.min(),plot_data.max()))
    
    plt.suptitle(f'{factor} Factor Dist.',fontsize = 20 )
    plt.savefig(f'{factor} Factor Dist.png')
    
    return df_factor_dist


def auto_corr_plot(df, factor):
    
    data = df.copy()    
    data = data.set_index(['formatted_date','ticker'])    
        
    vcv = data.loc[:,factor].unstack().T.corr()
    auto_corr = pd.DataFrame(np.diag(vcv, 1), columns=[factor], index=vcv.index[1:])
    
    #Plotting       
    auto_corr.plot(legend=False, color='darkred', figsize=(10,5), xlabel='')
    plt.title(f'{factor} Auto Corr.', fontsize=20)
    plt.savefig(f'{factor} Auto Corr.png')
    
    return auto_corr


def ols_resid(y, x):
    
    df = pd.concat([y, x], axis=1)

    if df.dropna().shape[0] > 0:
        resid = sm.OLS(y, x, missing='drop').fit().resid
        y_ = resid.reindex(df.index)
        
    return y_


def scale(df, neutral=True, trim=True, normal=True):
    
    if df.empty:
        data = df
        
    else:        
        data = df.copy()
        
        x = pd.DataFrame()
        
        if 'sector' in data.columns:
            sector = data['sector']
            data = data.drop(['sector'], axis=1)
            x = pd.get_dummies(sector, columns=['sector'], prefix='sector', 
                               prefix_sep="_", dummy_na=False, drop_first=True,
                               dtype=int)
            
        if 'TMC' in data.columns:
            tmc = data['TMC']
            data = data.drop(['TMC'], axis=1)
            x['TMC'] = np.log(tmc)        
        
        # neutralize for sector and market cap
        if neutral:
            x['Intercept'] = 1
            data = data.apply(func=ols_resid, x=x, axis=0)      
            
        # trim extreme values
        if trim:
            data = data.apply(lambda w:winsor(w),axis = 0)
            
        # normalize
        if normal:
            data = zscore(data)

        # reindex
        data = data.reindex(df.index)
        
    return data


def scale_factors(df, trim=True, neutral=True, normal=True):
    
    flist = []

    dates = df['formatted_date'].unique()

    for date in dates:  # dateuse = dates[0]
        data = df.loc[df['formatted_date'] == date]
        ticker = data[['formatted_date', 'ticker']]
        
        df_scale = scale(data.drop(['formatted_date', 'ticker'], axis=1), 
                        trim=trim, neutral=neutral, normal=normal)
        
        flist.append(pd.concat([ticker, df_scale], axis=1))

    df = pd.concat(flist, axis=0)
    df = df.sort_values(by=['formatted_date', 'ticker'])
    df.reset_index(drop=True)
    
    return df
        

def ic_icir_plot(df, factor, method = 'pearson'):
   
    ic_all = df.groupby('formatted_date').apply(lambda x: x.corr(method=method, numeric_only=True)['next_period_return']).reset_index()
    ic_all = ic_all.dropna().drop(['next_period_return'], axis=1).set_index('formatted_date')
    
    f_ic = ic_all[factor].mean()
    f_icir = ic_all[factor].mean() / ic_all[factor].std() * np.sqrt(12)
    
    # Plotting
    fig = plt.figure(figsize=(16, 9))
    
    ax = plt.axes()
    xtick = np.arange(0, ic_all.shape[0], 12)
    xticklabel = pd.Series(ic_all.index[xtick])
    ax.bar(np.arange(ic_all.shape[0]), ic_all[factor], color='darkred')
    
    ax1 = plt.twinx()
    ax1.plot(np.arange(ic_all.shape[0]), ic_all.cumsum(), color='orange', label='Cumulative')
    
    ax.set_xticks(xtick)
    ax.set_xticklabels(xticklabel, rotation=30, fontsize='small')
    
    plt.legend()
    plt.title(factor + ' IC = {}, ICIR = {}'.format(round(f_ic, 4), round(f_icir, 4)), fontsize=20)
    plt.savefig(f'{factor} IC ICIR.png')
    
    return f_ic, f_icir
    

def ic_decay_plot(df, factor, method='pearson'):
    
    ic_decay = []
    
    for lag in range(12):
        
        ret = df.pivot(index='formatted_date', columns='ticker', values='next_period_return').shift(-lag).stack().reset_index()
        ret = ret.rename(columns={ret.columns[-1]: 'next_period_return_lag'})
        
        df_lag = pd.merge(df,ret,on=['formatted_date','ticker'])
        df_lag.drop('next_period_return', axis=1, inplace=True)
        
        ic_all = df_lag.groupby('formatted_date').apply(lambda x: x.corr(method=method, numeric_only=True)['next_period_return_lag']).reset_index()
        ic_all = ic_all.dropna().drop(['next_period_return_lag'], axis=1).set_index('formatted_date')
        
        ic_decay.append(pd.DataFrame(ic_all.mean(), columns=[f'lag_{lag}']))
    
    ic_decay = pd.concat(ic_decay, axis=1)
    ic_decay = ic_decay.iloc[0]

    ic_half = half_func(ic_decay)
    
    # Plotting    
    fig = plt.figure(figsize=(10, 5))
    ic_decay.plot(kind='bar')
    #ic_decay.plot(kind='line', color='darkred', linewidth=2)
    
    plt.title(f'{factor} IC decay, half life: {ic_half}', fontsize=20)
    plt.savefig(f'{factor} IC decay.png')

    return ic_decay, ic_half


def half_func(x):
    target = abs(x.iloc[0] / 2)
    position = np.where(x.abs() < target)
    if len(position[0]) > 0:
        message = f'{position[0][0]}'
    else:
        message = f'>{len(x)}'
    return message

def group_ic_plot(df, factor, groups = 5, method = 'spearman'):

    factor_data = df[['formatted_date', 'ticker', factor]]#.dropna(subset=['formatted_date', factor])
    return_data = df[['formatted_date', 'ticker','next_period_return']].sort_values(['formatted_date', 'ticker']).reset_index(drop=True)

    dates = factor_data['formatted_date'].unique()
    return_pivot = return_data.pivot(index='formatted_date', columns='ticker', values='next_period_return')
    
    group_ic = []
    
    for date in dates:
        
        fd = factor_data.loc[factor_data['formatted_date'] == date, factor_data.columns[1:]].set_index('ticker')[factor]
        ret = return_pivot.loc[date]

        ic = get_group_ic(fd, ret, method, groups)
        ic.insert(0, 'factor', factor)            
        group_ic.append(ic)
    
    df_group_ic = pd.concat(group_ic, axis=0)
    
    # Plotting group ic
    #fig = plt.figure(figsize=(10, 5))    
    df_group_ic[df_group_ic.columns[1:]].mean(axis=0).plot(kind='bar', figsize=(10, 5))
    
    plt.title(f'{factor} Group IC', fontsize=20)    
    plt.savefig(f'{factor} Group IC.png')
    
    # Plotting cumulative group ic
    (df_group_ic[df_group_ic.columns[1:]]).cumsum().plot(xlabel='', figsize=(10, 5))
    
    plt.title(f'{factor} Cumulative Group IC', fontsize=20)    
    plt.savefig(f'{factor} Cumulative Group IC.png')
    
    return df_group_ic


def get_group_ic(fd, ret, method, groups):
    
    rt = pd.concat([fd, ret], axis=1).dropna()
    indexs = ["startdate"] + list(range(groups))
    
    if rt.empty:
        result = pd.DataFrame([rt.columns[1]] + [0] * groups, index=indexs).T
    
    else:
        groupdata = pd.qcut(rt.iloc[:, 0], q=groups, labels=False, duplicates='drop')

        if groupdata.unique().shape[0] == groups:
            
            rt['group'] = groupdata
            
            IC = rt.groupby('group').apply(lambda x: x.corr(method=method).fillna(0).iloc[0, 1])

            result = pd.DataFrame([rt.columns[1]] + IC.tolist(), index=indexs).T

        else:
            result = pd.DataFrame([rt.columns[1]] + [0] * groups, index=indexs).T
   
    return result.set_index('startdate')


def sector_ic_plot(df, factor, method='spearman'):
    #requires a sector to have more than 1 stock to calculate correlation, otherwise, drop nan    
    df_corr = df.groupby(['formatted_date', 'sector']).apply(lambda x: x.corr(method=method, numeric_only=True)['next_period_return']).reset_index()
    df_corr = df_corr.drop(['next_period_return'], axis=1).dropna().reset_index(drop=True)
    
    ic_sec_mean = df_corr.groupby('sector')[factor].mean()
    icir_sec = ic_sec_mean/df_corr.groupby('sector')[factor].std()*np.sqrt(12)
   
    # Sort ic by sector
    ic_sec_sort = ic_sec_mean.sort_values()
    icir_sec_sort = icir_sec.loc[ic_sec_sort.index]    
    df_ic_sec = df_corr[['formatted_date','sector',factor]].pivot(index = 'formatted_date',columns = 'sector',values = factor)
    df_ic_sec = df_ic_sec.loc[:,ic_sec_sort.index]

    # Plotting ic icir
    plt.figure(figsize = (16,9)) 
    plt.barh(y=np.arange(ic_sec_sort.shape[0]), height=0.5, 
             tick_label=ic_sec_sort.index, width=ic_sec_sort, label='IC')
    plt.barh(y=np.arange(ic_sec_sort.shape[0])+0.3, height=0.5, 
             tick_label=icir_sec_sort.index, width=icir_sec_sort, label='ICIR')
    
    plt.tick_params(axis='y', labelsize=12)
    plt.legend(fontsize=12)
    
    plt.title(f'{factor} Sector IC, ICIR', fontsize = 20)
    plt.savefig(f'{factor} Sector IC.png', dpi=300)

    # Plotting cumulative ic
    df_ic_sec.cumsum().plot(figsize = (16,9), linewidth=2, cmap = 'rainbow', xlabel='')
    plt.legend(fontsize=12)#, bbox_to_anchor=(1.05, 1.05))
    
    plt.title(f'{factor} Sector Cumulative IC',fontsize = 20)
    plt.savefig(f'{factor} Sector Cumulative IC', dpi=300)
        
    return df_ic_sec, ic_sec_mean, icir_sec


def group_test_plot(df, factor, df_index, groups=5):
    
    d_ = 'formatted_date'
    t_ = 'ticker'
    r_ = 'next_period_return'
    f_ = factor
    
    df_index.reset_index(inplace=True)
    
    df = df[[d_, t_, r_, f_]]
    
    # assign group number
    df['groups'] = df.groupby(d_)[f_].transform(lambda x: pd.qcut(x, q=groups, labels=False))
    
    # equal weghting of stocks for each group
    df_g = df.groupby([d_, 'groups']).apply(lambda x: x[r_].mean())
    
    df_g = df_g.unstack().reset_index()
    
    # construct L-S portfolio, choose group to invest 
    if df_g.iloc[:, -1].mean() > df_g.iloc[:, -groups].mean():
        df_g['L-S'] = df_g.iloc[:, -1] - df_g.iloc[:, -groups]
        df_choose = df.loc[df.groups == (groups-1)]
    else:
        df_g['L-S'] = df_g.iloc[:, -groups] - df_g.iloc[:, -1]
        df_choose = df.loc[df.groups == 0]   
    
    # calculate choosen group turnover rate
    df_group_turnover = get_turnover(df_choose) 
    
    # move return to the right period
    df_g_ret = pd.merge(df_index[d_],df_g,on=d_,how='left')
    df_g_ret.set_index(d_, inplace=True)
    df_g_ret= df_g_ret.shift(periods=1)
    df_g_ret.reset_index(inplace=True)
    df_g_ret.fillna(0)
    # insert factor name
    df_g_ret.insert(0, 'factor', f_)      
    
    # calculate return curve for each group
    df_group_nav = df_g_ret.set_index(d_).iloc[:, 1:].apply(lambda x: (1 + x).cumprod())
    
    # df_index = df_index.set_index(d_)
    
    # calculate return curve for index
    benchmark = (df_index['mkt_return']+1).cumprod()

    
    # Plotting group / benchmark ratio
    group_bench = df_group_nav.iloc[-1, -groups-1:-1] / benchmark.values[-1]
    group_bench  = np.power(group_bench, 12/len(benchmark))-1
    
    plt.figure(figsize=(10, 5))
    group_bench.plot(kind='bar')
    plt.title(f'{factor} group portfolio over benchmark ratio', fontsize=20)
    plt.savefig(f'{factor} group portfolio over benchmark ratio.png')

    
    # Plotting group performance
    plt.figure(figsize=(16, 9))

    line_width = [2] * groups + [4]
    line_style = ['-'] * groups + ['--']

    for i in range(groups+1):
        plt.plot(df_group_nav.iloc[:, i], linewidth=line_width[i], linestyle=line_style[i])
    
    plt.legend(list(range(groups)) + ['L-S'])
    
    plt.title(f'{factor} group performance', fontsize=20)
    plt.savefig(f'{factor} group performance.png')

    df_group_raw = df     

    return df_group_nav, df_group_turnover, df_group_raw


def get_turnover(data, plot=False):
    # data = df_choose.copy()

    turnover = pd.DataFrame()
    turnover['formatted_date'] = data['formatted_date'].unique()
    turnover['turnover'] = 0.0

    for i in range(1, len(turnover)):
        stock_now = set(data.loc[data.formatted_date == turnover.formatted_date[i], 'ticker'].tolist())
        stock_pre = set(data.loc[data.formatted_date == turnover.formatted_date[i - 1], 'ticker'].tolist())
        turnover.loc[i, 'turnover'] = len(stock_pre.difference(stock_now)) / len(stock_pre) 
        
    turnover = turnover.sort_values(by='formatted_date').set_index('formatted_date')
    
    if plot:
        # xtick = np.arange(0, turnover.shape[0], 12)
        # xticklabel = pd.Series(turnover.index[xtick])

        # plt.figure(figsize=(8, 4))
        # ax = plt.axes()

        # plt.bar(np.arange(turnover.shape[0]), turnover.turnover, color='darkred', width=0.2)
        
        # ax.set_xticks(xtick)
        # ax.set_xticklabels(xticklabel)

        turnover.plot(xlabel='')
        plt.title('Turnover', fontsize = 20)

    return turnover


def top_n_plot(df, factor, df_index, f_ic=1, top_n=10, weigh=False, weights=[0.1]*10 ):
    
    # Pick top/bottom n based on factor ic value
    ascending = 0 if f_ic > 1 else 1

    df = df.sort_values(by=factor, ascending=ascending)
    df_pick = df.groupby('formatted_date').head(top_n).reset_index(drop=True)
    
    # Determine weight scheme then calculated performance results
    if weigh:
        result = df_pick['next_period_return'].groupby(df_pick.formatted_date).apply(lambda x: np.average(x, weights=weights, axis=0))
    else:
        result = df_pick['next_period_return'].groupby(df_pick.formatted_date).mean()
    
    # move 1 period down since not trading in the first period    
    df_result = pd.DataFrame(result)
    
    # move return to the right period
    df_result = pd.merge(df_index['formatted_date'],df_result,on='formatted_date',how='left')
    df_result.set_index('formatted_date', inplace=True)
    df_result= df_result.shift(periods=1)
    df_result.reset_index(inplace=True)
    df_result.fillna(0)
    

    
    # get performance curve
    df_n = pd.DataFrame()
    df_n['formatted_date'] = df_index['formatted_date']
    df_n['benchmark'] = (df_index['mkt_return']+1).cumprod()
    df_n['portfolio'] = (df_result['next_period_return']+1).cumprod()
    df_n['RS'] = df_n['portfolio'] / df_n['benchmark']
    df_n = df_n.set_index('formatted_date')
    
    # get turnover info
    df_turnover = get_turnover(df_pick)
    
    # get yearly performance
    df_n['year'] = pd.Series(df_n.index).apply(lambda x: x.year).values
    df_year = df_n.groupby('year').last() / df_n.groupby('year').first() - 1
    df_year['ex_return'] =  df_year.iloc[:, 0] -  df_year.iloc[:, 1]    

    # Plotting turnover
    df_turnover.plot(figsize=(10, 5), color='darkred', xlabel='',legend=False)
    plt.title(f'{factor} Top {top_n} pick turnover', fontsize=20)
    plt.savefig(f'{factor} Top {top_n} pick turnover.png') 
    
    # Plotting performance curves
    plt.figure(figsize=(16, 9))
    plt.plot(df_n['portfolio'], linewidth=2, c='deepskyblue', label='Portfolio')
    plt.plot(df_n['benchmark'], linewidth=2, c='darkred', label='benchmark')
    plt.plot(df_n['RS'], linewidth=2, c='orange', label='Portfolio/benchmark')

    plt.legend(fontsize=12)
    plt.title(f'{factor} top {top_n} pick performance', fontsize=20)
    plt.savefig(f'{factor} top {top_n} pick performance.png')       
        
    return df_pick, df_n, df_turnover, df_year
        


if __name__ == '__main__':
    
    df_pool = pd.read_pickle('df_pool_SP500_M.pkl')
    df_pool = df_pool.sort_values(['formatted_date', 'ticker'])
    
    # factors = df_pool.columns[10:-3].to_list() 
    factors = ['price_volume']    
    ic_icirs = []
    
    # Calculate index return and next period return
    df_index = df_pool[['formatted_date', 'daily_mkt_returns']].drop_duplicates(subset=['formatted_date'])    
    df_index['mkt_return'] = df_index['daily_mkt_returns'].apply(lambda x: np.prod([1+i for i in x])-1)
    
    
    df_pool.dropna(subset=['next_period_daily_returns'], inplace=True)
    df_pool['next_period_return'] = df_pool['next_period_daily_returns'].apply(lambda x: np.cumprod(np.array(x) + 1)[-1] - 1)
    
    
    for factor in factors:
        # 
        df = df_pool.dropna(subset=['formatted_date',factor])
        
        # Plot factor distribution by year
        df_factor_dist = factor_dist_plot(df[['formatted_date',factor]], factor)
        
        # Plot factor auto correlation over time
        auto_corr_plot(df[['formatted_date','ticker',factor]], factor)
        
        # Scale factors with large sample, or optional
        # df_scale = scale_factors(df[['formatted_date', 'ticker',  'sector',  factor]])
        # df_scale = pd.merge(df_scale, df[['formatted_date', 'ticker',  'sector', 'next_period_return']], on=['formatted_date', 'ticker'])
        df_scale = df[['formatted_date', 'ticker',  'sector', factor, 'next_period_return']]
        
        # Plot ic icir
        f_ic, f_icir = ic_icir_plot(df_scale, factor)
        
        # save ic, icir to list
        ic_icirs.append((factor, f_ic, f_icir, abs(f_ic)))     
        
        # Plot factor ic decay
        ic_decay, ic_half = ic_decay_plot(df_scale, factor)
        
        # Plot group ic
        df_group_ic = group_ic_plot(df_scale, factor)
        
        # Plot sector ic
        df_ic_sec, ic_sec_mean, icir_sec = sector_ic_plot(df_scale, factor)
        
        # Plot group test
        df_group_nav, df_group_turnover, df_group_all = group_test_plot(df_scale, factor, df_index)
        
        # Plot portfolio with top picks performance 
        df_pick, df_n, df_turnover, df_year = top_n_plot(df_scale, factor, df_index, f_ic=1)
       
    # save 
    ic_icir_df = pd.DataFrame(ic_icirs, columns=['factor', 'IC', 'ICIR','IC_abs'])
    ic_icir_df.sort_values(by=['IC_abs'], inplace=True)
    ic_icir_df.to_csv('factors ic.csv', index=False)
    print(ic_icir_df)
   
        
        
  
        

       


     
        
        


