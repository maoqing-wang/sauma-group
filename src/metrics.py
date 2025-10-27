import pandas as pd
import math
from numpy import sqrt, maximum, ones, array, quantile, log, exp, nan, ndarray, isnan

def sharpe(dailyReturns: pd.Series) -> float:
    idx = dailyReturns.first_valid_index()
    if idx is None:
        return nan
    dailyReturns = dailyReturns[idx:]
    stdev = dailyReturns.std()
    if stdev == 0:
        return 0.0
    return dailyReturns.mean() / stdev * sqrt(252)

def mdd(cumpnl: pd.Series) -> float:
    idx = cumpnl.first_valid_index()
    if idx is None:
        return nan
    cumpnl = cumpnl[idx:]
    arr = cumpnl.to_numpy()
    top = maximum.accumulate(arr)
    return (top - cumpnl).max()

def currentDrawdownDays(cumpnl: pd.Series) -> float: # drawdown start(峰值时间点，存起来)
    idx = cumpnl.first_valid_index()
    if idx is None:
        return nan
    cumpnl = cumpnl[idx:]
    arr = cumpnl.to_numpy()
    top = maximum.accumulate(arr)
    drawdown = top - arr
    
    n = 0
    for dd in drawdown[::-1]:
        if dd == 0:
            break
        n += 1
    return n

def calmar(cumpnl: pd.Series) -> float:
    maxDD = mdd(cumpnl)
    if maxDD == 0:
        return 100.0
    # sometimes maxDD can be 0, so its capped at 100 to avoid division by 0
    return min(cumpnl.iloc[-1]/maxDD, 100)

def ldd(cumpnl: pd.Series) -> float:
    '''
        output: float, time units
    '''
    idx = cumpnl.first_valid_index()
    if idx is None:
        return nan
    cumpnl = cumpnl[idx:]
    arr = cumpnl.to_numpy()
    curmax = -math.inf
    maxlong, curlong = 0.0, 0.0
    for val in arr:
        if val >= curmax:
            curmax = val
            maxlong = max(maxlong, curlong)
            curlong = 0
        else:
            curlong += 1
    return max(maxlong, curlong)

def calc_metrics(cumpnl: pd.Series):
    return mdd(cumpnl), ldd(cumpnl), calmar(cumpnl), sharpe(cumpnl), currentDrawdownDays(cumpnl)

def compute_rolling_metrics(rolling: list[pd.DataFrame], ret:pd.DataFrame):
    files = [c for c in rolling[0].columns if c != 'price']

    rolling_metrics_df = {
        c: {'timestamp': [], 'mdd': [], 'ldd': [], 'cddDays': []
            , 'calmar': [], 'sharpe': [], 'excess_ret': []}
          for c in files}

    for df in rolling:
        timestamp = df.index[-1]
        pnl_arr = df[files].to_numpy()  # shape = (time, n_cols)
 
        for i, col in enumerate(files):
            cum_pnl = pnl_arr[:, i]
            maxdd, longdd, cr, sr, cddd = calc_metrics(pd.Series(cum_pnl, index=df.index))
            rolling_metrics_df[col]['timestamp'].append(timestamp)
            rolling_metrics_df[col]['mdd'].append(maxdd)
            rolling_metrics_df[col]['ldd'].append(longdd)
            rolling_metrics_df[col]['calmar'].append(cr)
            rolling_metrics_df[col]['sharpe'].append(sr)
            rolling_metrics_df[col]['cddDays'].append(cddd)
            rolling_metrics_df[col]['excess_ret'].append(ret.at[timestamp, col])
    
    rolling_metrics = {}
    for col in files:
        df_metrics = pd.DataFrame(rolling_metrics_df[col]).set_index('timestamp')
        df_metrics[df_metrics.isna().any(axis=1)] = nan
        rolling_metrics[col] = df_metrics

    return rolling_metrics


def linear_transformer(measures: pd.Series, alpha: float):
    measures = measures.dropna()

    z_alpha = quantile(measures, alpha)
    z_1m_alpha = quantile(measures, 1-alpha)
    if z_1m_alpha == z_alpha:
        return measures
    a = log((1-alpha)/alpha)
    beta = 2*a / (z_1m_alpha - z_alpha)

    return beta * (measures - z_alpha) - a

def linear_transformer(measures: ndarray, alpha: float):
    """Vectorized linear transformer for numpy arrays."""
    mask = ~isnan(measures)
    clean = measures[mask]
    if clean.size == 0:
        return measures
    
    z_alpha = quantile(clean, alpha)
    z_1m_alpha = quantile(clean, 1-alpha)
    if z_1m_alpha == z_alpha:
        return measures
    a = log((1-alpha)/alpha)
    beta = 2 * a / (z_1m_alpha - z_alpha)

    out = measures.copy()
    out[mask] = beta * (clean - z_alpha) - a
    return out

def sigmoid(measures: ndarray):
    return 1 / (1 + exp(-measures))

def two_steps_transformer(measures: ndarray, alpha: float):
    return sigmoid(linear_transformer(measures, alpha))

def compute_EI(rolling_metrics: dict[pd.DataFrame], alpha: float, weights = None) -> pd.DataFrame:
    positive_metrics_name = ['sharpe', 'calmar', 'excess_ret'] # may expand later
    EIs_list = []

    # transform first
    for name, df in rolling_metrics.items():
        df_vals = df.to_numpy()
        col_names = df.columns.to_list()
        pos_mask = array([c in positive_metrics_name for c in col_names])

        for j in range(df_vals.shape[1]):
            col_arr = df_vals[:, j]
            if pos_mask[j]:
                df_vals[:, j] = two_steps_transformer(col_arr, alpha)
            else:
                df_vals[:, j] = 1 - two_steps_transformer(col_arr, alpha)
        
        df_transformed = pd.DataFrame(df_vals, index=df.index, columns=df.columns)
    
    # get EIs
        K = len(col_names)
        if weights is None:
            w = ones(K) / K
        else:
            w = array(weights)
        tmp = df_transformed.dot(w)
        EIs_list.append(tmp.rename(name))
        
    EIs = pd.concat(EIs_list, axis=1)
    EIs.fillna(0, inplace=True)
    return EIs


