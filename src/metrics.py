import pandas as pd
import math
import numpy as np
from numpy import sqrt, maximum, ones, array, quantile, log, exp, nan, ndarray, isnan

# -----------------------  helpers -----------------------
def current_drawdown_amount(cumpnl: pd.Series) -> float:
    arr = cumpnl.to_numpy()
    top = maximum.accumulate(arr)
    return float((top - arr)[-1])


# -----------------------  13 metrics -----------------------

def mdd_percent(cumpnl: pd.Series) -> float:
    arr = cumpnl.to_numpy()
    top = maximum.accumulate(arr)
    mdd = (top - arr).max()
    total = cumpnl.iloc[-1]
    return nan if total == 0 else mdd / total

def ldd_percent(cumpnl: pd.Series) -> float:
    total = cumpnl.iloc[-1]
    return nan if total == 0 else ldd_days(cumpnl)/total

def sharpe_ratio(returns: pd.Series) -> float:
    st = returns.std()
    return 0.0 if st == 0 else returns.mean()/st*sqrt(252)

def cdd_percent(cumpnl: pd.Series) -> float:
    total = cumpnl.iloc[-1]
    return nan if total == 0 else current_drawdown_amount(cumpnl)/total

def mdd_days(cumpnl: pd.Series) -> float:
    arr = cumpnl.to_numpy()
    top = maximum.accumulate(arr)
    dd = top - arr
    i = dd.argmax()
    if dd[i] == 0: return 0
    d=0
    while i>=0 and dd[i]>0:
        d+=1; i-=1
    return d

def ldd_days(cumpnl: pd.Series) -> float:
    arr = cumpnl.to_numpy()
    curmax = -1e99; maxlong=0; curlong=0
    for v in arr:
        if v>=curmax:
            curmax=v; maxlong=max(maxlong,curlong); curlong=0
        else:
            curlong+=1
    return max(maxlong,curlong)

def pnl_t(cumpnl: pd.Series) -> float:
    return cumpnl.iloc[-1]

def days_since_first_trade(cumpnl: pd.Series) -> float:
    return len(cumpnl)

def excess_mean_return(returns: pd.Series, rf:float=0.0) -> float:
    return (returns.mean()-rf)*252

def cvar_5(returns: pd.Series) -> float:
    q5 = quantile(returns,0.05)
    t = returns[returns<=q5]
    return t.mean()

def calmar(cumpnl: pd.Series) -> float:
    arr = cumpnl.to_numpy()
    top = maximum.accumulate(arr)
    mdd = (top-arr).max()
    return 100.0 if mdd==0 else min(cumpnl.iloc[-1]/mdd,100)

def excess_return(df: pd.DataFrame):
    import numpy as np
    cols = [c for c in df.columns if c!='price']

    position = np.array([-1 if 'short' in c else 1 for c in cols])

    L = min(len(df.iloc[1::2]), len(df.iloc[::2]))

    if L == 0:
        # fallback – keep shape consistent
        return pd.DataFrame(index=df.index, columns=cols)

    # trade-level pnl = every two rows diff
    df1 = (
        df.iloc[1:1+2*L:2].reset_index(drop=True) 
        - df.iloc[:2*L:2].reset_index(drop=True)
    )

    # align timestamp to the exit rows
    df1.index = df.index[1:1+2*L:2]
   

    market = df1['price'].cumsum()
    pnl = df1[cols].cumsum()

    traded_days = (df1[cols]!=0).astype(int).cumsum().replace(0,np.nan)
    market_days = ((market!=0).astype(int)).cumsum().replace(0,np.nan)

    ret = pnl/traded_days - (market.values.reshape(-1,1)/market_days.values.reshape(-1,1))*position
    return ret


def success_rate_from_excess(ex_df: pd.DataFrame, col: str) -> float:
    s = ex_df[col].dropna()
    return nan if len(s)==0 else (s>0).sum()/len(s)

def drawdown_beta(asset_ret: pd.Series, price: pd.Series) -> float:
    bench = price.pct_change().dropna()
    a = asset_ret.loc[bench.index]
    m = bench<0
    if m.sum()<2: return nan
    return np.cov(a[m],bench[m])[0,1]/np.var(bench[m])

def drawup_beta(asset_ret: pd.Series, price: pd.Series) -> float:
    bench = price.pct_change().dropna()
    a = asset_ret.loc[bench.index]
    m = bench>0
    if m.sum()<2: return nan
    return np.cov(a[m],bench[m])[0,1]/np.var(bench[m])


def calc_metrics(cumpnl: pd.Series, returns: pd.Series, price: pd.Series, ex_df: pd.DataFrame, col: str):
    m1  = mdd_percent(cumpnl)
    m2  = ldd_percent(cumpnl)
    m3  = sharpe_ratio(returns)
    m4  = cdd_percent(cumpnl)
    m5  = mdd_days(cumpnl)
    m6  = ldd_days(cumpnl)
    m7  = pnl_t(cumpnl)
    m8  = days_since_first_trade(cumpnl)
    m9  = drawdown_beta(returns, price)
    m10 = drawup_beta(returns, price)
    m11 = excess_mean_return(returns)
    m12 = ex_df[col].iloc[-1] if (ex_df is not None and col in ex_df.columns) else np.nan
    m13 = cvar_5(returns)
    m14 = success_rate_from_excess(ex_df, col) if (ex_df is not None and col in ex_df.columns) else np.nan

    return {
        'mdd_percent':m1,
        'ldd_percent':m2,
        'sharpe':m3,
        'cdd_percent':m4,
        'mdd_days':m5,
        'ldd_days':m6,
        'pnl_t':m7,
        'days_since_first_trade':m8,
        'drawdown_beta':m9,
        'drawup_beta':m10,
        'excess_mean_ret':m11,
        'excess_ret':m12,
        'cvar_5':m13,
        'success_rate':m14
    }




def compute_rolling_metrics(rolling: list[pd.DataFrame], ret: pd.DataFrame):
    
    if len(rolling)==0:
        return {}
    # we must identify files from original df, not filtered windows
    base_df = rolling[0]                     # full df
    files   = [c for c in base_df.columns if c not in ('price','naive_ave_pnl')]

    # NOW filter rolling windows (after files defined)

    rolling = [df for df in rolling if ('price' in df.columns and df['price'].notna().sum()>2)]

    metric_names = [
        'mdd_percent','ldd_percent','sharpe','cdd_percent',
        'mdd_days','ldd_days','pnl_t','days_since_first_trade',
        'drawdown_beta','drawup_beta','excess_mean_ret',
        'excess_ret','cvar_5','success_rate'
    ]

    rolling_metrics_df = {c:{'timestamp':[]} for c in files}
    for c in files:
        for m in metric_names:
            rolling_metrics_df[c][m] = []

    # main loop
    for df in rolling:

        timestamp = df.index[-1]
        price     = df['price']
        ex_df     = excess_return(df) if len(df) >= 2 else None

        for col in files:
            rs = ret[col].reindex(df.index).fillna(0)
            cumpnl = df[col]                        # FIX HERE: no cumsum()

            vals = calc_metrics(cumpnl, rs, price, ex_df, col)

            rolling_metrics_df[col]['timestamp'].append(timestamp)
            for m in metric_names:
                rolling_metrics_df[col][m].append(vals[m])

    rolling_metrics = {}
    for col in files:
        df_metrics = pd.DataFrame(rolling_metrics_df[col]).set_index('timestamp')
        #df_metrics[df_metrics.isna().any(axis=1)] = nan
        rolling_metrics[col] = df_metrics

    return rolling_metrics



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
    positive_metrics_name = ['sharpe','pnl_t','days_since_first_trade',
        'drawup_beta','excess_mean_ret','excess_ret','success_rate'] # may expand later
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


