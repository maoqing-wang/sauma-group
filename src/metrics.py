import pandas as pd
import numpy as np
from tdigest import TDigest

def mdd_percent(cumpnl: pd.Series) -> np.array:
    arr = cumpnl.to_numpy()
    top = np.maximum.accumulate(arr)
    running_dd = top - arr
    mdd = np.maximum.accumulate(running_dd)

    out = np.where(arr != 0, mdd / np.where(arr != 0, arr, np.nan), np.nan)
    return out

def cdd_percent(cumpnl: pd.Series) -> np.array:
    arr = cumpnl.to_numpy()
    top = np.maximum.accumulate(arr)
    running_dd = top - arr

    out = np.where(arr != 0, running_dd / np.where(arr != 0, arr, np.nan), np.nan)
    return out

def ldd_days(cumpnl: pd.Series) -> np.array:
    arr = cumpnl.to_numpy()
    curmax = -1e99; maxlong=0; curlong=0
    ldd = []
    for v in arr:
        if v>=curmax:
            curmax=v; maxlong=max(maxlong,curlong); curlong=0
        else:
            curlong+=1
        ldd.append(max(maxlong, curlong))
    return np.array(ldd)

def ldd_percent(cumpnl: pd.Series) -> np.array:
    days = ldd_days(cumpnl)
    arr = cumpnl.to_numpy()
    out = np.where(arr != 0, days / np.where(arr != 0, arr, np.nan), np.nan)
    return out

def mdd_days(cumpnl: pd.Series) -> np.array:
    arr = cumpnl.to_numpy()
    top = np.maximum.accumulate(arr)
    running_dd = top - arr
    mddd = np.zeros(len(arr))

    for t in range(len(arr)):
        dd = running_dd[:t+1]
        i = dd.argmax()
        if dd[i] == 0:
            mddd[t] = 0
        else:
            d = 0
            j = i
            while j >= 0 and dd[j] > 0:
                d += 1
                j -= 1
            mddd[t] = d
    mddd[cumpnl == 0] = np.nan
    return mddd

def sharpe_ratio(returns: pd.Series) -> np.array:
    r = returns.to_numpy()
    n = np.arange(1, len(r) + 1)
    runningr = np.cumsum(r)
    runningr2 = np.cumsum(r**2)

    rmean = runningr / n
    rvar = (runningr2 / n) - rmean ** 2
    rstd = np.sqrt(np.maximum(rvar, 0))
    out = np.where(rstd == 0, np.nan, rmean / rstd * np.sqrt(252))
    out[returns == 0] = np.nan
    return out

def calmar(cumpnl: pd.Series) -> np.array:
    arr = cumpnl.to_numpy()
    top = np.maximum.accumulate(arr)
    running_dd = top - arr
    mdd = np.maximum.accumulate(running_dd)
    out = np.where(mdd != 0, arr / mdd, np.nan)
    return out

def pnl_t(cumpnl: pd.Series) -> np.array:
    return cumpnl


def days_since_first_trade(cumpnl: pd.Series) -> np.array:
    if (cumpnl != 0).sum() == 0:
        return np.full(len(cumpnl), np.nan)
    first_date = (cumpnl != 0).idxmax()

    deltas = pd.to_timedelta(cumpnl.index - first_date).days
    out = np.where(deltas >= 0, deltas, np.nan)
    return out

def excess_return(cumpnl: pd.Series, price: pd.Series, position) -> np.array:
    out = cumpnl - price.values * position
    out[cumpnl == 0] = np.nan
    return out
def excess_mean_return(cumpnl: pd.Series, price: pd.Series, position) -> np.array:
    traded_days = (cumpnl != 0).astype(int).cumsum().replace(0, np.nan)
    market_days = (price != 0).astype(int).cumsum().replace(0, np.nan)
    ret = cumpnl / traded_days.to_numpy() - (price.values / market_days.to_numpy()) * position
    return ret

def drawdown_beta(cumpnl: pd.Series, price: pd.Series, position) -> float:
    if (cumpnl != 0).sum() == 0:
        return np.nan

    start_idx = (cumpnl != 0).idxmax()
    cumpnl = cumpnl.loc[start_idx:]
    price = price.loc[start_idx:]

    # drawdown period
    top = np.maximum.accumulate(price)
    in_drawdown = price <= top

    if in_drawdown.sum() < 2:
        return np.nan

    bench = price[in_drawdown] * position
    pnl = cumpnl[in_drawdown]

    # must return a scalar!!
    cov = np.cov(bench, pnl, ddof=1)[0, 1]
    var = np.var(bench, ddof=1)

    return float(cov / var) if var != 0 else np.nan

def drawup_beta(cumpnl: pd.Series, price: pd.Series, position) -> float:
    if (cumpnl != 0).sum() == 0:
        return np.nan

    start_idx = (cumpnl != 0).idxmax()
    cumpnl = cumpnl.loc[start_idx:]
    price = price.loc[start_idx:]

    top = np.maximum.accumulate(price)
    in_drawup = price >= top

    if in_drawup.sum() < 2:
        return np.nan

    bench = price[in_drawup] * position
    pnl = cumpnl[in_drawup]

    cov = np.cov(bench, pnl, ddof=1)[0, 1]
    var = np.var(bench, ddof=1)

    return float(cov / var) if var != 0 else np.nan

def running_drawdown_beta(cumpnl: pd.Series, price: pd.Series, position) -> np.array:
    out = []
    for i in range(len(cumpnl)):
        sub_cumpnl = cumpnl.iloc[:i+1]
        sub_price = price.iloc[:i+1]
        sub_pos = position
        out.append(drawdown_beta(sub_cumpnl, sub_price, sub_pos))
    return np.array(out)

def running_drawup_beta(cumpnl: pd.Series, price: pd.Series, position) -> np.array:
    out = []
    for i in range(len(cumpnl)):
        sub_cumpnl = cumpnl.iloc[:i+1]
        sub_price = price.iloc[:i+1]
        sub_pos = position
        out.append(drawup_beta(sub_cumpnl, sub_price, sub_pos))
    return np.array(out)

def success_rate(cumpnl: pd.Series) -> np.array:
    if (cumpnl != 0).sum() == 0:
        return np.full(len(cumpnl), np.nan)
    start_idx = (cumpnl != 0).idxmax()
    idx_pos = cumpnl.index.get_loc(start_idx)
    cumpnl2 = cumpnl.loc[start_idx:]
    pnl = cumpnl2.diff().fillna(0)

    out = []
    wins = 0
    for i in range(len(pnl)):
        if pnl.iloc[i] > 0:
            wins += 1
        out.append(wins / (i + 1))
    out = np.array(out)
    pad = np.full(idx_pos, np.nan)
    return np.concatenate([pad, out])

def cvar_5(cumpnl: pd.Series) -> np.array:
    if (cumpnl != 0).sum() == 0:
        return np.full(len(cumpnl), np.nan)

    start = (cumpnl != 0).idxmax()
    idx = cumpnl.index.get_loc(start)
    c2 = cumpnl.loc[start:]

    out = []
    for i in range(len(c2)):
        q5 = np.quantile(c2[:i+1], 0.05)
        t = c2[c2 <= q5]
        out.append(t.mean())

    out = np.array(out)
    pad = np.full(idx, np.nan)
    return np.concatenate([pad, out])

def cvar_5_tdigest(cumpnl: pd.Series) -> pd.Series:
    if (cumpnl != 0).sum() == 0:
        return np.full(len(cumpnl), np.nan)
    
    start = (cumpnl != 0).idxmax()
    idx_start = cumpnl.index.get_loc(start)
    c2 = cumpnl.loc[start:]

    n = len(c2)
    out = np.empty(n)

    digest = TDigest()
    cumulative_sum = []
    
    for i, val in enumerate(c2):
        digest.update(val)
        q5 = digest.percentile(5)
        values_so_far = np.array(c2.iloc[:i+1])
        out[i] = values_so_far[values_so_far <= q5].mean()

    pad = np.full(idx_start, np.nan)
    result = pd.Series(np.concatenate([pad, out]), index=cumpnl.index)
    return result

# -----------------------  calc matrics -----------------------
def calc_metrics(df: pd.DataFrame) -> dict:
    cols = df.columns
    price = df['price']
    dic = {}

    for col in cols:
        if 'price' in col : continue
        position = -1 if 'short' in col else 1
        cumpnl = df[col]
        m1  = mdd_percent(cumpnl)
        m2  = ldd_percent(cumpnl)
        m3  = sharpe_ratio(cumpnl)
        m4  = cdd_percent(cumpnl)
        m5  = mdd_days(cumpnl)
        m6  = ldd_days(cumpnl)
        m7  = pnl_t(cumpnl)
        m8  = days_since_first_trade(cumpnl)
        m9  = running_drawdown_beta(cumpnl, price, position)
        m10 = running_drawup_beta(cumpnl, price, position)
        m11 = excess_mean_return(cumpnl, price, position)
        m12 = excess_return(cumpnl, price, position)
        m13 = cvar_5_tdigest(cumpnl)
        m14 = success_rate(cumpnl)
        m15 = calmar(cumpnl)
        
        temp = {
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
            'success_rate':m14,
            'calmar': m15,
        }
        dic[col] = pd.DataFrame(temp, index=df.index)
    return dic


# -----------------------  transformation -----------------------

def linear_transformer(measures: np.ndarray, alpha: float):

    """Vectorized linear transformer for numpy arrays.""" 
    mask = measures != 0 
    clean = measures[mask]
    if clean.size == 0:
        return measures
    
    z_alpha = np.quantile(clean, alpha)
    z_1m_alpha = np.quantile(clean, 1-alpha)
    if z_1m_alpha == z_alpha:
        return measures
    a = np.log((1-alpha)/alpha)
    beta = 2 * a / (z_1m_alpha - z_alpha)

    out = measures.copy()
    out[mask] = beta * (clean - z_alpha) - a
    return out

def sigmoid(measures: np.ndarray):
    arr = np.asarray(measures)
    out = arr.copy()
    mask = arr != 0
    out[mask] = 1 / (1 + np.exp(-arr[mask]))
    return out

def two_steps_transformer(measures: np.ndarray, alpha: float):
    return sigmoid(linear_transformer(measures, alpha))

# -----------------------  get EI -----------------------

def compute_EI_time(dic: dict, alpha: float, weights = None) -> pd.DataFrame:
    positive_metrics_name = ['sharpe','pnl_t','days_since_first_trade',
        'drawup_beta','excess_mean_ret','excess_ret','success_rate', 'calmar']
    
    EIs_list = []
    # transform each measure
    for col, df in dic.items():
        df_vals = df.to_numpy()
        metric_names = df.columns.to_list()
        pos_mask = np.array([c in positive_metrics_name for c in metric_names])

        for j in range(df_vals.shape[1]):
            col_arr = df_vals[:, j]
            if pos_mask[j]:
                df_vals[:, j] = two_steps_transformer(col_arr, alpha)
            else:
                temp = two_steps_transformer(col_arr, alpha)
                mask = temp == 0
                df_vals[:, j] = 1 - temp
                df_vals[mask, j] = 0
        df_transformed = pd.DataFrame(df_vals, index=df.index, columns=df.columns)
    
     # get EIs
     # weights as the distri
        K = len(metric_names)
        if weights is None:
            w = np.ones(K) / K
        else:
            w = np.array(weights)
        
        tmp = df_transformed.dot(w)
        EIs_list.append(tmp.rename(col))
        
    EIs = pd.concat(EIs_list, axis=1)
    EIs.fillna(0, inplace=True)
    return EIs

def compute_EI_trader(dic: dict, alpha: float, ts, weights = None) -> pd.DataFrame:
    positive_metrics_name = ['sharpe','pnl_t','days_since_first_trade',
        'drawup_beta','excess_mean_ret','excess_ret','success_rate', 'calmar']
    
    # re_arrange_df
    time_dic = {}
    time_index = dic[ts].index
    for t in time_index:
        records = {}
        for trader, df in dic.items():
            records[trader] = df.loc[t]
        time_dic[t] = pd.DataFrame.from_dict(records, orient='index')

    EIs_list = []
    # transform each measure
    for time, df in time_dic.items():
        df_vals = df.to_numpy()
        metric_names = df.columns.to_list()
        pos_mask = np.array([c in positive_metrics_name for c in metric_names])

        for j in range(df_vals.shape[1]):
            col_arr = df_vals[:, j]
            if pos_mask[j]:
                df_vals[:, j] = two_steps_transformer(col_arr, alpha)
            else:
                temp = two_steps_transformer(col_arr, alpha)
                mask = temp == 0
                df_vals[:, j] = 1 - temp
                df_vals[mask, j] = 0
        df_transformed = pd.DataFrame(df_vals, index=df.index, columns=df.columns)

     # get EIs
     # weights as the distri
        K = len(metric_names)
        if weights is None:
            w = np.ones(K) / K
        else:
            w = np.array(weights)
        
        tmp = df_transformed.dot(w)
        EIs_list.append(tmp.rename(time))
        
    EIs = pd.concat(EIs_list, axis=1)
    EIs.fillna(0, inplace=True)
    return EIs

def get_weights(rt = 0.25, drawdown = 0.25, length = 0.25, volatility = 0.25):
    # rt: drawup_beta, pnl_t, excess_mean_ret, excess_ret. --- 4
    # drawdown: mdd_p, ldd_p, cvar_5, drawdown_beta, cdd_percent, mdd_days, ldd_days, calmar --- 8
    # length: days_since_first_trade, success_rate --- 2
    # volatility: sharpe --- 1

    # order: 'mdd_percent’,’ ldd_percent’, ‘sharpe’:, ‘cdd_percent’, ‘mdd_days'
    # 'ldd_days’, ‘pnl_t’, ‘days_since_first_trade’, ‘drawdown_beta’, ‘drawup_beta’
    # ‘excess_mean_ret’, ‘excess_ret’, cvar_5’, ‘success_rate’, ‘calmar'

    return np.array([drawdown/8, drawdown/8, volatility, drawdown/8, drawdown/8, drawdown/8, rt/4, 
            length/2, drawdown/8, rt/4, rt/4, rt/4, drawdown/8, length/2, drawdown/8])
    

        
    