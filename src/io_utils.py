import os
import pandas as pd
from numpy import isin

def read_parquet_dir(
    directory: str, 
    daily: bool = True, 
    price: bool = False, 
    begin: pd.Timestamp = None, 
    end: pd.Timestamp = None
) -> pd.DataFrame:
    """Reads all parquet files in a directory and concatenates them into a single DataFrame."""
    
    # sanity check
    if price == True and (begin is None or end is None):
        raise ValueError("begin and end timestamps must be provided")

    target_times = [begin.time(), end.time()]
    pnls, longshorts = [], []

    for file in os.listdir(directory):
        if not file.endswith(".parquet"):
            continue

        file_path = os.path.join(directory, file)
        df = pd.read_parquet(file_path)

        pnl = df['cum_ot_pnl']
        pnl = pnl[isin(pnl.index.time, target_times)] if daily else pnl
        pnls.append(pnl)

        longshorts.append('short' if 'short' in file_path else 'long')

    # concatenate all pnl columns
    result = pd.concat(pnls, axis=1)
    result.columns = [f"{i}_{longshorts[i]}" for i in range(result.shape[1])]
    result['naive_ave_pnl'] = result.mean(axis=1)

    # add price column if requested
    if price:
        price_series = df['price']
        if daily:
            price_series = price_series[isin(price_series.index.time, target_times)]
        result['price'] = price_series

    return result

                       
def save_dataframe(df: pd.DataFrame, path: str):
    if path.endswith(".csv"):
        df.to_csv(path, index=False)
    elif path.endswith(".parquet"):
        df.to_parquet(path, index=False)
