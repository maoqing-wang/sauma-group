import pandas as pd
from numpy import nan
def rolling_window(df: pd.DataFrame, window: int, step: int = 1) -> list[pd.DataFrame]:
    """
        output: list of DataFrame, each element is the original data within current window
    """
    dfs = []
    for start in range(0, len(df) - window + 1, step):
        dfs.append(df.iloc[start:start + window])
    return dfs

def expanding_window(df: pd.DataFrame, step: int = 1, chunk = False) -> list[pd.DataFrame]:
    """
        output: list of DataFrame, each element is the original data within current window
    """
    dfs = []

    if chunk == True:
        # avoid data explosion
        # remove initial no trading rows
        for col in df.columns:
            if col == 'price':
                continue
            mask = (df[col] != 0)
            first = mask.idxmax() if mask.any() else None
            if first != None:
                df.loc[:first, col] = nan
            
    for end in range(1, len(df)+1, step):
        dfs.append(df.iloc[:end])
    return dfs

def align_series(series1: pd.Series, series2: pd.Series) -> pd.DataFrame:
    """
        output: DataFrame, each column is the aligned series
    """
    return pd.concat([series1, series2], axis=1)



