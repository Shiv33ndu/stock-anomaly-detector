import pandas as pd
import numpy as np



def compute_range_percentile(df: pd.DataFrame, window=63) -> pd.Series:
    """
    Computes intraday range percentile 
    
    :param df: Dataframe of ticker
    :type df: pd.DataFrame
    :param window: No of days of windows
    :return: Returns Series of intraday percentiles
    :rtype: Series[Any]
    """

    if df.empty:
        raise ValueError('The DataFrame is empty!')
    
    # computing the intraday range 
    intra_range = (df['high'] - df['low']) / df['close']

    # range container 
    range_pct = pd.Series(index=df.index, dtype=float)

    for i in range(len(intra_range)):
        if i < window:
            range_pct.iloc[i] = np.nan
            continue

        past_window = intra_range.iloc[i - window : i]
        range_pct.iloc[i] = (past_window <= intra_range.iloc[i]).mean()
    
    return range_pct

