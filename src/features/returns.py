import pandas as pd
import numpy as np


def compute_daily_return(df: pd.DataFrame) -> pd.Series:
    """
    compute_daily_return computes the daily_return on Adj Close 
    using pct_change()
    and returns a Series 
    
    :param df: Validated and Cleaned Dataframe
    :type df: pd.DataFrame
    :return: Daily Return Series 
    :rtype: Series[Any]
    """

    if df.empty:
        raise ValueError(f"The DataFrame is empty!")
    
    daily_return = df['adj close'].pct_change()

    return daily_return

def compute_return_zscore(returns: pd.Series, window: int = 63) -> pd.Series:
    """
    Computes the return zscore with window of 63 days from past 
    
    :param df: Series data of daily return
    :type df: pd.Series
    :param window: No of days used as window
    :type window: int
    :return: Series data of return zscore
    :rtype: Series[Any]
    """

    if returns.empty:
        raise ValueError(f"The series is empty!")
    
    shift_return_mean = returns.shift(1).rolling(window=window).mean()
    shift_return_std = returns.shift(1).rolling(window=window).std()

    z_score = (returns - shift_return_mean) / np.where(shift_return_std == 0, np.nan, shift_return_std) 

    return z_score




