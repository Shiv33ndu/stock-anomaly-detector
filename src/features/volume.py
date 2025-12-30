import pandas as pd
import numpy as np



def compute_volume_zscore(volume: pd.Series, window:int = 21) -> pd.Series:
    """
    Using log(volume), this computes volume_zscore
    
    :param df: DataFrame
    :type df: pd.DataFrame
    :param window: No of days of window
    :type window: int
    :return: Series of volume_zscore
    :rtype: Series[Any]
    """

    if volume.empty:
        raise ValueError("This dataframe is empty!")
    
    # compute log(volume)
    volume = volume.replace(0, np.nan)

    log_volume = np.log(volume)

    # shift the series data of log(vol)
    shifted_log_vol = log_volume.shift(1)

    # rolling mean & std 
    rolling_mean = shifted_log_vol.rolling(window=window).mean()
    rolling_std = shifted_log_vol.rolling(window=window).std()

    vol_zscore = (log_volume - rolling_mean) / np.where(rolling_std == 0, np.nan, rolling_std)

    return vol_zscore
