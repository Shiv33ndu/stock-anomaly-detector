import numpy as np
import pandas as pd


def detect_rule_anomaly(
        df_features: pd.DataFrame,
        ticker: str,
        ret_z_thresh: float = 2.5,
        vol_z_thresh: float = 2.5,
        range_pct_thresh: float = 0.95
) -> pd.DataFrame:
    """
    This is a rule based anomaly detector that fires on given data
    - |ret_z| > 2.5  -> Crash
    - or vol_z > 2.5 -> Volume Shock
    - or range_pct > 0.95 -> contributes to anomaly flag and explanation
    
    :param df_features: Dataframe of concenred Ticker 
    :type df_features: pd.DataFrame
    :param ticker: Name of the Ticker
    :type ticker: str
    :param ret_z_thresh: return z score
    :type ret_z_thresh: float
    :param vol_z_thresh: log volume z score
    :type vol_z_thresh: float
    :param range_pct_thresh: intra day range percentile
    :type range_pct_thresh: float
    :return: Gives back a rule based anomaly for given Ticker as dataframe
    :rtype: DataFrame
    """

    # placeholder 
    anomalies = []

    if df_features.empty:
        raise ValueError("Dataframe is empty!")
    
    # getting ret_z, vol_z, range_pct
    returns = df_features['ret']
    ret_z = df_features['ret_z']

    # vol_z
    vol_z = df_features['vol_z']

    # range_pct
    range_pct = df_features['range_pct'] 

    
    # iterating row-wise
    for idx, row in df_features.iterrows():
        if (
            pd.isna(ret_z.loc[idx]) or
            pd.isna(vol_z.loc[idx]) or
            pd.isna(range_pct.loc[idx])
        ):
            continue
        
        anomaly_flag = 0
        anomaly_type = []
        anomaly_why = []       

        if abs(ret_z.loc[idx]) > ret_z_thresh: 
            anomaly_flag = 1
            if returns.loc[idx] < 0:
                anomaly_type.append('crash')
            else:
                anomaly_type.append('spike')
            anomaly_why.append(f'|ret_z| > {ret_z_thresh}') 
        
        
        if vol_z.loc[idx] > vol_z_thresh:
            anomaly_flag = 1
            anomaly_type.append('volume_shock')
            anomaly_why.append(f'vol_z > {vol_z_thresh}')

        if range_pct.loc[idx] > range_pct_thresh:
            anomaly_flag = 1
            anomaly_why.append(f'range_pct > {range_pct_thresh}')

        if anomaly_flag == 1:
            anomaly = {
                'date': row['date'],
                'ticker': ticker,
                'anomaly_flag': anomaly_flag,
                'type': '+ '.join(anomaly_type),
                'ret': returns.loc[idx],
                'ret_z': ret_z.loc[idx],
                'vol_z': vol_z.loc[idx],
                'range_pct': range_pct.loc[idx],
                'why': '; '.join(anomaly_why)
            }
            anomalies.append(anomaly)

    return pd.DataFrame(anomalies)


    
