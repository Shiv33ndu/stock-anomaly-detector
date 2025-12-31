import pandas as pd
import numpy as np

def build_feature_matrix(df_features: pd.DataFrame):
    """
    Takes a feature DataFrame with columns:
    date, ticker, ret, ret_z, vol_z, range_pct

    Returns:
    X   -> numpy array of shape (n_samples, 3)
    meta-> DataFrame with date, ticker, ret (aligned with X) 
    """

    if df_features.empty:
        raise ValueError("DataFrame is empty!")
    
    feature_col = ['ret_z', 'vol_z', 'range_pct']

    df = df_features.dropna(subset=feature_col).copy()

    X = df[feature_col].values

    meta = df[['date', 'ticker', 'ret']].reset_index(drop=True)

    return X, meta


def split_by_date(X: np.array, meta: pd.DataFrame):
    """
    Splits X and meta into train / val / test
    based on meta['date']
    """

    if meta.empty:
        raise ValueError("Metadata is missing!")
    
    if len(X) == 0:
        raise ValueError("Feature Matrix X is missing!")
    
    meta = meta.copy()

    if meta['date'].dtype != "datetime64[ns]":
        meta['date'] = pd.to_datetime(meta['date'],format="%Y-%m-%d")

    dates = meta['date']

    # date masks 
    train_year = dates.dt.year == 2018
    val_year = dates.dt.year == 2019
    test_year = (dates >= '2020-01-01') & (dates <= '2020-03-31')
    
    # splitting X as per mask
    X_train = X[train_year]
    X_val = X[val_year]
    X_test = X[test_year]

    # splittin meta as per mask
    meta_train = meta.loc[train_year].reset_index(drop=True)
    meta_val = meta.loc[val_year].reset_index(drop=True)
    meta_test = meta.loc[test_year].reset_index(drop=True)

    return X_train, X_val, X_test, meta_train, meta_val, meta_test
