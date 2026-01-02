import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN 


def fit_dbscan(X_ref: np.ndarray, eps: float, min_sample: int) -> DBSCAN:
    """
    Docstring for fit_dbscan
    
    Fit DBSCAN on reference data (training or expanding window).

    Returns:
    fitted DBSCAN model
    """
    if len(X_ref) == 0:
        raise ValueError("Input data is empty!")
    dbscan = DBSCAN(eps=eps, min_samples=min_sample)
    dbscan.fit(X_ref)

    return dbscan

def detect_dbscan_anomaly(labels: np.ndarray, meta: pd.DataFrame) -> pd.DataFrame:
    """
    detect_dbscan_anomaly for given labels and meta
    by extracting DBSCAN anomalies (label == -1) and attach metadata.
    
    Returns:
    dataframe of anomalies
    """

    if len(labels) != len(meta):
        raise ValueError("Labels and meta must be aligned")
    
    # masking for noise points
    anomaly_mask = labels == -1

    anomalies = meta.loc[anomaly_mask].copy()
    anomalies['anomaly_flag'] = 1
    anomalies['cluster'] = -1

    return anomalies.reset_index(drop=True)