import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

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


def scale_features(X_train, X_val, X_test):
    """
    Fit StandardScaler on X_train only, 
    transform all splits.
    """
    # instantiate the Scaling object
    scaler = StandardScaler()

    # fit to X_train
    scaler.fit(X_train)

    # transform all the splits as per X_train
    X_train_scaled = scaler.transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    return scaler, X_train_scaled, X_val_scaled, X_test_scaled


def fit_kmeans(X_train_scaled, k, random_state=42):
    """
    Fit the KMeans to X_train_scaled data
    """
    if len(X_train_scaled) == 0 :
        raise ValueError("X_train_scaled is empty for KMeans")
    
    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init='auto')
    kmeans.fit(X_train_scaled)

    return kmeans


def compute_kmeans_distance(X_scaled, kmeans):
    """
    Compute distance to nearest centroid for each point.
    """
    if len(X_scaled) == 0:
        raise ValueError("Input is empty")
    
    # distance to all centroids from each point
    all_distances = kmeans.transform(X_scaled)

    # distance to nearest centroid
    min_distances = all_distances.min(axis=1)

    # cluster assignment
    labels = kmeans.predict(X_scaled)

    return min_distances, labels 


def compute_cluster_thresholds(distances, labels, percentile=97.5):
    """
    Compute distance threshold per cluster using training data only.
    """
    thresholds = {}

    # we take out clusters from labels
    clusters = np.unique(labels)

    for c in clusters:
        # we create bool mask to pick the distances 
        # where the cluster value matches the labels
        # this extracts only c cluster's distance values
        cluster_distances = distances[labels == c]

        # if there are no fetched distances we continue
        if len(cluster_distances) == 0:
            continue

        # we calculate the percentile for cluster_distances
        thresholds[c] = np.percentile(cluster_distances, percentile)

    return thresholds


def detect_kmeans_anomaly(distances, labels, thresholds, meta):
    """
    Returns DataFrame of KMeans-based anomalies

    The logic behind using minimum distance as base for threshold
    computation is, the points that have larger minimum distance than 
    other points are potential anomalies and to make it more reliable 
    we use thesholding, that says like 95% of the points are having
    minimum distance less than this and only 5% percent of the points
    have quite large minimum distance making them anomaly.
    """
    if meta.empty:
        raise ValueError("Metadata is empty")

    meta = meta.copy()

    if len(distances) != len(labels) or len(distances) != len(meta):
        raise ValueError("distance, labels, and meta must be aligned")
    
    records = []

    for i in range(len(distances)):
        cluster = labels[i]

        if cluster not in thresholds:
            continue

        is_anomaly = distances[i] > thresholds[cluster]

        if is_anomaly:
            records.append({
                "date": meta.iloc[i]['date'],
                "ticker": meta.iloc[i]['ticker'],
                'anomaly_flag': 1,
                "cluster": int(cluster),
                "distance": distances[i]
            })
    
    return pd.DataFrame(records)
        