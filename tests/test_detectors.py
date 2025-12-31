import pandas as pd
import numpy as np

from src.detectors.rules import detect_rule_anomaly
from src.detectors.kmeans import build_feature_matrix, split_by_date

def make_base_df():
    return pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=5),
        "ret":       [0.01,  0.01,  0.01,  0.01,  0.01],
        "ret_z":     [0.1,   0.2,   0.3,   0.2,   0.1],
        "vol_z":     [0.1,   0.2,   0.3,   0.2,   0.1],
        "range_pct": [0.2,   0.3,   0.4,   0.3,   0.2],
    })



# crash detection
def test_crash_detection():
    df = make_base_df()
    df.loc[3, "ret"] = -0.05
    df.loc[3, "ret_z"] = -3.0

    out = detect_rule_anomaly(df, ticker="TEST")

    assert len(out) == 1
    row = out.iloc[0]

    assert row["type"] == "crash"
    assert "|ret_z| > 2.5" in row["why"]


# spike detection
def test_spike_detection():
    df = make_base_df()
    df.loc[2, "ret"] = 0.06
    df.loc[2, "ret_z"] = 3.2

    out = detect_rule_anomaly(df, ticker="TEST")

    assert len(out) == 1
    row = out.iloc[0]

    assert row["type"] == "spike"
    assert "|ret_z| > 2.5" in row["why"]




# volume shock detection
def test_volume_shock_only():
    df = make_base_df()
    df.loc[1, "vol_z"] = 3.1

    out = detect_rule_anomaly(df, ticker="TEST")

    assert len(out) == 1
    row = out.iloc[0]

    assert row["type"] == "volume_shock"
    assert "vol_z > 2.5" in row["why"]




# crash + volume shock detection
def test_crash_and_volume_shock():
    df = make_base_df()
    df.loc[4, "ret"] = -0.07
    df.loc[4, "ret_z"] = -3.5
    df.loc[4, "vol_z"] = 3.2

    out = detect_rule_anomaly(df, ticker="TEST")

    assert len(out) == 1
    row = out.iloc[0]

    assert row["type"] == "crash+ volume_shock"
    assert "|ret_z| > 2.5" in row["why"]
    assert "vol_z > 2.5" in row["why"]





# range percentile detection
def test_range_pct_only():
    df = make_base_df()
    df.loc[0, "range_pct"] = 0.99

    out = detect_rule_anomaly(df, ticker="TEST")

    assert len(out) == 1
    row = out.iloc[0]

    assert row["type"] == ""
    assert "range_pct > 0.95" in row["why"]



# NaN handling (Warm up safety)
def test_nan_row_is_ignored():
    df = make_base_df()
    df.loc[2, "ret_z"] = np.nan

    out = detect_rule_anomaly(df, ticker="TEST")

    assert len(out) == 0



# output schema test
def test_output_schema():
    df = make_base_df()
    df.loc[1, "vol_z"] = 3.0

    out = detect_rule_anomaly(df, ticker="TEST")

    expected_cols = {
        "date", "ticker", "anomaly_flag",
        "type", "ret", "ret_z", "vol_z",
        "range_pct", "why"
    }

    assert set(out.columns) == expected_cols


# test feature matrix shape
def test_feature_matrix_shapes():
    df = pd.DataFrame({
        "date": ["2020-01-01", "2020-01-02"],
        "ticker": ["A", "A"],
        "ret": [0.01, -0.02],
        "ret_z": [0.5, 1.2],
        "vol_z": [0.3, 2.1],
        "range_pct": [0.2, 0.8],
    })

    X, meta = build_feature_matrix(df)

    assert X.shape == (2, 3)
    assert len(meta) == 2


# split_by_date test
def test_split_by_date():
    meta = pd.DataFrame({
        "date": ["2018-01-02", "2018-06-15", "2019-03-10", "2019-11-20", "2020-02-27"],
        "ticker": ['AAPL', 'MSFT', 'AAPL', 'MSFT', 'AAPL'],
        "ret": [0.01, 0.02, -0.01, 0.03, -0.04],
    })

    X = np.array([
      [ 0.2,  0.3, 0.10 ],   # row 0
      [ 0.1,  0.4, 0.20 ],   # row 1
      [ 1.5,  2.1, 0.80 ],   # row 2
      [ 0.3,  0.5, 0.15 ],   # row 3
      [ 3.2,  3.8, 0.97 ],   # row 4
    ])

    X_train, X_val, X_test, meta_train, meta_val, meta_test = split_by_date(X, meta)

    assert len(X_train) == len(meta_train)
    assert len(X_val) == len(meta_val)
    assert len(X_test) == len(meta_test)

    assert meta_train['date'].dt.year.nunique() == 1
    assert meta_train['date'].dt.year.iloc[0] == 2018

    
