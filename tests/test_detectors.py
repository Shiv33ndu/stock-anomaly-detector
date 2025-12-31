import pandas as pd
import numpy as np

from src.detectors.rules import detect_rule_anomaly

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
