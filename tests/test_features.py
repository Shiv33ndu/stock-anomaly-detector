import numpy as np
import pandas as pd 
import pytest 

from src.features.returns import compute_daily_return, compute_return_zscore
from src.features.volume import compute_volume_zscore
from src.features.range import compute_range_percentile


def test_compute_daily_return_basic():
    df = pd.DataFrame(
        {"adj close": [100, 110, 121]},
        index=pd.date_range("2020-01-01", periods=3)
    )

    returns = compute_daily_return(df)

    assert np.isnan(returns.iloc[0])
    assert returns.iloc[1] == pytest.approx(0.10)
    assert returns.iloc[2] == pytest.approx(0.10)



def test_return_zscore_no_leakage():
    base_returns = pd.Series(
        [0.01, -0.01, 0.02, -0.02],
        index=pd.date_range("2020-01-01", periods=4)
    )

    spike_returns = pd.Series(
        [0.01, -0.01, 0.02, -0.02, 0.50],
        index=pd.date_range("2020-01-01", periods=5)
    )

    z_base = compute_return_zscore(base_returns, window=3)
    z_spike = compute_return_zscore(spike_returns, window=3)

    # z-score at t=3 must be identical
    assert z_base.iloc[3] == pytest.approx(z_spike.iloc[3])

    # spike must be detected
    assert z_spike.iloc[4] > 2.0




def test_volume_zscore_zero_and_spike():
    volume = pd.Series(
        [100, 120, 90, 110, 10000],
        index=pd.date_range("2020-01-01", periods=5)
    )

    z = compute_volume_zscore(volume, window=3)

    # warm-up
    assert np.isnan(z.iloc[0])
    assert np.isnan(z.iloc[1])
    assert np.isnan(z.iloc[2])

    # calm before spike
    assert abs(z.iloc[3]) < 1.0

    # spike detected
    assert z.iloc[4] > 2.0





def test_range_percentile_basic():
    df = pd.DataFrame(
        {
            "high":  [10, 10, 10, 10, 20],
            "low":   [9,  9,  9,  9,  9],
            "close": [10, 10, 10, 10, 10],
        },
        index=pd.date_range("2020-01-01", periods=5)
    )

    pct = compute_range_percentile(df, window=3)

    # warm-up
    assert np.isnan(pct.iloc[0])
    assert np.isnan(pct.iloc[1])
    assert np.isnan(pct.iloc[2])

    # calm day
    assert pct.iloc[3] == pytest.approx(1.0)

    # extreme range should be 100th percentile
    assert pct.iloc[4] == pytest.approx(1.0)




def test_zscore_constant_series():
    returns = pd.Series(
        [0.01] * 10,
        index=pd.date_range("2020-01-01", periods=10)
    )

    z = compute_return_zscore(returns, window=5)

    assert z.isna().all()
