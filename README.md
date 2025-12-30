## Stock Market Anomaly Detection

Detect unusual market days and stock days using only daily price and volume data. “Unusual” means very large moves or spikes or dip, unusually high volume, or unusually wide ranges compared to recent history. No labels, no news, no complex models.

## Project Structure

```
└── 📁stock-anomaly-detector
    └── 📁data
        └── 📁processed
        └── 📁raw
    └── 📁notebooks
        ├── 01_eda.ipynb
        ├── 02_feature_sanity.ipynb
    └── 📁src
        └── 📁cli
            ├── __init__.py
            ├── monthly.py
            ├── query.py
            ├── walkforward.py
        └── 📁data
            ├── load.py
            ├── validate.py
        └── 📁detectors
            ├── __init__.py
            ├── dbscan.py
            ├── kmeans.py
            ├── rules.py
        └── 📁evaluation
            ├── __init__.py
            ├── metrics.py
        └── 📁features
            ├── __init__.py
            ├── range.py
            ├── returns.py
            ├── volume.py
        └── 📁market
            ├── __init__.py
            ├── aggregate.py
        └── 📁reporting
            ├── __init__.py
            ├── daily_card.py
            ├── monthly_report.py
        ├── __init__.py
        ├── config.py
    └── 📁tests
        ├── __init__.py
        ├── test_detectors.py
        ├── test_features.py
        ├── test_leakage.py
    ├── .gitignore
    ├── README.md
    └── requirements.txt
```


## Step 1: Data Loader and Validator

This step implements loading, schema validating, cleaning and revalidating the data with financial sanity. The loader strictly enforce the required schema for the data, sorting by date in ascending order for time-series data preparation. This structural enforcement gets validated and financial sanity gets checked, if any issue persists the cleaner layer sanitizes the data and revalidation layer checks for any erroneous entry, if any the pipeline stops and raises the error avoiding broken data to be passed further.  

* `load_ticker_csv(file_path)` is a method in `src/data/load.py` that does following: 
    - parse Date as datetime
    - sort by Date ascending 
    - enforce the columns
    - returns A clean DataFrame df

* `validat_ohlcv(df, strict=False)` is the method in `src/data/validate.py` that receives the Dataframe from `load_ticker_csv()` and does the following:
    - checks for empty df
    - checks for duplicated dates entry
    - checks if the dates sorted in ascending order, if not then sorts it
    - check for required columns 
    - checks volumne, High, Low erronous entries and returns issues, if there's any

* `clean_ohlcv(df, issues, ticker)` is the method in `src/data/clean_ohlcv.py` that works as financial sanitizer if `validate_ohlcv` returns issues in `strict=False` mode
    - it drops all the erronous rows by the given indices `idx`
    - cleans the dataframe
    - returns the dataframe `df` for validation via `validate_ohlcv` in `strict=True` mode 

*Note : Known data issues (rare OHLC violations) are handled by dropping the affected rows and logging the event. No price values are imputed or modified.* 

--- 

# Step 2: Leakage-Safe Feature Engineering

This step implements the core feature set used for anomaly detection, strictly following the project specification and enforcing time-series hygiene (no look-ahead bias).

All features are computed per ticker, using rolling windows that rely only on past data.

# Objectives

- Transform raw OHLCV data into a small, interpretable feature set
- Prevent any form of data leakage
- Ensure features are mathematically correct, robust, and testable
- Validate behavior through unit tests on synthetic edge cases

# Feature Overview

For each ticker \( i \) and trading day \( t \), the following features are computed:

| Feature     | Description                     | Rolling Window |
|-------------|---------------------------------|----------------|
| ret         | Daily return using Adjusted Close | —              |
| ret_z       | Z-score of daily return          | 63 days        |
| vol_z       | Z-score of log(volume)           | 21 days        |
| range_pct   | Percentile of intraday range     | 63 days        |

All rolling statistics are computed over \( t - W, \dots, t - 1 \) (past only).

## 1. Daily Return

Daily return is computed using Adjusted Close to correctly account for stock splits and dividends:

$$ r_t = \frac{AdjClose_t - AdjClose_{t-1}}{AdjClose_{t-1}} $$

Implementation uses `pandas.Series.pct_change()`, which is mathematically equivalent and well-tested.

- First return value is NaN
- Raw data is never modified in place

## 2. Return Z-Score (ret_z)

The return z-score measures how extreme today’s return is relative to recent history:

$$ ret\_z_t = \frac{r_t - \mu(r_{t-63:t-1})}{\sigma(r_{t-63:t-1})} $$

Key design decisions:

- Numerator uses today’s return
- Mean and standard deviation are computed from past data only
- Rolling windows are explicitly shifted to avoid leakage
- If rolling standard deviation is zero, the z-score is set to NaN

This makes the feature:

- leakage-safe
- regime-adaptive
- numerically stable

## 3. Volume Z-Score (vol_z)

Volume anomalies are detected using the z-score of log(volume):
$$
vol\_z_t = \frac{\log(V_t) - \mu(\log(V_{t-21:t-1}))}{\sigma(\log(V_{t-21:t-1}))}
$$


Additional safeguards:

- Zero volumes are converted to NaN before applying log
- Only past observations contribute to rolling statistics
- Standard deviation equal to zero yields NaN
- Using log-volume reduces skewness and improves robustness

## 4. Intraday Range Percentile (range_pct)

Intraday range is defined as:

$$ range_t = \frac{High_t - Low_t}{Close_t} $$

Instead of a z-score, a non-parametric percentile is used:

$$ range\_pct_t = \frac{1}{63} \sum_{k=t-63}^{t-1} 1(range_k \leq range_t) $$

Rationale:

- Intraday range is heavy-tailed and non-Gaussian
- Percentiles are more robust across volatility regimes
- Output is naturally bounded in \([0, 1]\)

This feature highlights unusually wide trading ranges without distributional assumptions.

## Warm-Up Logic

A minimum of 63 past observations is required before any feature is considered valid:

$$ min\_obs = \max(63, 21, 63) = 63 $$

Before this warm-up period:

- All feature values are set to NaN
- No anomalies can be flagged

This ensures statistical reliability.

## Testing & Validation

All feature functions are covered by `pytest` unit tests using synthetic data to verify:

- Correct handling of warm-up periods
- Absence of future data leakage
- Proper behavior when rolling variance is zero
- Detection of extreme spikes when variance exists
- Correct percentile computation for intraday range

Leakage tests explicitly confirm that adding future spikes does not affect past feature values.

## Design Principles Followed

- Feature functions are pure and deterministic
- Minimal input contracts (Series instead of full DataFrames where possible)
- No I/O inside feature modules
- Numerical edge cases handled explicitly
- Behavior validated through tests, not assumptions

## Outcome

At the end of Step 2:

- A compact, interpretable, and leakage-safe feature matrix is available
- Features are robust across calm and volatile regimes
- The system is ready for rule-based anomaly detection