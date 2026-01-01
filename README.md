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

$$ 
retz_{t} = \frac{r_t - \mu(r_{t-63:t-1})}{\sigma(r_{t-63:t-1})} $$

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
volz_{t} = \frac{\log(V_t) - \mu(\log(V_{t-21:t-1}))}{\sigma(\log(V_{t-21:t-1}))}
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

---

## Step 3: Rule-Based Anomaly Detection (Interpretable Baseline)

This step implements a simple, fully interpretable rule-based anomaly detector that flags unusual stock-day behavior using the leakage-safe features created in Step 2.

This detector serves as:

- a baseline anomaly system
- an explainability anchor for later ML-based detectors
- a sanity check during evaluation and debugging

All rules and thresholds are implemented exactly as specified in the project guidelines.

### Inputs

For each ticker and trading day, the detector consumes the following precomputed features:

| Feature   | Description                     |
|-----------|---------------------------------|
| ret       | Daily return (Adj Close–based)  |
| ret_z     | Z-score of daily return         |
| vol_z     | Z-score of log(volume)          |
| range_pct | Intraday range percentile       |

Days with missing feature values (warm-up period) are explicitly ignored.

### Rule Definitions

A stock-day anomaly is flagged if any of the following conditions hold:

- |ret_z| &gt; 2.5

- vol_z &gt; 2.5

- range_pct &gt; 0.95

No learning or tuning is involved at this stage—these rules are intentionally simple and interpretable.

### Anomaly Type Classification

If an anomaly is detected, it is assigned one or more type labels based on the triggering conditions:

| Condition                                      | Type           |
|------------------------------------------------|----------------|
| ret &lt; 0 and ($abs(ret_z) \gt 2.5$)                  | crash          |
| ret &gt; 0 and ($abs(ret_z) \gt 2.5$)                  | spike          |
| $vol_z \gt 2.5$                               | volume_shock   |

Notes:

- Multiple types can co-exist (e.g., crash + volume_shock)
- Intraday range (range_pct) contributes to anomaly detection but does not introduce a new type

### Explainability (why Field)

Each anomaly includes a human-readable explanation listing the rule(s) that fired, for example:

|ret_z| &gt; 2.5; vol_z &gt; 2.5

This makes the detector:

- transparent
- easy to debug
- suitable for reporting and diagnostics

### Output Schema (Daily Anomaly Card)

The detector outputs one row per anomalous ticker-day with the following schema:

| Column        | Description                     |
|---------------|---------------------------------|
| date          | Trading date                    |
| ticker        | Stock symbol                    |
| anomaly_flag  | 1 if anomalous                  |
| type          | Anomaly type(s)                 |
| ret           | Daily return                    |
| ret_z         | Return z-score                  |
| vol_z         | Volume z-score                  |
| range_pct     | Intraday range percentile       |
| why           | Triggered rule(s)               |

This structure directly supports downstream aggregation and reporting.

### Warm-Up and Safety Handling

No anomaly is flagged if any required feature is NaN

This prevents false positives during the rolling-window warm-up period

Ensures statistical validity and leakage safety

### Testing & Validation

The rule detector is validated using unit tests with synthetic data, covering:

- crash detection
- spike detection
- volume-only anomalies
- combined anomalies
- intraday range-only anomalies
- proper handling of NaN rows

These tests ensure correctness, interpretability, and robustness.

### Design Principles

- Pure function: no mutation of input DataFrames
- Deterministic behavior
- Explicit rule logic (no hidden heuristics)
- Easy extensibility for future detectors

### Outcome

At the end of Step 3:

- A reliable, interpretable anomaly baseline is in place
- Feature behavior has been validated in real market stress periods
- The system is ready for unsupervised ML-based anomaly detection

---

## Step 4: K-Means–Based Anomaly Detection (Unsupervised)

In this step, we implement an unsupervised anomaly detection method using **K-Means clustering** on behavior-normalized market features.

Unlike the rule-based detector, which relies on fixed thresholds, this approach learns **market behavior regimes directly from data** and flags observations that deviate significantly from these regimes.

---

### Feature Space

Each data point represents one `(ticker, date)` pair and is described by the following features:

- `ret_z` — standardized daily return
- `vol_z` — standardized log-volume
- `range_pct` — intraday range percentile

These features are:
- computed per ticker (rolling statistics)
- comparable across tickers
- leakage-safe by construction

---

### Train / Validation / Test Split

To avoid temporal leakage, the data is split strictly by date:

| Split | Period |
|------|--------|
| Train | 2018 |
| Validation | 2019 |
| Test | 2020 Q1 |

- Feature scaling (StandardScaler) is **fit on training data only**
- K-Means is trained **only on the training split**
- Thresholds are derived **only from training distances**

---

### Choosing K using Elbow & Silhouette Method

The Elbow and Silhouette methods were used on scaled training X.

* Elbow Method performed on $k \in [2, 8)$
<p align="center">
<img src="reports\figures\kmeans_elbow_method.png"/>
</p>

* Silhouette Analysis performed on $k \in [2, 8)$
<p align="center">
<img src="reports\figures\kmeans_silhouette_analysis.png"/>
</p>

**Conclusion**
In Elbow plot, the K=3 is the point from which the inertia value starts plateauing, this kind of stabilization states, any further increment in k value is no good, since too low inertia value will have too densed and higher number of cluster. choosing High value of inertia will lead to less clusters that have point too far from the centroids. k=3 as per the plot seems the sweet spot.

Choosing k=3 decision is further agreed by the silhouette score plot. 
As k=3 has the highest silhouette score, making it the right choice for our anomaly detection problem.

---

### K-Means Training

- Features are standardized using `StandardScaler`
- K-Means is trained with a fixed number of clusters `k` chosen using Elbow and Silhouette Method
- Each observation is assigned to its nearest cluster centroid

---

### Anomaly Scoring

For each observation:
- Compute Euclidean distance to the nearest centroid
- Distance serves as the anomaly score

---

### Cluster-Specific Thresholding

Instead of using a global threshold, we compute **cluster-specific thresholds**:

- For each cluster:
  - collect distances of training points assigned to that cluster
  - threshold = 95th percentile of those distances
- An observation is flagged as anomalous if:
  - distance > threshold[assigned_cluster]
  - This avoids bias toward dense or sparse clusters.

---

### Output Schema

Each detected anomaly contains:

- `date`
- `ticker`
- `anomaly_flag`
- `cluster`
- `distance`

This format enables direct comparison with rule-based anomalies and downstream aggregation.

---

### Key Observations

- Anomaly density increases significantly during market stress periods
- 2020 Q1 shows a clear spike in detected anomalies
- K-Means captures both:
- extreme price/volume shocks
- regime-level behavioral shifts missed by fixed rules

---

### Testing

All major components are unit-tested, including:

- feature scaling
- K-Means fitting
- distance computation
- cluster-specific thresholding
- anomaly detection logic

This ensures correctness, reproducibility, and robustness.

---


### Rule-based vs K-Means Anomaly Dates (2020 Q1)

Axis:

* Y-axis: Number of Dates
<p align="center">
<img src="reports\figures\rule_vs_Kmeans_anomaly.png"/>
</p>

During the 2020 Q1 stress period, the rule-based detector flagged 43 anomalous days, while the K-Means detector flagged 22 regime-level anomalies. 19 days were identified by both methods, indicating strong agreement on the most severe market stress events.