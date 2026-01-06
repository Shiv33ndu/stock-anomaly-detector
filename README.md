# Stock Market Anomaly Detection

Detect unusual market days and stock days using only daily price and volume data. “Unusual” means very large moves or spikes or dips, unusually high volume, or unusually wide trading ranges compared to recent history.

This project implements unsupervised anomaly detection on financial time-series data using:

- rule-based heuristics,
- distance-based clustering (K-Means),
- density-based clustering (DBSCAN),

without using labels, news, sentiment, or external information.

The goal is to identify abnormal market behavior, highlight stress periods, and provide explainable anomaly reports at both the stock level and market level.

## 📖 Motivation & Key Ideas

Financial markets rarely provide labeled “anomaly” data. Instead of supervised learning, this project relies on:

- rolling statistics and z-scores,
- regime deviation via clustering,
- structural isolation via density estimation,
- walk-forward validation to avoid look-ahead bias.

The system is designed to be:

- label-free
- explainable
- time-aware
- production-oriented (batch serving + query interface)

---

# Project Explanation


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

## Validation Set Flag Rate

On the 2019 validation set, we calibrated the K-Means detector using:
- ( k = 3 ) (chosen via elbow and silhouette on training data)
- ( q = 95 ) (percentile threshold tuned on validation).

The resulting ticker-day–level anomaly flag rate is 5.5%, which lies within the 2–8% target range specified in the project guidelines.

Refer to the section `**I: Rule-Based vs K-Means Comparison & Flag Rate (Validation Set)**` in the notebook `notebooks/04_kmeans_analysis.ipynb` for details.

## Numerical Summary (2019 Validation Set):

- Rule-based anomalous dates: 75
- K-Means anomalous dates: 72
- Overlapping anomalous dates: 34
- Total ticker-day points (validation): 1763
- Anomalous ticker-day points (K-Means): 97
- Validation flag rate (point-level): 5.5%


---

### Rule-based vs K-Means Anomaly Dates (2020 Q1)

Axis:

* Y-axis: Number of Dates
<p align="center">
<img src="reports\figures\rule_vs_Kmeans_anomaly.png"/>
</p>

During the 2020 Q1 stress period, the rule-based detector flagged 43 anomalous days, while the K-Means detector flagged 22 regime-level anomalies. 19 days were identified by both methods, indicating strong agreement on the most severe market stress events.


---

# Step 5: DBSCAN (Density-Based Anomaly Detection)

In this step, we implemented a density-based anomaly detector using DBSCAN, providing a complementary perspective to rule-based and K-Means approaches.

## Methodology

- **Features used**: standardized
  - return z-score (ret_z)
  - volume z-score (vol_z)
  - intraday range percentile (range_pct)

- **min_samples**: was set to 7 using the heuristic $2 \times (\text{number of features}) + 1$

- **eps**: was selected as 0.7 based on a k-distance elbow plot on the training set

- **Anomaly Treatment**: Points labeled as noise (\(-1\)) by DBSCAN were treated as anomalies

- **Temporal Ordering**: To respect temporal ordering, DBSCAN was refit on an expanding window:
  - **Validation (2019)**: trained on training data
  - **Test (2020 Q1)**: trained on training + validation data
  - **Labels**: were extracted only for the newly scored period

## Anomaly Rates

| Split          | Point-level anomaly rate |
|----------------|--------------------------|
| Training       | 1.59%                    |
| Validation (2019) | 1.19%                    |
| Test (2020 Q1)  | 1.84%                    |

DBSCAN produces fewer anomalies than K-Means and rule-based detectors, reflecting its conservative, density-based nature. The increase in anomaly rate during the 2020 Q1 stress period indicates a breakdown of historical density structure.

## Comparison with Other Detectors

- **Rule-based detection**: flags many anomalies due to fixed thresholds, leading to high sensitivity but lower precision.
- **K-Means detection**: identifies regime-level deviations and achieves a balanced anomaly rate.
- **DBSCAN detection**: flags only structurally isolated points, producing the lowest anomaly rate but highest confidence anomalies.

Overlap analysis shows that days flagged by all three detectors correspond to extreme market stress events, providing strong consensus signals.

## Comparison Plots with Other Detectors

* #### Anomalous Dates by Detectors (2020 Q1)
<p>
<img src="reports\figures\rule_vs_kmeans_vs_dbscan.png" />
</p>

* #### Daily Anomly Counts by Detectors (2020 Q1)
<p>
<img src="reports\figures\daily_anomaly_count.png" />
</p>

* #### Point-level Flag Rate (2020 Q1)
<p>
<img src="reports\figures\flag_rate.png" />
</p> 

## Key Takeaway

DBSCAN complements K-Means by identifying anomalies that lie outside any dense region of normal behavior. Together, the three detectors provide a multi-perspective view of market anomalies, balancing sensitivity and robustness.

---

### Hybrid and Consensus Detection (Optional)

We combine K-Means and DBSCAN anomaly signals to form consensus detectors.  
Two strategies are considered:

- **Union**: flags a point as anomalous if either K-Means or DBSCAN flags it, yielding higher recall.
- **Intersection**: flags a point as anomalous only if both detectors agree, yielding higher precision.

The union strategy increases sensitivity during stress periods, while the intersection strategy produces a small set of high-confidence anomalies corresponding to structurally isolated regime deviations.

---

### Market-Level Aggregation

We aggregate anomaly signals across tickers to form daily market-level indicators.  

The daily anomaly flag rate captures the fraction of assets exhibiting abnormal behavior and serves as a simple measure of market stress. We also report market return and breadth to contextualize anomaly signals. During the 2020 Q1 period, elevated flag rates coincide with negative market returns and reduced breadth, reflecting systemic stress.

---

## Project Structure

```
└── 📁stock-anomaly-detector
    └── 📁data
        └── 📁processed                         # processed data : features, daily_reports etc..
        └── 📁raw                               # chosen tickers, a small stock universe
    └── 📁notebooks
        ├── 01_eda.ipynb                        # eda and exploratory plots
        ├── 02_feature_sanity.ipynb             # checking the feature hygienes, leakage, rolling window logic
        ├── 03_rule_based_analysis.ipynb        # rule base anomaly detection and plot narrative
        ├── 04_kmeans_analysis.ipynb            # kmeans anomaly detection and comparison with rule_based baseline
        ├── 05_dbscan_analysis.ipynb            # dbcan expanding window check, anomaly detection & comparison with Kmeans and baseline
        ├── 06_consensus.ipynb                  # hybrid & consensus, market daily breadth, flag rate
    └── 📁src                         
        ├── __main__.py                         # entry point for cmd queries
        └── 📁cli
            ├── __init__.py
            ├── monthly.py                      # monthly report 
            ├── query.py                        # date query
            ├── walkforward.py                  # batch process, csv persistence logic
        └── 📁data
            ├── load.py                         # ticker data loading, sorting, columns' case consistency  
            ├── validate.py                     # validation layer 
            ├── clean_ohlcv.py                  # financial sanity related cleaning
        └── 📁detectors
            ├── __init__.py
            ├── dbscan.py                       # density based unsupervised anomaly detector
            ├── kmeans.py                       # distance based unsupervised anomaly detector
            ├── rules.py                        # rule-based anomaly detector
        └── 📁evaluation
            ├── __init__.py
            ├── metrics.py                      # metric related logic
        └── 📁features
            ├── __init__.py
            ├── range.py                        # range_pct feature logic
            ├── returns.py                      # return, ret_z related feature logic
            ├── volume.py                       # log(vol) and vol_z related feature logic
        └── 📁market
            ├── __init__.py
            ├── aggregate.py                    # monthly aggregation logic
        └── 📁reporting
            ├── __init__.py
            ├── daily_card.py                   # daily card csv logic
            ├── monthly_report.py               # monthly report csv logic
        ├── __init__.py
    └── 📁tests                                 # test codes 
        └── 📁fixtures                          # dummy csvs for testing 
            ├── bad_date.csv
            ├── bad_ohlc.csv
            ├── duplicate_dates.csv
            ├── missing_column.csv
            ├── negative_volume.csv
            ├── unsorted_dates.csv
            ├── valid.csv
        ├── __init__.py
        ├── conftest.py
        ├── test_detectors.py                   
        ├── test_features.py
        ├── test_kmeans_detector.py
        ├── test_load.py
        └── test_validate.py
    ├── .gitignore
    ├── README.md
    └── requirements.txt
```

---

# Project Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Shiv33ndu/stock-anomaly-detector
cd stock-anomaly-detector
```

### 2. Create Virtual Environment (Recommended)

```bash
python -m venv venv
```
Activate:
* **Windows**
```bash
venv\Scripts\activate
```
* **Linux / MacOS**
```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Dataset Preparation
Download the dataset from Kaggle:

https://www.kaggle.com/datasets/jacksoncrow/stock-market-dataset

Place selected tickers CSVc into:
```bash
data/raw/
```

Example tickers used:
```bash
QQQ, AMZN, GOOGL, MSFT, NVDA, TSLA, META
```

---

## How to Run (Quick Start)

The project follows a **batch-then-serve** workflow.

### Step 1: Generate All Final Outputs(walkforward)
Run the full pipeline and generate serving artifacts
```bash
python -m src walkforward
```

This command:
* builds final anomaly tables,
* generates daily and market-level CSVs,
* materializes all outputs used by query & reporting.

Generated Files:
```bash
data/processed/
├── daily_anomaly_card.csv
├── market_day_table.csv
```

### Step 2: Query a Specific Date
```bash
python -m src query --date 2020-02-27
```

Output:
* market status,
* anomalous tickers,
* explanation for each anomaly.

### Step 3: Generate Monthly Mini-Report
```bash
python -m src monthly --month 2020-02
```
Output:

* abnormal dates,
* flagged tickers,
* return & volume statistics,
* market stress flag.


### Help Command
```python
python -m src help
```
Output: 

* lists all the available commands
* with command syntax

---

## Running Tests 
Run all unit tests:
```python
pytest -s tests/
```
Tests include:
* feature correctness
* leakage prevention
* detector sanity checks.

---

## Output Produced
**A. Daily Anomaly Card**
```pgsql
date, ticker, anomaly_flag, type, ret, ret_z, vol_z, range_pct, why
```

**B. Market Day Table**
```bash
date, market_ret, breadth, market_anomaly_flag
```

**C. Query Interface**
- per-date anomaly lookup

**D. Monthly Mini-Report**
- summarized anomaly events


## Key Takeaways
- Rule-based detectors provide high recall but over-flag.
- K-Means captures regime-level deviations.
- DBSCAN isolates structurally abnormal points.
- Market-level aggregation converts micro anomalies into macro stress signals.
- Together, these form a robust, interpretable anomaly detection system for financial time series.

---