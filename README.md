## Stock Market Anomaly Detection

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


## Data Loader and Validator

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