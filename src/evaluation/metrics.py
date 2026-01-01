import pandas as pd

def validate_anomaly_df(df: pd.DataFrame):
    """
    Basic validation for anomaly DataFrames.
    """
    required_cols = {"date", "ticker", "anomaly_flag"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def unique_anomaly_dates(df: pd.DataFrame) -> int:
    """
    Returns number of unique dates with anomalies.
    """
    validate_anomaly_df(df)
    return df["date"].nunique()



def anomaly_rate(df_anomalies: pd.DataFrame, df_reference: pd.DataFrame) -> float:
    """
    Fraction of days flagged as anomalous.
    
    df_reference should contain all dates in the period.
    """
    anomaly_days = df_anomalies["date"].nunique()
    total_days = df_reference["date"].nunique()
    return anomaly_days / total_days


def anomaly_overlap_dates(
    df_rule: pd.DataFrame,
    df_kmeans: pd.DataFrame
) -> dict:
    """
    Computes date-level overlap between two detectors.
    """
    rule_dates = set(df_rule["date"])
    kmeans_dates = set(df_kmeans["date"])

    return {
        "rule_only": len(rule_dates - kmeans_dates),
        "kmeans_only": len(kmeans_dates - rule_dates),
        "both": len(rule_dates & kmeans_dates),
    }


def daily_anomaly_counts(df: pd.DataFrame) -> pd.Series:
    """
    Returns anomaly count per date.
    """
    validate_anomaly_df(df)
    return df.groupby("date").size().sort_index()


def ticker_anomaly_counts(df: pd.DataFrame) -> pd.Series:
    """
    Returns anomaly count per ticker.
    """
    validate_anomaly_df(df)
    return df.groupby("ticker").size().sort_values(ascending=False)


def anomaly_summary(df: pd.DataFrame) -> dict:
    """
    High-level summary statistics for anomalies.
    """
    validate_anomaly_df(df)

    return {
        "rows": len(df),
        "unique_dates": df["date"].nunique(),
        "unique_tickers": df["ticker"].nunique(),
        "avg_anomalies_per_day": len(df) / df["date"].nunique()
    }
