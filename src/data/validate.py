import pandas as pd
import numpy as np
from pathlib import Path

from src.data.load import load_ticker_csv, REQUIRED_COL



def validate_ohlcv(df: pd.DataFrame, strict: bool= False):
    """
    This method validates the ticker's loaded 
    data received from load_ticker_csv or a preloaded DataFrame. 
    
    :param df: preloded DataFrame or received from load_ticker_csv
    :type df: pd.DataFrame

    returns
        In strict mode, raises on first violation; 
        otherwise returns (df, issues).
    """

    if df.empty:
        raise ValueError(f"[src.data.validate]: ❌ Dataframe is empty!")

    # check for duplicate dates
    if df['date'].duplicated().sum() > 0:
        dups = df.loc[df['date'].duplicated(keep=False)]
        raise ValueError(f"[src.data.validate]: ❌ Dataset has duplicate values\n{dups}")
    
    
    # check for unsorted dates
    if not df["date"].is_monotonic_increasing:
        raise ValueError(f"[src.data.validate]: ❌ Date is not sorted in ascending order!")

    # check for missing required column
    missing_col = [col for col in REQUIRED_COL if col not in df.columns]

    if missing_col:
        raise ValueError(f"[src.data.validate]: ❌ Missing required columns : {missing_col}")     
    # Financial Sanity checks
    rows_with_issue = []
    # checking for negative volumne
    volume_errors = df[df['volume'] < 0]
    if not volume_errors.empty:
        if strict:
            raise ValueError(f"[src.data.validate]: ❌ Negative Volume is present!\n{volume_errors}")
        else:
            for idx, row in volume_errors.iterrows():
                err = {
                    'idx': idx,
                    'date': row['date'],
                    'reason': "Negative Volume"
                }
                rows_with_issue.append(err)

    # checking for High < max(Open, Close, Low)
    high_errors = df[df['high'] < df[['open', 'close', 'low']].max(axis=1)]
    if not high_errors.empty:
        if strict:
            raise ValueError(f"[src.data.validate]: ❌ High is less than max(open, close, low)\n{high_errors}")
        else:
            for idx, row in high_errors.iterrows():
                err = {
                    'idx': idx,
                    'date': row['date'],
                    'reason': "High < max(open, close, low)"
                }
                rows_with_issue.append(err)

    # check for low > min(open, close)
    low_errors = df[df['low'] > df[['open', 'close']].min(axis=1)]
    if not low_errors.empty:
        if strict:
            raise ValueError(f"[src.data.validate]: ❌ Low is greater than min(open, close)\n{low_errors}")
        else:
            for idx, row in low_errors.iterrows():
                err = {
                    'idx': idx,
                    'date': row['date'],
                    'reason': "Low > min(open, close)"
                }
                rows_with_issue.append(err)

    return df, rows_with_issue


# example test
# ROOT_DIR = Path(__file__).parent.parent.parent
# ticker_name = "QQQ"
# file_path = ROOT_DIR / "data" / "raw" / f"{ticker_name}.csv"

# df = load_ticker_csv(file_path)
# validated_df, issues = validate_ohlcv(df)

# if issues:
#     print(issues)