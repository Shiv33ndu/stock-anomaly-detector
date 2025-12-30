import pandas as pd
from pathlib import Path
import logging

# logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

REQUIRED_COL = ["date", "open", "high", "low", "close", "adj close", "volume"]

def load_ticker_csv(file_path: Path):
    """
    This method will load the raw CSV, read it 
    - parse Date as datetime
    - sort by Date ascending 
    - enforce the columns
    returns
      df: A clean DataFrame
    """

    if not file_path.is_file():
        raise FileNotFoundError(f"[src.data.load]: ❌ File note found: {file_path}")

    df = pd.read_csv(file_path)

    # for case consistency, lower casing the columns
    df.columns = df.columns.str.lower()

    # check datatype of Date column & parse it into Datetime type 
    if df["date"].dtype != "datetime64[ns]":
        df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce")
        if df["date"].isna().any():
            raise ValueError(f"[src.data.load]: ℹ️ Some date couldn't be parsed correctly. Check the input data.")

    # sort the data by Date, ascending
    df = df.sort_values("date", ascending=True)

    # check for required columns
    missing_cols = [col for col in REQUIRED_COL if col not in df.columns] 
    
    if not missing_cols:
        logging.info("[data.load]: ✅ All required columns exist")
    else:
        raise ValueError(f"Missing columns are : {missing_cols}")

    # keeping only required columns
    df = df[REQUIRED_COL]

    return df

# df = load_ticker_csv("FB")
# print(df.head(1))