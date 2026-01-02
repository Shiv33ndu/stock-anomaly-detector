import pandas as pd
import argparse
from pathlib import Path

def monthly_report(month):
    
    ROOT_PATH = Path(__file__).parent.parent.parent
    DATA = Path(ROOT_PATH / "data/processed")

    daily = pd.read_csv( DATA / "daily_anomaly_card.csv", parse_dates=["date"])
    market = pd.read_csv(DATA / "market_day_table.csv", parse_dates=["date"])

    df = (
        daily
        .merge(market[["date", "market_anomaly_flag"]], on="date", how="left")
        .query("date.dt.strftime('%Y-%m') == @month")
    )

    return df[
        ["date", "ticker", "type", "ret_z", "vol_z",
         "market_anomaly_flag", "why"]
    ].sort_values("date")


def main():
    parser = argparse.ArgumentParser(
        description="Generate monthly anomaly mini-report"
    )
    parser.add_argument(
        "--month",
        required=True,
        help="Month in YYYY-MM format (e.g., 2020-02)"
    )

    args = parser.parse_args()
    month = args.month

    df_report = monthly_report(month)

    if df_report.empty:
        print(f"No anomalies found for month {month}")
        return

    print(f"\n=== Monthly Anomaly Report: {month} ===")
    print(df_report.to_string(index=False))


if __name__ == "__main__":
    main()