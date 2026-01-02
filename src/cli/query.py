import argparse
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True)
    args = parser.parse_args()

    date = args.date

    daily = pd.read_csv("data/processed/daily_anomaly_card.csv", parse_dates=["date"])
    market = pd.read_csv("data/processed/market_day_table.csv", parse_dates=["date"])

    print("\n=== Market Status ===")
    print(market[market["date"] == date])

    print("\n=== Anomalous Tickers ===")
    print(
        daily[daily["date"] == date][
            ["ticker", "type", "ret_z", "vol_z", "why"]
        ]
    )

if __name__ == "__main__":
    main()
