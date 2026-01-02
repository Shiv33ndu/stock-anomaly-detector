from pathlib import Path
import pandas as pd

from src.reporting.daily_card import build_daily_anomaly_card
from src.market.aggregate import build_market_day_table


def main():
    
    ROOT_PATH = Path(__file__).parent.parent.parent
    DATA = Path(ROOT_PATH / "data/processed")

    # Load inputs
    df_features = pd.read_csv(DATA / "features.csv", parse_dates=["date"])
    df_rule = pd.read_csv(DATA / "anomalies_rule_test.csv", parse_dates=["date"])
    df_kmeans = pd.read_csv(DATA / "anomalies_kmeans_test.csv", parse_dates=["date"])
    df_dbscan = pd.read_csv(DATA / "anomalies_dbscan_test.csv", parse_dates=["date"])

    # Build Daily Anomaly Card
    daily_card = build_daily_anomaly_card(
        df_features=df_features,
        df_rule=df_rule,
        df_kmeans=df_kmeans,
        df_dbscan=df_dbscan
    )

    daily_card.to_csv(
        DATA / "daily_anomaly_card.csv",
        index=False
    )

    # Build Market Day Table
    market_table = build_market_day_table(
        df_features=df_features,
        df_anomalies=daily_card
    )

    market_table.to_csv(
        DATA / "market_day_table.csv",
        index=False
    )

    print("✔ Walkforward completed")
    print("✔ Generated daily_anomaly_card.csv")
    print("✔ Generated market_day_table.csv")


if __name__ == "__main__":
    main()