def build_market_day_table(df_features, df_anomalies, flag_thresh=0.05):
    n_tickers = df_features["ticker"].nunique()

    market = (
        df_features
        .groupby("date")
        .agg(
            market_ret=("ret", "mean"),
            breadth=("ret", lambda x: (x > 0).mean())
        )
    )

    flag_rate = (
        df_anomalies
        .groupby("date")["ticker"]
        .nunique()
        .div(n_tickers)
        .rename("flag_rate")
    )

    market = market.join(flag_rate, how="left").fillna(0)
    market["market_anomaly_flag"] = (market["flag_rate"] > flag_thresh).astype(int)

    return market.reset_index()
