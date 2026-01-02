# def build_daily_anomaly_card(
#     df_features,
#     df_rule,
#     df_kmeans,
#     df_dbscan
# ):
#     # --- Normalize merge keys ---
#     for df in [df_features, df_rule, df_kmeans, df_dbscan]:
#         if "ticker" in df.columns:
#             df["ticker"] = df["ticker"].astype(str)

#     df = (
#         df_rule
#         .merge(
#             df_features,
#             on=["date", "ticker"],
#             how="left"
#         )
#         .merge(
#             df_kmeans[["date", "ticker", "distance"]],
#             on=["date", "ticker"],
#             how="left"
#         )
#         .merge(
#             df_dbscan[["date", "ticker"]]
#                 .assign(dbscan_flag=1),
#             on=["date", "ticker"],
#             how="left"
#         )
#     )

#     df["dbscan_flag"] = df["dbscan_flag"].fillna(0).astype(int)

#     return df[
#         ["date", "ticker", "anomaly_flag", "type",
#          "ret", "ret_z", "vol_z", "range_pct", "why"]
#     ]

import pandas as pd

def build_daily_anomaly_card(
    df_features: pd.DataFrame,
    df_rule: pd.DataFrame,
    df_kmeans: pd.DataFrame,
    df_dbscan: pd.DataFrame
):
    # Normalize ticker dtype
    for df in [df_features, df_rule, df_kmeans, df_dbscan]:
        if "ticker" in df.columns:
            df["ticker"] = df["ticker"].astype(str)

    # Start from rule-based anomalies (structure + why)
    df = df_rule[
        ["date", "ticker", "anomaly_flag", "type", "why"]
    ].copy()

    # Join FEATURES (single source of truth)
    df = df.merge(
        df_features[
            ["date", "ticker", "ret", "ret_z", "vol_z", "range_pct"]
        ],
        on=["date", "ticker"],
        how="left"
    )

    # Join K-Means distance (optional enrichment)
    if "distance" in df_kmeans.columns:
        df = df.merge(
            df_kmeans[["date", "ticker", "distance"]],
            on=["date", "ticker"],
            how="left"
        )

    # Join DBSCAN flag
    df = df.merge(
        df_dbscan[["date", "ticker"]]
            .assign(dbscan_flag=1),
        on=["date", "ticker"],
        how="left"
    )

    df["dbscan_flag"] = df["dbscan_flag"].fillna(0).astype(int)

    return df
