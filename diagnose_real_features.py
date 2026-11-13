import pandas as pd
from src.features import load_processed, prepare_features
BASE = "data/processed_real_1000blocks"
trades, transfers = load_processed(BASE)
print("\n= INPUT DATA =")
print("Total transactions:", len(trades))
print("Unique pools:", trades["pool"].nunique())
print("Missing pools:", trades["pool"].isna().sum())
print("Empty pools:", (trades["pool"] == "").sum())
print("\nPool counts:")
print(trades["pool"].value_counts().head())
print("\n= TIMESTAMP CHECK =")
trades["timestamp"] = pd.to_datetime(
    trades["timestamp"], utc=True
)
print("First:", trades["timestamp"].min())
print("Last:", trades["timestamp"].max())

ordered = trades.sort_values(
    ["pool", "timestamp", "event_id"]
).copy()
ordered["previous_time"] = (
    ordered.groupby("pool")["timestamp"].shift(1)
)
ordered["minutes_since_previous"] = (
    ordered["timestamp"] - ordered["previous_time"]
).dt.total_seconds() / 60
print(
    "Events with prior same-pool event within 60 min:",
    int(
        ordered["minutes_since_previous"]
        .between(0, 60)
        .sum()
    )
)
print("\n= FRESH FEATURE CALCULATION =")
features = prepare_features(trades, transfers)
for column in [
    "prior_actor_1h",
    "prior_pair_1h",
    "prior_pool_1h",
    "prior_reverse_1h",
    "prior_other_wallets_pool_5m"
]:
    print(
        column,
        "Nonzero:", int((features[column] > 0).sum()),
        "Maximum:", features[column].max()
    )
print("\nWeak labels:")
print(features["weak_label"].value_counts().to_dict())
print("\n= FIRST 5 FEATURE ROWS =")
print(
    features[
        [
            "event_id",
            "prior_actor_1h",
            "prior_pool_1h",
            "prior_reverse_1h"
        ]
    ].head().to_string(index=False)
)