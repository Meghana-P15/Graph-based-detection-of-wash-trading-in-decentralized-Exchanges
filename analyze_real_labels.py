import pandas as pd
import json
BASE = "data/processed_real_1000blocks"
features = pd.read_csv(
    f"{BASE}/features_and_weak_labels.csv"
)
trades = pd.read_csv(f"{BASE}/trades.csv")
df = features.merge(
    trades[["event_id", "pool", "timestamp"]],
    on="event_id",
    how="left",
    validate="one_to_one",
    suffixes=("", "_trade")
)
time_column = (
    "timestamp" if "timestamp" in df.columns
    else "timestamp_trade"
)
df["event_time"] = pd.to_datetime(
    df[time_column], utc=True
)
df = df.sort_values(
    ["event_time", "event_id"]
).reset_index(drop=True)
print("\n= DATASET SUMMARY =")
print("Total records:", len(df))
print("Unique pools:", df["pool"].nunique())
print("Weak labels:")
print(df["weak_label"].value_counts().sort_index())
print("\n= CHRONOLOGICAL SPLIT =")
n = len(df)
a = int(n * 0.6)
b = int(n * 0.8)
for name, part in [
    ("Training", df.iloc[:a]),
    ("Validation", df.iloc[a:b]),
    ("Testing", df.iloc[b:])
]:
    print(
        name,
        "Rows:", len(part),
        "Labels:",
        part["weak_label"].value_counts().to_dict()
    )
print("\n= POSITIVE REASONS =")
positive = df[df["weak_label"] == 1]
for reason in [
    "activity_burst_1h",
    "repeated_pair_1h",
    "reverse_direction_1h",
    "same_pool_multiwallet_5m"
]:
    count = positive["label_reason_codes"].apply(
        lambda x: reason in json.loads(x)
    ).sum()
print(f"{reason}: {count}")
print("\n= POOL-WISE LABEL PERCENTAGES =")
counts = pd.crosstab(
    df["pool"], df["weak_label"]
)
percentages = counts.div(
    counts.sum(axis=1), axis=0
) * 100
print(percentages.round(2))
print("\n= AUDIT COMPLETE =")