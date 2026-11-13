import pandas as pd
import json
from collections import Counter
path = "data/processed_real_1000blocks/features_and_weak_labels.csv"
df = pd.read_csv(path)
print("Total events:", len(df))
print("\nWeak-label distribution:")
print(df["weak_label"].value_counts(dropna=False))
reason_counts = Counter()
for value in df["label_reason_codes"].fillna("[]"):
    try:
        reasons = json.loads(value)
        reason_counts.update(reasons)
    except (ValueError, TypeError):
        pass
print("\nIndividual evidence frequencies:")
for reason, count in reason_counts.most_common():
    print(f"{reason}: {count}")
single_reason = df[
    df["label_reason_codes"].fillna("") ==
    '["same_pool_multiwallet_5m"]'
]
print("\nSame-pool indicator only:")
print(single_reason["weak_label"].value_counts(dropna=False))
print("\nWeak labels per pool:")
trades = pd.read_csv(
    "data/processed_real_1000blocks/trades.csv"
)
merged = df.merge(
    trades[["event_id", "pool"]],
    on="event_id",
    how="left",
    validate="one_to_one"
)
print(
    pd.crosstab(
        merged["pool"],
        merged["weak_label"]
    )
)
print("\nUnmatched pool records:")
print(merged["pool"].isna().sum())
