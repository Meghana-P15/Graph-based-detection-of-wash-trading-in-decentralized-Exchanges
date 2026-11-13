import json
from collections import Counter
import pandas as pd
BASE = "data/processed_real_1000blocks"
df = pd.read_csv(
    f"{BASE}/features_and_weak_labels.csv"
)
def parse_reasons(value):
    if pd.isna(value):
        return []
    try:
        result = json.loads(value)
        return result if isinstance(result, list) else []
    except (ValueError, TypeError):
        return []
df["parsed_reasons"] = df["label_reason_codes"].apply(
    parse_reasons
)
positive = df[df["weak_label"] == 1].copy()
counts = Counter(
    reason
    for reasons in positive["parsed_reasons"]
    for reason in reasons
)
print("= POSITIVE EVIDENCE AUDIT =")
print("Positive candidates:", len(positive))
print("\nReason frequencies among positives:")
for reason, count in counts.most_common():
    print(f"{reason}: {count}")
print("\nNumber of reasons per positive:")
print(
    positive["parsed_reasons"]
    .apply(len)
    .value_counts()
    .sort_index()
    .to_string()
)
pool_only = positive["parsed_reasons"].apply(
    lambda reasons: reasons == [
        "same_pool_multiwallet_5m"
    ]
)
print("\nPositive candidates with only same-pool evidence:",
      int(pool_only.sum()))
print("\n= AUDIT COMPLETE =")