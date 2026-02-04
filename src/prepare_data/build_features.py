import json
import pandas as pd

rows = []
with open("data/raw/transactions.txt") as f:
    for line in f:
        rows.append(json.loads(line))

df = pd.DataFrame(rows)
df["transactionDateTime"] = pd.to_datetime(df["transactionDateTime"])

df = df.sort_values(["customerId", "transactionDateTime"])

features = (
    df.groupby("customerId")
    .agg(
        txn_count_30d=("transactionAmount", "count"),
        avg_amount_30d=("transactionAmount", "mean"),
        max_amount_30d=("transactionAmount", "max"),
        card_present_ratio=("cardPresent", "mean"),
        fraud_rate=("isFraud", "mean"),
        event_timestamp=("transactionDateTime", "max"),
    )
    .reset_index()
)

# features.to_parquet("feature_store/data/customer_features.parquet")
features.to_csv("feature_store/data/customer_features.csv")

