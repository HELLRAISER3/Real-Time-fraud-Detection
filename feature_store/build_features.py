import json
import pandas as pd

RAW_PATH = "data/raw/transactions.txt"
OUTPUT_PATH = "data/feature_store/customer_features.parquet"
OUTPUT_PATH_CSV = "data/feature_store/customer_features.csv"


rows = []
with open(RAW_PATH) as f:
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

features.rename(columns={"customerId": "customer_id"}, inplace=True)

features.to_parquet(OUTPUT_PATH, index=False)
features.to_csv(OUTPUT_PATH_CSV, index=False)

