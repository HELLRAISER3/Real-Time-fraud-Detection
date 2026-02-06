import json
import pandas as pd

RAW_PATH = "data/raw/transactions.txt"
FEATURES_PATH = "data/feature_store/customer_features.parquet"
OUTPUT_PATH = "data/training/training_dataset.parquet"
OUTPUT_PATH_CSV = "data/training/training_dataset.csv"


rows = []
with open(RAW_PATH) as f:
    for line in f:
        rows.append(json.loads(line))

tx = pd.DataFrame(rows)
tx["transactionDateTime"] = pd.to_datetime(tx["transactionDateTime"])

tx.rename(
    columns={
        "customerId": "customer_id",
        "transactionDateTime": "event_timestamp",
        "isFraud": "label",
    },
    inplace=True,
)

features = pd.read_parquet(FEATURES_PATH)

training_df = tx.merge(
    features,
    on="customer_id",
    how="left",
    suffixes=("", "_hist"),
)

training_df = training_df[
    [
        "transactionAmount",
        "cardPresent",
        "txn_count_30d",
        "avg_amount_30d",
        "max_amount_30d",
        "card_present_ratio",
        "fraud_rate",
        "label",
    ]
]

training_df.to_parquet(OUTPUT_PATH, index=False)
training_df.to_csv(OUTPUT_PATH_CSV, index=False)


