import pandas as pd

def show_dataset(path = 'feature_store/data/customer_features.parquet'):
    df = pd.read_parquet(path)
    print(df.head())
    print(df["event_timestamp"].min(), df["event_timestamp"].max())
    print(df["customerId"].nunique())

if __name__ == "__main__":
    show_dataset()
