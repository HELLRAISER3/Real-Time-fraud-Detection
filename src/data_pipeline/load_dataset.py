import pandas as pd


def load_dataset(filepath: str = 'data/processed/custom_features.csv', ext: str = 'csv'):
    if ext == 'csv':
        return pd.read_csv(filepath)
    if ext == 'parquet':
        return pd.read_parquet(filepath)
    
    return pd.DataFrame({"Exception" : "Wrong file extension"})