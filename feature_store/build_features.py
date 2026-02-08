import pandas as pd
import numpy as np
from pathlib import Path


RAW_PATH = "data/raw/transactions.csv" 
OUTPUT_PATH = "data/feature_store/customer_features.parquet"
OUTPUT_PATH_CSV = "data/feature_store/customer_features.csv"

def load_data(path):
    path = Path(path)
    print(path.absolute())
    return pd.read_csv(path)

def engineer_features(df):

    features = pd.DataFrame()
    features['SK_ID_CURR'] = df['SK_ID_CURR']
    
    features['credit_to_income_ratio'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']
    
    features['annuity_to_income_ratio'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    
    features['credit_term_approx'] = df['AMT_CREDIT'] / df['AMT_ANNUITY']
    
    features['goods_price_to_credit_ratio'] = df['AMT_GOODS_PRICE'] / df['AMT_CREDIT']

    features['age_years'] = df['DAYS_BIRTH'] / -365.25
    
    features['years_employed'] = df['DAYS_EMPLOYED'].replace(365243, np.nan) / -365.25
    
    features['employed_to_age_ratio'] = features['years_employed'] / features['age_years']
    
    features['income_per_person'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']

    features['ext_source_1'] = df['EXT_SOURCE_1']
    features['ext_source_2'] = df['EXT_SOURCE_2']
    features['ext_source_3'] = df['EXT_SOURCE_3']
    
    features['ext_source_mean'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis=1)

    features['flag_own_car'] = df['FLAG_OWN_CAR'].apply(lambda x: 1 if x == 'Y' else 0)
    features['flag_own_realty'] = df['FLAG_OWN_REALTY'].apply(lambda x: 1 if x == 'Y' else 0)
    
    return features

if __name__ == "__main__":
    df_raw = load_data(RAW_PATH)
    
    df_features = engineer_features(df_raw)
    
    print(f"Saving {len(df_features)} rows to feature store...")

    df_features['event_timestamp'] = pd.Timestamp.now()
    df_features['created_timestamp'] = pd.Timestamp.now()

    df_features.to_parquet(OUTPUT_PATH, index=False)
    df_features.to_csv(OUTPUT_PATH_CSV, index=False)