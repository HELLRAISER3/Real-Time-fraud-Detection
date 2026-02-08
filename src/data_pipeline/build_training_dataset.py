import pandas as pd
import numpy as np

RAW_PATH = "data/raw/transactions.csv"
FEATURES_PATH = "data/feature_store/customer_features.parquet"
OUTPUT_PATH = "data/training/training_dataset.parquet"
OUTPUT_PATH_CSV = "data/training/training_dataset.csv"

def run():
    df_raw = pd.read_csv(RAW_PATH)
    features = pd.read_parquet(FEATURES_PATH)
    
    cohort = df_raw.copy()
    cohort.rename(columns={'TARGET': 'label'}, inplace=True)

    cohort['is_cash_loan'] = (cohort['NAME_CONTRACT_TYPE'] == 'Cash loans').astype(int)
    cohort['is_male'] = (cohort['CODE_GENDER'] == 'M').astype(int)
    
    training_df = cohort.merge(
        features,
        on="SK_ID_CURR",
        how="left",
        suffixes=('', '_feat') 
    )

    training_df['goods_to_credit_ratio'] = training_df['AMT_GOODS_PRICE'] / training_df['AMT_CREDIT']
    
    training_df['children_ratio'] = training_df['CNT_CHILDREN'] / training_df['CNT_FAM_MEMBERS']
    
    training_df['employed_to_age_ratio'] = training_df['DAYS_EMPLOYED'] / training_df['DAYS_BIRTH']
    
    training_df['ext_sources_sum'] = training_df[['ext_source_1', 'ext_source_2', 'ext_source_3']].sum(axis=1)
    training_df['ext_sources_prod'] = training_df['ext_source_1'] * training_df['ext_source_2'] * training_df['ext_source_3']

    final_columns = training_df.select_dtypes(include=[np.number]).columns.tolist()
    
    if 'TARGET' in final_columns: final_columns.remove('TARGET')
    
    training_df = training_df[final_columns]

    training_df = training_df.replace([np.inf, -np.inf], np.nan)

    print(f"Final training set shape: {training_df.shape}")

    training_df.to_parquet(OUTPUT_PATH, index=False)
    training_df.to_csv(OUTPUT_PATH_CSV, index=False)

if __name__ == "__main__":
    run()