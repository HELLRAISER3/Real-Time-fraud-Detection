import pandas as pd

RAW_PATH = "data/raw/transactions.csv"
FEATURES_PATH = "data/feature_store/customer_features.parquet"
OUTPUT_PATH = "data/training/training_dataset.parquet"
OUTPUT_PATH_CSV = "data/training/training_dataset.csv"

def run():
    df_raw = pd.read_csv(RAW_PATH)
    
    cohort = df_raw[[
        'SK_ID_CURR', 
        'TARGET', 
        'NAME_CONTRACT_TYPE', 
        'CODE_GENDER'
    ]].copy()

    cohort['is_cash_loan'] = cohort['NAME_CONTRACT_TYPE'].apply(lambda x: 1 if x == 'Cash loans' else 0)
    cohort['is_male'] = cohort['CODE_GENDER'].apply(lambda x: 1 if x == 'M' else 0)

    cohort.rename(columns={'TARGET': 'label'}, inplace=True)

    features = pd.read_parquet(FEATURES_PATH)

    training_df = cohort.merge(
        features,
        on="SK_ID_CURR",
        how="left"
    )

    selected_columns = [
        'SK_ID_CURR', 'label',
        'is_male', 'age_years', 'years_employed', 'flag_own_car', 'flag_own_realty',
        'credit_to_income_ratio', 
        'annuity_to_income_ratio',
        'income_per_person',
        'ext_source_1', 
        'ext_source_2', 
        'ext_source_3',
        'ext_source_mean'
    ]
    
    training_df = training_df[selected_columns]

    print(f"Saving training set with shape {training_df.shape}...")
    training_df.to_parquet(OUTPUT_PATH, index=False)
    training_df.to_csv(OUTPUT_PATH_CSV, index=False)

if __name__ == "__main__":
    run()