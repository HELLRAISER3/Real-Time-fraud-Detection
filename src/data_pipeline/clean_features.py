import pandas as pd
import numpy as np

df = pd.read_parquet("data/training/training_dataset.parquet")

columns_to_check = [col for col in df.columns if col not in ['SK_ID_CURR', 'label']]

duplicate_groups = {}
checked = set()

for i, col1 in enumerate(columns_to_check):
    if col1 in checked:
        continue

    duplicates = [col1]
    for col2 in columns_to_check[i+1:]:
        if col2 in checked:
            continue

        if df[col1].equals(df[col2]):
            duplicates.append(col2)
            checked.add(col2)

    if len(duplicates) > 1:
        duplicate_groups[col1] = duplicates
        for dup in duplicates:
            checked.add(dup)

if duplicate_groups:
    print(f"\nFound {len(duplicate_groups)} groups of duplicate columns:")

    columns_to_drop = []
    for keep, dups in duplicate_groups.items():
        columns_to_drop.extend([d for d in dups if d != keep])

    df_clean = df.drop(columns=columns_to_drop)
else:
    df_clean = df.copy()


df_clean.to_parquet("data/training/training_dataset_clean.parquet", index=False)
df_clean.to_csv("data/training/training_dataset_clean.csv", index=False)


print("\n" + "="*60)
print("Running feature selection on clean data...")
print("="*60)

y = df_clean['label']
X = df_clean.drop(['SK_ID_CURR', 'label'], axis=1)

correlations = X.corrwith(y).abs().sort_values(ascending=False)

print(f"\nTop 30 features by correlation with target:")
for i, (feat, corr) in enumerate(correlations.head(30).items(), 1):
    print(f"{i:2d}. {feat:45s} {corr:.6f}")

n_features = 20
selected_features = correlations.head(n_features).index.tolist()

print(f"\n{'='*60}")
print(f"Selected TOP {n_features} FEATURES:")
print(f"{'='*60}")
for i, feat in enumerate(selected_features, 1):
    print(f"{i:2d}. {feat}")

df_final = df_clean[['SK_ID_CURR', 'label'] + selected_features]

df_final.to_parquet("data/training/training_dataset_final.parquet", index=False)
df_final.to_csv("data/training/training_dataset_final.csv", index=False)

with open("data/training/selected_features_final.txt", 'w') as f:
    for feat in selected_features:
        f.write(f"{feat}\n")
