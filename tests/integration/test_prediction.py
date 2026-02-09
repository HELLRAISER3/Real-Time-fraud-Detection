import requests
import pandas as pd

df = pd.read_csv("data/training_processed/training_dataset_final.csv")

sample = df.drop(columns=['label']).iloc[0].to_dict()

sample = {k: (None if pd.isna(v) else v) for k, v in sample.items()}

print(f"Testing API. Features len: {len(sample)}")

try:
    response = requests.post(
        "http://localhost:8000/" \
        "predict",
        json=sample
    )

    if response.status_code == 200:
        result = response.json()
        print("\n" + "="*60)
        print("PREDICTION:")
        print("="*60)
        print(f"Prediction: {result['prediction']} ({'FRAUD' if result['prediction'] == 1 else 'NO FRAUD'})")
        print(f"Probability: {result['probability']:.4f}")
        print(f"Risk Level: {result['risk_level'].upper()}")
        print(f"Timestamp: {result['timestamp']}")
        print("="*60)
    else:
        print(f"\nError: {response.status_code}")
        print(response.text)
except Exception as e:
    print(f"\nError connecting to API: {e}")
