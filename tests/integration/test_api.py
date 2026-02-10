import requests
import time
import json

BASE_URL = "http://localhost:8000" 

def test_health():
    print("\n--- Testing Health Endpoint ---")
    try:
        response = requests.get(f"{BASE_URL}/health")
        response.raise_for_status()
        data = response.json()
        
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(data, indent=2)}")
        
        if data["model_loaded"] and data["status"] == "healthy":
            print("System is Healthy")
        else:
            print("System Unhealthy")
            
    except requests.exceptions.ConnectionError:
        print("Could not connect. Is the server running?")

def test_predict_flow():
    print("\n--- Testing Prediction Endpoint ---")
    
    payload = {
        "SK_ID_CURR": 100002,
        "AMT_CREDIT": 150000.0,
        "AMT_GOODS_PRICE": 120000.0 
    }
    
    try:
        start_time = time.time()
        response = requests.post(f"{BASE_URL}/predict", json=payload)
        latency = (time.time() - start_time) * 1000
        
        print(f"Latency: {latency:.2f}ms")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Prediction Success!")
            print(f"Risk Level: {data['risk_level']}")
            print(f"Probability: {data['probability']}")
            
            used_features = data.get("used_features", {})
            if "age_years" in used_features:
                print(f"Feast Retrieval Verified: age_years = {used_features['age_years']}")
            else:
                print("Warning: 'age_years' missing. Feast lookup might have failed or ID not found.")
                
        else:
            print(f"Failed: {response.status_code}")
            print(f"Error: {response.text}")
            
    except Exception as e:
        print(f"Error during request: {e}")

if __name__ == "__main__":
    
    test_health()
    test_predict_flow()