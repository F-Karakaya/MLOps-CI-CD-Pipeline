
import requests
import json
import numpy as np

# Create dummy data
data = {
    "instances": np.random.rand(5, 10).tolist() # 5 samples, 10 features
}

url = "http://localhost:5000/predict"

try:
    response = requests.post(url, json=data)
    print("Status Code:", response.status_code)
    print("Response:", response.json())
except Exception as e:
    print("Could not connect to server (expected if server not running).")
    print("Payload would be:", json.dumps(data))
