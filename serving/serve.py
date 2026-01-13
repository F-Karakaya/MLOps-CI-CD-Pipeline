
from flask import Flask, request, jsonify
import joblib
import pandas as pd
import logging

app = Flask(__name__)
model = None

def load_model():
    global model
    # For demo purposes, we load the local artifact. 
    # In production, this would fetch from MLflow Model Registry using mlflow.pyfunc.load_model(model_uri)
    try:
        model = joblib.load("models/model.pkl")
        logging.info("Model loaded successfully.")
    except Exception as e:
        logging.error(f"Error loading model: {e}")

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({"error": "Model not loaded"}), 500
    
    try:
        data = request.get_json()
        input_df = pd.DataFrame(data['instances'])
        predictions = model.predict(input_df)
        return jsonify({"predictions": predictions.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    load_model()
    app.run(host='0.0.0.0', port=5000)
