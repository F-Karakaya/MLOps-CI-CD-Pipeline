
import argparse
import mlflow
import mlflow.sklearn
import pandas as pd
import os
import sys
import logging
import joblib

# Add project root to path
sys.path.append(os.getcwd())

from training.utils import load_data, preprocess_data, load_params
from training.model import ModelWrapper

logging.basicConfig(level=logging.INFO)

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="data/raw/dataset.csv", help="Path to raw data")
    parser.add_argument("--params_file", default="hyperparams.yaml", help="Path to params file")
    args = parser.parse_args()

    # Load Params
    all_params = load_params(args.params_file)
    model_params = all_params["model"]["params"]
    data_params = all_params["data"]
    
    # Load and Preprocess Data
    df = load_data(args.data_path)
    X_train, X_test, y_train, y_test = preprocess_data(df, test_size=data_params["test_size"], random_state=data_params["random_state"])

    # MLflow Tracking
    mlflow.set_experiment(all_params["training"]["experiment_name"])
    
    with mlflow.start_run(run_name=all_params["training"]["run_name"]) as run:
        logging.info(f"Starting run ID: {run.info.run_id}")
        
        # Log params
        mlflow.log_params(model_params)
        mlflow.log_params(data_params)

        # Train Model
        logging.info("Training model...")
        wrapper = ModelWrapper(model_params)
        wrapper.fit(X_train, y_train)
        
        # Log simplified metrics (accuracy on train)
        train_score = wrapper.model.score(X_train, y_train)
        mlflow.log_metric("train_accuracy", train_score)
        
        # Save Model artifact
        mlflow.sklearn.log_model(wrapper.model, "model")
        
        # Save locally for eval step
        os.makedirs("models", exist_ok=True)
        joblib.dump(wrapper.model, "models/model.pkl")
        
        # Save X_test and y_test for evaluation step
        os.makedirs("data/processed", exist_ok=True)
        X_test.to_csv("data/processed/X_test.csv", index=False)
        y_test.to_csv("data/processed/y_test.csv", index=False)
        
        logging.info("Training complete. Model saved.")

if __name__ == "__main__":
    train()
