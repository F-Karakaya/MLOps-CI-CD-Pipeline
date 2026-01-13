
import os
import pandas as pd
import numpy as np
import logging
from sklearn.datasets import make_classification

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def generate_synthetic_data(output_path, n_samples=1000, n_features=10):
    """Generates a synthetic binary classification dataset."""
    logging.info(f"Generating synthetic data: {n_samples} samples, {n_features} features")
    X, y = make_classification(n_samples=n_samples, n_features=n_features, random_state=42)
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(n_features)])
    df["target"] = y
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    logging.info(f"Data saved to {output_path}")
    return df

def validate_data(file_path):
    """Validates the schema and quality of the dataset."""
    logging.info(f"Validating data at {file_path}")
    if not os.path.exists(file_path):
        logging.error(f"File not found: {file_path}")
        return False
    
    df = pd.read_csv(file_path)
    
    # Check for missing values
    if df.isnull().values.any():
        logging.warning("Data contains missing values!")
        # In a real scenario, we might fail or impute here.
    else:
        logging.info("No missing values found.")
        
    # Check target column exists
    if "target" not in df.columns:
        logging.error("Target column missing!")
        return False
    
    logging.info(f"Data validation passed. Shape: {df.shape}")
    return True

if __name__ == "__main__":
    # Default behavior: generate if not exists, then validate
    data_path = "data/raw/dataset.csv"
    if not os.path.exists(data_path):
        generate_synthetic_data(data_path)
    
    validate_data(data_path)
