
import pandas as pd
from sklearn.model_selection import train_test_split
import yaml
import logging

def load_data(path):
    """Loads CSV data."""
    logging.info(f"Loading data from {path}")
    return pd.read_csv(path)

def preprocess_data(df, target_col="target", test_size=0.2, random_state=42):
    """Splits data into features and target, then train/test split."""
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test

def load_params(param_file="hyperparams.yaml"):
    """Loads hyperparameters from YAML."""
    with open(param_file, "r") as f:
        params = yaml.safe_load(f)
    return params
