
import os
import pytest
import pandas as pd
from training.model import ModelWrapper

def test_data_existence():
    # Ensure raw data exists or generation works
    if not os.path.exists("data/raw/dataset.csv"):
        pytest.skip("Data not found, skipping validation test.")
    df = pd.read_csv("data/raw/dataset.csv")
    assert not df.empty
    assert "target" in df.columns

def test_model_training():
    # Test model wrapper locally
    model = ModelWrapper({"C": 0.1})
    assert model.params["C"] == 0.1
    # Simple mock train
    X = [[0,0], [1,1]]
    y = [0, 1]
    model.fit(X, y)
    assert model.predict([[0,0]])[0] == 0

def test_imports():
    from training import train
    from evaluation import evaluate
    assert True
