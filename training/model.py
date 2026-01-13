
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator

class ModelWrapper:
    """A wrapper for the model to standardize interface if needed."""
    def __init__(self, params=None):
        self.params = params or {}
        self.model = LogisticRegression(**self.params)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)
    
    def get_params(self):
        return self.model.get_params()
