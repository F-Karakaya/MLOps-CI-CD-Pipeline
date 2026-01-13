
import mlflow
import argparse
import sys
import logging
from mlflow.tracking import MlflowClient

logging.basicConfig(level=logging.INFO)

def register_best_model(experiment_name="mlops_cicd_experiment", model_name="mlops_model"):
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    
    if not experiment:
        logging.error(f"Experiment {experiment_name} not found.")
        return

    # Get all runs
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.train_accuracy DESC"]
    )
    
    if not runs:
        logging.error("No runs found.")
        return

    best_run = runs[0]
    logging.info(f"Best run ID: {best_run.info.run_id} with Accuracy: {best_run.data.metrics.get('train_accuracy')}")
    
    # Register model
    run_id = best_run.info.run_id
    model_uri = f"runs:/{run_id}/model"
    
    logging.info(f"Registering model from URI: {model_uri}")
    result = mlflow.register_model(model_uri, model_name)
    
    # Create model_versions.md
    version_info = f"""
# Model Versions
| Version | Run ID | Source | Status |
| -- | -- | -- | -- |
| {result.version} | {run_id} | {model_uri} | {result.current_stage} |
"""
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/model_versions.md", "w") as f:
        f.write(version_info)
    
    logging.info(f"Model registered as version {result.version}")

if __name__ == "__main__":
    register_best_model()

import os 
