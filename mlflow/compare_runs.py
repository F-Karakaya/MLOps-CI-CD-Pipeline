
import mlflow
import pandas as pd
import logging
from mlflow.tracking import MlflowClient

logging.basicConfig(level=logging.INFO)

def compare_runs(experiment_name="mlops_cicd_experiment", top_n=5):
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    
    if not experiment:
        return

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.train_accuracy DESC"],
        max_results=top_n
    )
    
    data = []
    for run in runs:
        data.append({
            "run_id": run.info.run_id,
            "train_accuracy": run.data.metrics.get("train_accuracy"),
            "start_time": run.info.start_time
        })
        
    df = pd.DataFrame(data)
    print("Top Runs Config Comparison:")
    print(df)
    
    df.to_csv("outputs/run_comparison.csv", index=False)

if __name__ == "__main__":
    compare_runs()
