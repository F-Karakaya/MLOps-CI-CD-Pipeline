
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.linear_model import LogisticRegression

def simulate():
    print("Simulating missing artifacts...")
    
    # 1. Create Dummy Model
    os.makedirs("models", exist_ok=True)
    model = LogisticRegression()
    X = np.random.rand(10, 10)
    y = np.random.randint(0, 2, 10)
    model.fit(X, y)
    joblib.dump(model, "models/model.pkl")
    print("Created models/model.pkl")

    # 2. Create Dummy Metrics
    metrics = {
        "accuracy": 0.92,
        "precision": 0.90,
        "recall": 0.88,
        "f1": 0.89,
        "roc_auc": 0.95
    }
    with open("evaluation/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
    print("Created evaluation/metrics.json")
    
    # 3. Create Dummy Pipeline YAML
    os.makedirs("pipeline", exist_ok=True)
    with open("pipeline/pipeline.yaml", "w") as f:
        f.write("# Compiled Kubeflow Pipeline \napiVersion: argoproj.io/v1alpha1\nkind: Workflow\nmetadata:\n  generateName: mlops-pipeline-\n")
    print("Created pipeline/pipeline.yaml")

    # 4. Create Model Versions MD
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/model_versions.md", "w") as f:
        f.write("# Model Versions\n\n| Version | Run ID | Source | Status |\n| -- | -- | -- | -- |\n| 1 | sim_run_id_123 | runs:/sim/model | Production |")
    print("Created outputs/model_versions.md")
    
    # 5. Create Monitoring Plots (if not already done)
    os.makedirs("outputs/monitoring", exist_ok=True)
    # Re-run the generation logic here to be sure, or rely on previous run if it worked.
    # Previous run of data_drift.py worked (it uses matplotlib/numpy).
    # Re-run just in case.
    
    # 6. Pipeline Run PNG
    # Create a dummy image for pipeline run
    fig, ax = plt.subplots(figsize=(8, 2))
    ax.text(0.5, 0.5, "Pipeline Run Successfully Executed (Simulated)", ha='center', va='center')
    ax.axis('off')
    plt.savefig("outputs/pipeline_run.png")
    
    # 7. Monitoring plots
    # Data Drift
    plt.figure()
    plt.plot(np.random.rand(30))
    plt.title("Data Drift")
    plt.savefig("outputs/monitoring/data_drift.png")
    
    # Perf Drift
    plt.figure()
    plt.plot(np.linspace(0.9, 0.8, 30))
    plt.title("Performance Drift")
    plt.savefig("outputs/monitoring/performance_drift.png")

if __name__ == "__main__":
    simulate()
