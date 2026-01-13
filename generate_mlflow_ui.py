
import matplotlib.pyplot as plt
import pandas as pd

def create_mlflow_screenshot():
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('tight')
    ax.axis('off')
    
    columns = ["Run ID", "Start Time", "Source", "Metrics.Accuracy", "Metrics.F1", "Params.C"]
    data = [
        ["a1b2c3d4", "2026-01-13 14:00", "train.py", "0.92", "0.91", "1.0"],
        ["e5f6g7h8", "2026-01-13 13:45", "train.py", "0.88", "0.87", "0.1"],
        ["i9j0k1l2", "2026-01-13 13:30", "train.py", "0.85", "0.84", "0.01"],
    ]
    
    table = ax.table(cellText=data, colLabels=columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)
    
    plt.title("MLflow Experiments UI (Simulated)", fontsize=14)
    output_path = "outputs/mlflow_runs/run_screenshot.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved {output_path}")

if __name__ == "__main__":
    create_mlflow_screenshot()
