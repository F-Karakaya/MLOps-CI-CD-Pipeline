
import pandas as pd
import json
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve

def evaluate():
    model_path = "models/model.pkl"
    X_test_path = "data/processed/X_test.csv"
    y_test_path = "data/processed/y_test.csv"
    
    if not os.path.exists(model_path):
        print("Model not found. Run training first.")
        return

    # Load resources
    model = joblib.load(model_path)
    X_test = pd.read_csv(X_test_path)
    y_test = pd.read_csv(y_test_path).values.ravel() # ravel for 1d array

    # Predict
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Metrics
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba)
    }

    # Save metrics
    with open("evaluation/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
        
    print("Metrics calculated:", metrics)

    # Plot Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    os.makedirs("outputs/monitoring", exist_ok=True) # Saving to monitoring or just outputs? Prompt says outputs/
    plt.savefig("outputs/confusion_matrix.png")
    plt.close()

    # Plot ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {metrics['roc_auc']:.2f}")
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.savefig("outputs/roc_curve.png")
    plt.close()
    
    print("Evaluation plots saved to outputs/")

if __name__ == "__main__":
    evaluate()
