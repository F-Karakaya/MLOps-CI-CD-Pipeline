# MLOps Overview

This project demonstrates a production-grade MLOps pipeline for an ML application.

## Pipeline Components

1.  **Data Ingestion & Validation**: Raw data is ingested and validated for schema consistency and missing values.
2.  **Training**: A Logistic Regression model is trained using Scikit-Learn. All training parameters and metrics are logged to MLflow.
3.  **Evaluation**: The trained model is evaluated on a hold-out test set. Metrics such as Accuracy, F1-Score, and ROC AUC are computed.
4.  **Registration**: The best performing model is registered in the MLflow Model Registry.
5.  **Deployment**: The model is containerized using Docker and served via a Flask API.
6.  **Monitoring**: The systems monitors for data drift and performance decay.

## Technologies Used

-   **MLflow**: Experiment tracking and model registry.
-   **Kubeflow Pipelines**: Pipeline orchestration.
-   **Docker**: Containerization.
-   **GitHub Actions**: CI/CD automation.
-   **Flask**: Model serving.
-   **Evidently/Custom**: Monitoring.
