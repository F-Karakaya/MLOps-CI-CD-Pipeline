# Monitoring Strategy

Monitoring is crucial for maintaining model performance in production.

## Data Drift
We monitor the statistical properties of the input features. If the distribution of the incoming data deviates significantly from the training data (baseline), a drift alert is triggered.
-   **Method**: Statistical tests (e.g., KS-test) or tracking mean/std deviations.
-   **Visualization**: Plots showing feature means over time (see `outputs/monitoring/data_drift.png`).

## Performance Drift
We track the model's predictive performance (e.g., accuracy) over time.
-   **Method**: Compare ground truth (when available) with predictions.
-   **Visualization**: Plots showing accuracy trends over time (see `outputs/monitoring/performance_drift.png`).

## Feedback Loop
When drift is detected:
1.  Trigger retrianing with new data.
2.  Update the model in the registry.
3.  Redeploy the service.
