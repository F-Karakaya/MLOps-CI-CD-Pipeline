
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Simulate data drift
np.random.seed(42)
days = 30
baseline_mean = 0.5
drift_magnitude = 0.05

means = [baseline_mean + (i * drift_magnitude * 0.1) + np.random.normal(0, 0.01) for i in range(days)]

plt.figure(figsize=(10, 6))
plt.plot(range(days), means, marker='o', linestyle='-', color='b')
plt.title("Data Drift Simulation: Feature Mean Over Time")
plt.xlabel("Day")
plt.ylabel("Mean Value of Feature X")
plt.axhline(baseline_mean, color='r', linestyle='--', label='Baseline')
plt.legend()
plt.grid(True)

os.makedirs("outputs/monitoring", exist_ok=True)
plt.savefig("outputs/monitoring/data_drift.png")
print("Data drift plot saved to outputs/monitoring/data_drift.png")
