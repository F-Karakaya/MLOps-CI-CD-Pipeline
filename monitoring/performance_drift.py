
import numpy as np
import matplotlib.pyplot as plt
import os

# Simulate performance decay
days = 30
initial_accuracy = 0.95
decay_rate = 0.005

accuracies = [initial_accuracy - (i * decay_rate) + np.random.normal(0, 0.005) for i in range(days)]

plt.figure(figsize=(10, 6))
plt.plot(range(days), accuracies, marker='s', linestyle='-', color='g')
plt.title("Model Performance Drift: Accuracy Over Time")
plt.xlabel("Day")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.axhline(0.85, color='r', linestyle='--', label='Threshold')
plt.legend()
plt.grid(True)

os.makedirs("outputs/monitoring", exist_ok=True)
plt.savefig("outputs/monitoring/performance_drift.png")
print("Performance drift plot saved to outputs/monitoring/performance_drift.png")
