import pandas as pd
import joblib
import time
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import psutil
from sklearn.metrics import accuracy_score, classification_report

# Load trained model
model_path = "sybil_gwo_rf_model.pkl"
model = joblib.load(model_path)

# Load test data
X_test = pd.read_csv("X_test_gwo.csv")
y_test = pd.read_csv("y_test_gwo.csv").squeeze()

# Start prediction timer
start_time = time.time()
y_pred = model.predict(X_test)
end_time = time.time()

# Metrics
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Model size
model_size = os.path.getsize(model_path)
model_size_kb = model_size / 1024
model_size_mb = model_size / (1024 * 1024)

# Time metrics
total_pred_time = end_time - start_time
avg_pred_time = total_pred_time / len(X_test)

# Memory usage
process = psutil.Process(os.getpid())
memory_usage_mb = process.memory_info().rss / (1024 * 1024)

# Feature importance
importances = model.feature_importances_
feature_names = X_test.columns
feature_importance = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# Extract speed importance
try:
    speed_importance = feature_importance[feature_importance["Feature"] == "Speed"]["Importance"].values[0]
except IndexError:
    speed_importance = 0

# ------------------------- COMPLEXITY SCORE -------------------------
alpha = 1       # model size (KB)
beta = 1000     # average prediction time
gamma = 100     # speed importance

complexity_score = (alpha * model_size_kb) + (beta * avg_pred_time) - (gamma * speed_importance)

# ------------------------- RESULTS -------------------------
print("\n📌 GWO + Random Forest Sybil Detection Results")
print("==============================================")
print(f"✅ Accuracy Score         : {accuracy:.4f}")
print(f"📦 Model Size             : {model_size_kb:.2f} KB")
print(f"🧠 Memory Usage           : {memory_usage_mb:.2f} MB")
print(f"🕒 Total Prediction Time  : {total_pred_time:.6f} seconds")
print(f"⚡ Avg Time per Sample    : {avg_pred_time:.8f} seconds")
print("\n📊 Classification Report:")
print(report)

# ------------------------- COMPLEXITY SUMMARY -------------------------
print("🧠 Complexity Score Summary")
print("---------------------------------------------------")
print(f"Model:              GWO + Random Forest")
print(f"Accuracy:           {accuracy:.4f}")
print(f"Model Size (KB):    {model_size_kb:.2f}")
print(f"Total Time (s):     {total_pred_time:.6f}")
print(f"Avg Time/Sample (s):{avg_pred_time:.8f}")
print(f"Speed Importance:   {speed_importance:.6f}")
print(f"📉 Complexity Score: {complexity_score:.4f}")
print("---------------------------------------------------")

# ------------------------- FEATURE IMPORTANCE -------------------------
plt.figure(figsize=(10, 6))
sorted_idx = np.argsort(importances)
sns.barplot(x=importances[sorted_idx], y=feature_names[sorted_idx])
plt.title("Feature Importance - GWO + RF")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()
