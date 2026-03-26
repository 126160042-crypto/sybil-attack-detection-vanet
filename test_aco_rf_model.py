import pandas as pd
import joblib
import time
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import MinMaxScaler

# Load model
print("🔄 Loading ACO-tuned Random Forest model...")
model_path = "sybil_aco_rf_model.pkl"
model = joblib.load(model_path)
print("✅ Model loaded successfully!")

# Load test data
X_test = pd.read_csv("X_test_aco.csv")
y_test = pd.read_csv("y_test_aco.csv").squeeze()

# Predict
start = time.time()
y_pred = model.predict(X_test)
end = time.time()

# Metrics
total_time = end - start
avg_time = total_time / len(X_test)
accuracy = accuracy_score(y_test, y_pred)
model_size_kb = os.path.getsize(model_path) / 1024

# Display core metrics
print(f"\n✅ Accuracy: {accuracy:.4f}")
print("📊 Classification Report:")
print(classification_report(y_test, y_pred))

print(f"🕒 Total Prediction Time: {total_time:.4f} seconds")
print(f"⏱ Average Prediction Time per Sample: {avg_time:.8f} seconds")
print(f"💾 Model Size: {model_size_kb:.4f} KB")

# Feature importance
importances = model.feature_importances_
features = X_test.columns

# Get 'Speed' feature importance if it exists
speed_importance = 0.0
for feat, imp in zip(features, importances):
    if "speed" in feat.lower():
        speed_importance = imp
        break

# Complexity score calculation
scaler_size = MinMaxScaler().fit([[50], [400]])
scaler_avg = MinMaxScaler().fit([[0.00001], [0.01]])

norm_model_size = 1 - scaler_size.transform([[model_size_kb]])[0][0]
norm_avg_time = 1 - scaler_avg.transform([[avg_time]])[0][0]

# Final complexity score (you can tweak this formula if needed)
complexity_score = norm_model_size * 50 + norm_avg_time * 50  # out of 100
complexity_score = round(complexity_score, 4)

# Display complexity section
print("\n🧮 Complexity Analysis:")
print(f"📦 Model Size (KB): {model_size_kb:.4f}")
print(f"⚡ Avg Prediction Time (s): {avg_time:.8f}")
print(f"🚗 Speed Feature Importance: {speed_importance:.4f}")
print(f"📉 Complexity Score: {complexity_score}\n")

# Optional: plot feature importances
sorted_idx = np.argsort(importances)
plt.figure(figsize=(10, 6))
sns.barplot(x=importances[sorted_idx], y=features[sorted_idx])
plt.title("Feature Importance (ACO + RF)")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.tight_layout()
plt.show()
