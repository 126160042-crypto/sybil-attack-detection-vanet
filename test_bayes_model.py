import pandas as pd
import time
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import psutil
import os
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

# Ignore unnecessary warnings
warnings.filterwarnings("ignore")

# Load the model and test data
model = joblib.load("sybil_bayes_rf_model.pkl")
X_test = pd.read_csv("X_test_bayes_rf.csv")
y_test = pd.read_csv("y_test_bayes_rf.csv")

# Measure memory usage
process = psutil.Process(os.getpid())
memory_usage_mb = process.memory_info().rss / (1024 * 1024)

# Measure prediction time
start_pred = time.time()
y_pred = model.predict(X_test)
end_pred = time.time()

total_prediction_time = end_pred - start_pred
avg_prediction_time = total_prediction_time / len(X_test)

# Recalculate accuracy and classification report
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Estimate model size
model_file = "sybil_bayes_rf_model.pkl"
model_size_bytes = os.path.getsize(model_file)
model_size_mb = model_size_bytes / (1024 * 1024)
model_size_kb = model_size_bytes / 1024

# Feature importance
importance_dict = dict(zip(X_test.columns, model.feature_importances_))
speed_importance = importance_dict.get("Speed (m/s)", 0.0)

# Calculate a simple complexity score (sum of size and timing)
complexity_score = model_size_kb + avg_prediction_time * 1e6

# Pretty print output
print(f"\n✅ Accuracy: {accuracy:.4f}")
print("📊 Classification Report:")
print(report)

print(f"⏱️ Total Prediction Time: {total_prediction_time:.4f} seconds")
print(f"⚡ Average Prediction Time per Sample: {avg_prediction_time:.8f} seconds")
print(f"💾 Model Size: {model_size_mb:.4f} MB")
print(f"🧠 Memory Usage: {memory_usage_mb:.4f} MB")

print("\n📚 Complexity Analysis:")
print(f"📦 Model Size (KB): {model_size_kb:.2f}")
print(f"⚡ Avg Prediction Time (s): {avg_prediction_time:.8f}")
print(f"🚀 Speed Feature Importance: {speed_importance:.4f}")
print(f"🧮 Complexity Score: {complexity_score:.4f}")

# 🔍 Feature Importance Plot
importances = pd.Series(model.feature_importances_, index=X_test.columns)
importances_sorted = importances.sort_values(ascending=True)

plt.figure(figsize=(10, 6))
sns.barplot(x=importances_sorted, y=importances_sorted.index, palette="coolwarm")
plt.title("🔍 Feature Importance (Bayesian RF Model)")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.show()
