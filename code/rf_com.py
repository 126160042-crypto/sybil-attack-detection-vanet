import pandas as pd
import numpy as np
import joblib
import time
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

# 🔄 Load model
model_path = "sybil_rf_model.pkl"  # <--- Replace with respective model path
print(f"📥 Loading model from: {model_path}")
model = joblib.load(model_path)
print("✅ Model loaded successfully!\n")

# 📊 Load test data
X_test = pd.read_csv("X_test.csv")
y_test = pd.read_csv("y_test.csv").squeeze()

# 🕒 Predict and measure time
start = time.time()
y_pred = model.predict(X_test)
end = time.time()

total_time = end - start
avg_time = total_time / len(X_test)

# 🎯 Evaluation
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Accuracy: {accuracy:.4f}\n")
print("🔍 Classification Report:")
print(classification_report(y_test, y_pred))

# 📦 Model size
model_size_kb = os.path.getsize(model_path) / 1024
print(f"\n📦 Model Size: {model_size_kb:.2f} KB")
print(f"🕒 Total Prediction Time: {total_time:.6f} sec")
print(f"⏱️ Average Time per Sample: {avg_time:.8f} sec\n")

# 📊 Feature Importance
df = pd.read_excel("synthetic_75000_combined_traffic_data.xlsx", sheet_name="Sheet1")
df["Sybil"] = df["Sybil"].map({"Yes": 1, "No": 0})
df = df.dropna(subset=["Sybil"])
categorical_cols = ["Vehicle ID", "Vehicle Type", "Edge ID"]
for col in categorical_cols:
    if col in df.columns:
        df[col] = LabelEncoder().fit_transform(df[col])
df = df.drop(columns=["Timestamp"], errors="ignore")

feature_names = np.array(df.drop(columns=["Sybil"]).columns)
importances = model.feature_importances_
sorted_idx = np.argsort(importances)

plt.figure(figsize=(10, 6))
sns.barplot(x=importances[sorted_idx], y=feature_names[sorted_idx])
plt.title("Feature Importance (Random Forest)")

plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()
