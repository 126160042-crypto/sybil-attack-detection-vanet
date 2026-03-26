import pandas as pd
import numpy as np
import joblib
import time
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

# ----------------------------
# 📥 Load Trained Model
# ----------------------------
print("🔄 Loading WOA-tuned Random Forest model...")
model = joblib.load("sybil_woa_rf_model.pkl")
print("✅ Model loaded successfully!\n")

# ----------------------------
# 📁 Load Test Data
# ----------------------------
X_test = pd.read_csv("X_test_woa.csv")
y_test = pd.read_csv("y_test_woa.csv").squeeze()

# ----------------------------
# 🔮 Prediction
# ----------------------------
start = time.time()
y_pred = model.predict(X_test)
end = time.time()

total_time = end - start
avg_time = total_time / len(X_test)

# ----------------------------
# 📊 Evaluation
# ----------------------------
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Accuracy: {accuracy:.4f}")
print("📊 Classification Report:")
print(classification_report(y_test, y_pred))

# ----------------------------
# 💾 Model Size
# ----------------------------
model_size_kb = os.path.getsize("sybil_woa_rf_model.pkl") / 1024
print(f"🕒 Total Prediction Time: {total_time:.4f} seconds")
print(f"⏱ Average Prediction Time per Sample: {avg_time:.8f} seconds")
print(f"💾 Model Size: {model_size_kb:.4f} KB")

# ----------------------------
# 🔍 Feature Importance
# ----------------------------
df = pd.read_excel("synthetic_75000_combined_traffic_data.xlsx", sheet_name="Sheet1")
df["Sybil"] = df["Sybil"].map({"Yes": 1, "No": 0})
df = df.dropna(subset=["Sybil"])

categorical_cols = ["Vehicle ID", "Vehicle Type", "Edge ID"]
for col in categorical_cols:
    df[col] = LabelEncoder().fit_transform(df[col])
df = df.drop(columns=["Timestamp"], errors="ignore")

feature_names = np.array(df.drop(columns=["Sybil"]).columns)
importances = model.feature_importances_

# Get Speed feature importance (default = 0 if not found)
speed_importance = 0.0
if "Speed" in feature_names:
    speed_index = list(feature_names).index("Speed")
    speed_importance = importances[speed_index]

# ----------------------------
# 🧠 Complexity Score
# ----------------------------
alpha = 1.0          # Weight for model size
beta = 10000.0       # Weight for avg prediction time
gamma = 100.0        # Weight for speed importance

complexity_score = (alpha * model_size_kb) + (beta * avg_time) - (gamma * speed_importance)

print("\n🧮 Complexity Analysis:")
print(f"📦 Model Size (KB): {model_size_kb:.4f}")
print(f"⚡ Avg Prediction Time (s): {avg_time:.8f}")
print(f"🚗 Speed Feature Importance: {speed_importance:.4f}")
print(f"📉 Complexity Score: {complexity_score:.4f}")

# ----------------------------
# 📈 Feature Importance Plot
# ----------------------------
sorted_idx = np.argsort(importances)
plt.figure(figsize=(10, 6))
sns.barplot(x=importances[sorted_idx], y=feature_names[sorted_idx])
plt.title("Feature Importance (WOA + RF)")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.tight_layout()
plt.show()
