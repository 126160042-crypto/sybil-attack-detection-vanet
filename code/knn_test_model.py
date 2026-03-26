import pandas as pd
import numpy as np
import joblib
import time
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load model
print("🔄 Loading model...")
knn_model = joblib.load("sybil_knn_model.pkl")
scaler = joblib.load("knn_scaler.pkl")
print("✅ Model and scaler loaded successfully!\n")

# Load test data
X_test = pd.read_csv("knn_X_test.csv")
y_test = pd.read_csv("knn_y_test.csv").squeeze()

# Apply scaling on X_test using the scaler
X_test_scaled = scaler.transform(X_test)

# Predict and measure time
start = time.time()
y_pred = knn_model.predict(X_test_scaled)
end = time.time()

total_time = end - start
avg_time = total_time / len(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model Accuracy: {accuracy:.4f}\n")

print("🔹 Classification Report:")
print(classification_report(y_test, y_pred))

# Model size
model_size_kb = os.path.getsize("sybil_knn_model.pkl") / 1024
print(f"\n📦 Model Size: {model_size_kb:.2f} KB")
print(f"🕒 Total Prediction Time: {total_time:.6f} seconds")
print(f"📈 Average Prediction Time per Sample: {avg_time:.8f} seconds\n")

# Load full dataset for feature importance (KNN does not provide direct feature importance)
df = pd.read_excel("synthetic_75000_combined_traffic_data.xlsx", sheet_name="Sheet1")

# Clean and encode again for consistency
print("🔍 Unique values in 'Sybil' before processing:", df["Sybil"].unique())
df["Sybil"] = df["Sybil"].map({"Yes": 1, "No": 0})
df = df.dropna(subset=["Sybil"])
print(f"✅ Remaining samples after cleaning: {len(df)}")

# Label encode categorical features
categorical_cols = ["Vehicle ID", "Vehicle Type", "Edge ID"]
for col in categorical_cols:
    if col in df.columns:
        df[col] = LabelEncoder().fit_transform(df[col])
df = df.drop(columns=["Timestamp"], errors="ignore")

# Feature importance visualization for KNN (approximation using correlation matrix)
# Since KNN does not provide feature importance, we can show a correlation matrix
corr_matrix = df.corr()
plt.figure(figsize=(10, 6))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Feature Correlation Matrix (KNN)")
plt.tight_layout()
plt.show()
