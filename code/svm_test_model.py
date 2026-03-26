import pandas as pd
import numpy as np
import joblib
import time
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# ------------------------- LOAD MODEL -------------------------
print("🔄 Loading model...")
svm_model = joblib.load("sybil_svm_model.pkl")
scaler = joblib.load("svm_scaler.pkl")
print("✅ Model and scaler loaded successfully!\n")

# ------------------------- LOAD TEST DATA -------------------------
X_test = pd.read_csv("svm_X_test.csv")
y_test = pd.read_csv("svm_y_test.csv").squeeze()
X_test_scaled = scaler.transform(X_test)

# ------------------------- PREDICT -------------------------
start = time.time()
y_pred = svm_model.predict(X_test_scaled)
end = time.time()

# ------------------------- METRICS -------------------------
accuracy = accuracy_score(y_test, y_pred)
total_time = end - start
avg_time = total_time / len(X_test)

print(f"✅ Model Accuracy: {accuracy:.4f}\n")
print("🔹 Classification Report:")
print(classification_report(y_test, y_pred))

# ------------------------- COMPLEXITY METRICS -------------------------
model_size_kb = os.path.getsize("sybil_svm_model.pkl") / 1024
print(f"\n📦 Model Size: {model_size_kb:.2f} KB")
print(f"🕒 Total Prediction Time: {total_time:.6f} seconds")
print(f"📈 Average Prediction Time per Sample: {avg_time:.8f} seconds")

# ------------------------- FEATURE IMPORTANCE (for linear kernel only) -------------------------
df = pd.read_excel("synthetic_75000_combined_traffic_data.xlsx", sheet_name="Sheet1")

print("\n🔍 Unique values in 'Sybil' before processing:", df["Sybil"].unique())
df["Sybil"] = df["Sybil"].map({"Yes": 1, "No": 0})
df = df.dropna(subset=["Sybil"])
print(f"✅ Remaining samples after cleaning: {len(df)}")

categorical_cols = ["Vehicle ID", "Vehicle Type", "Edge ID"]
for col in categorical_cols:
    if col in df.columns:
        df[col] = LabelEncoder().fit_transform(df[col])
df = df.drop(columns=["Timestamp"], errors="ignore")

# ------------------------- PLOT: CONFUSION MATRIX -------------------------
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Normal", "Sybil"], yticklabels=["Normal", "Sybil"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

# ------------------------- PLOT: FEATURE IMPORTANCE (if linear) -------------------------
if hasattr(svm_model, 'kernel') and svm_model.kernel == 'linear':
    feature_names = np.array(df.drop(columns=["Sybil"]).columns)
    coefficients = svm_model.coef_.flatten()
    sorted_idx = np.argsort(coefficients)

    plt.figure(figsize=(10, 6))
    sns.barplot(
        x=coefficients[sorted_idx],
        y=feature_names[sorted_idx],
    )
    plt.xlabel("Coefficient Value")
    plt.ylabel("Features")
    plt.title("Feature Importance (SVM - Linear Kernel)")
    plt.tight_layout()
    plt.show()
else:
    print("⚠️ SVM model does not provide feature importance for non-linear kernels.")
