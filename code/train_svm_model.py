import pandas as pd
import numpy as np
import joblib
import time
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load dataset
FILE_NAME = "synthetic_75000_combined_traffic_data.xlsx"
df = pd.read_excel(FILE_NAME, sheet_name="Sheet1")

# Convert "Sybil" to binary
if "Sybil" in df.columns:
    df["Sybil"] = df["Sybil"].map({"Yes": 1, "No": 0})

# Encode categorical features
categorical_cols = ["Vehicle ID", "Vehicle Type", "Edge ID"]
label_encoders = {}

for col in categorical_cols:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le  # Save encoders if needed

# Drop unused column
df = df.drop(columns=["Timestamp"], errors="ignore")

# 🔧 Drop rows with missing values
df = df.dropna()

# Define features and target
X = df.drop(columns=["Sybil"])
y = df["Sybil"]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)

# ⏱️ Measure training time
start_time = time.time()

# Train SVM model
svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', class_weight='balanced')
svm_model.fit(X_train, y_train)

end_time = time.time()
training_time = end_time - start_time

# Save model and scaler
joblib.dump(svm_model, "sybil_svm_model.pkl")
joblib.dump(scaler, "svm_scaler.pkl")
print("✅ SVM model saved as 'sybil_svm_model.pkl'!")
print("✅ Scaler saved as 'svm_scaler.pkl'!")

# Save test data
X_test_df = pd.DataFrame(X_test, columns=X.columns)
X_test_df.to_csv("svm_X_test.csv", index=False)
y_test.to_csv("svm_y_test.csv", index=False)
print("✅ Test data saved (svm_X_test.csv & svm_y_test.csv)!")

# Show training time
print(f"🕒 SVM Training Time: {training_time:.4f} seconds")
