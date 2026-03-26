import pandas as pd
import numpy as np
import joblib
import time
import os  # <-- Add this import
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.impute import SimpleImputer

# Load dataset
FILE_NAME = "synthetic_75000_combined_traffic_data.xlsx"
df = pd.read_excel(FILE_NAME, sheet_name="Sheet1")

# Convert 'Sybil' column to numeric: Yes → 1, No → 0
if "Sybil" in df.columns:
    df["Sybil"] = df["Sybil"].map({"Yes": 1, "No": 0})

# Encode categorical columns
categorical_cols = ["Vehicle ID", "Vehicle Type", "Edge ID"]
label_encoders = {}

for col in categorical_cols:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le  # Save encoders if you want to decode later

# Drop unnecessary column
df = df.drop(columns=["Timestamp"], errors="ignore")

# Handle missing values (Option 1: Remove rows with NaNs)
df = df.dropna(subset=["Sybil"] + df.drop(columns=["Sybil"]).columns.tolist())

# Alternatively, Option 2: Impute missing values (uncomment to use)
# imputer = SimpleImputer(strategy="mean")
# df[df.columns] = imputer.fit_transform(df[df.columns])

# Define features and target
X = df.drop(columns=["Sybil"])
y = df["Sybil"]

# Train-test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Feature Scaling (Standardization)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ⏱️ Measure training time
start = time.time()

# Train KNN model
knn_model = KNeighborsClassifier(n_neighbors=5)  # You can adjust the number of neighbors (n_neighbors) as needed
knn_model.fit(X_train_scaled, y_train)

end = time.time()
training_time = end - start

# Save the trained model and scaler
joblib.dump(knn_model, "sybil_knn_model.pkl")
joblib.dump(scaler, "knn_scaler.pkl")
print("✅ KNN model and scaler saved successfully!\n")

# Evaluate the model on the test set
y_pred = knn_model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model Accuracy: {accuracy:.4f}")

# Classification report
print("🔹 Classification Report:")
print(classification_report(y_test, y_pred))

# Model size
model_size_kb = os.path.getsize("sybil_knn_model.pkl") / 1024  # This line works now that os is imported
print(f"\n📦 Model Size: {model_size_kb:.2f} KB")
print(f"🕒 Training Time: {training_time:.6f} seconds")

# Save test data for later use in the testing phase
X_test.to_csv("knn_X_test.csv", index=False)
y_test.to_csv("knn_y_test.csv", index=False)
print("✅ Test data saved for evaluation (knn_X_test.csv & knn_y_test.csv)!\n")
