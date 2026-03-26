import pandas as pd
import numpy as np
import joblib  # For saving the model
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

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

# Define features and target
X = df.drop(columns=["Sybil"])
y = df["Sybil"]

# Split dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Train the Random Forest model
rf_model = RandomForestClassifier(
    n_estimators=100, max_depth=10, random_state=42, class_weight="balanced"
)
rf_model.fit(X_train, y_train)

# Save the trained model
joblib.dump(rf_model, "sybil_rf_model.pkl")
print("✅ Model saved as 'sybil_rf_model.pkl'!")

# Save test data for evaluation
X_test.to_csv("X_test.csv", index=False)
y_test.to_csv("y_test.csv", index=False)
print("✅ Test data saved for evaluation (X_test.csv & y_test.csv)!")
import pandas as pd
import numpy as np
import joblib
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load dataset
FILE_NAME = "synthetic_75000_combined_traffic_data.xlsx"
df = pd.read_excel(FILE_NAME, sheet_name="Sheet1")

# Convert "Sybil" to binary
if "Sybil" in df.columns:
    df["Sybil"] = df["Sybil"].map({"Yes": 1, "No": 0})

# Encode categorical features
categorical_cols = ["Vehicle ID", "Vehicle Type", "Edge ID"]
for col in categorical_cols:
    if col in df.columns:
        df[col] = LabelEncoder().fit_transform(df[col])

# Drop unused columns
df = df.drop(columns=["Timestamp"], errors="ignore")

# Define X and y
X = df.drop(columns=["Sybil"])
y = df["Sybil"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ⏱️ Measure training time
start_time = time.time()

# Train model
rf_model = RandomForestClassifier(
    n_estimators=100, max_depth=10, random_state=42, class_weight="balanced"
)
rf_model.fit(X_train, y_train)

end_time = time.time()
training_time = end_time - start_time

# Save model
joblib.dump(rf_model, "sybil_rf_model.pkl")
print("✅ Model saved as 'sybil_rf_model.pkl'!")

# Save test data
X_test.to_csv("X_test.csv", index=False)
y_test.to_csv("y_test.csv", index=False)
print("✅ Test data saved for evaluation (X_test.csv & y_test.csv)!")

# ✅ Print training time
print(f"🕒 Training Time: {training_time:.4f} seconds")
