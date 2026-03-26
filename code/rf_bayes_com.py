import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import time
import os
from skopt import BayesSearchCV

# -------------------- Training Script (with Bayesian Optimization) --------------------

# Load dataset
data = pd.read_excel("synthetic_75000_combined_traffic_data.xlsx")

# Encode categorical features consistently using the same LabelEncoder
label_encoder_vehicle_type = LabelEncoder()
label_encoder_edge_id = LabelEncoder()

data['Vehicle Type'] = label_encoder_vehicle_type.fit_transform(data['Vehicle Type'])
data['Edge ID'] = label_encoder_edge_id.fit_transform(data['Edge ID'])

# Selected ACO best features
features = ['Vehicle Type', 'Speed (m/s)', 'Position Y', 'Edge ID']
X = data[features]
y = data['Sybil']

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Bayesian Optimization search space
param_space = {
    'n_estimators': (50, 300),
    'max_depth': (4, 32),
    'min_samples_split': (2, 10),
    'min_samples_leaf': (1, 10),
    'bootstrap': [True, False],
}

# Bayesian Optimizer
opt = BayesSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    search_spaces=param_space,
    n_iter=25,
    cv=3,
    n_jobs=-1,
    random_state=42
)

# Train and time the model
start_time = time.time()
opt.fit(X_train, y_train)
training_time = time.time() - start_time

# Best model after optimization
model = opt.best_estimator_

# Save the model to disk
joblib.dump(model, "sybil_bayes_rf_model.pkl")

# Output best parameters and training time
print(f"✅ Bayesian RF Model saved as 'sybil_bayes_rf_model.pkl'")
print(f"🏆 Best Parameters: {opt.best_params_}")
print(f"⏱️ Training Time: {training_time:.4f} seconds")

# -------------------- Testing Script --------------------

# Load the model for prediction
print("📥 Loading Bayesian-tuned Random Forest model...")
model = joblib.load("sybil_bayes_rf_model.pkl")
print("✅ Model loaded successfully!\n")

# Load test data (ensure test data is encoded in the same way)
X_test = pd.read_csv("X_test.csv")
y_test = pd.read_csv("y_test.csv").squeeze()

# Encode categorical features in the test data using the same encoder
# Here, we will handle unseen labels by transforming only known categories.
X_test['Vehicle Type'] = X_test['Vehicle Type'].map(lambda x: label_encoder_vehicle_type.transform([x])[0] if x in label_encoder_vehicle_type.classes_ else -1)
X_test['Edge ID'] = X_test['Edge ID'].map(lambda x: label_encoder_edge_id.transform([x])[0] if x in label_encoder_edge_id.classes_ else -1)

# Remove any rows where features have -1 values (indicating unseen labels)
X_test = X_test[X_test['Vehicle Type'] != -1]
X_test = X_test[X_test['Edge ID'] != -1]

# Ensure X_test contains only the same columns as X_train (the selected features)
X_test = X_test[features]

# Prediction and timing
start = time.time()
y_pred = model.predict(X_test)
end = time.time()

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Get model file size
model_size_kb = round(os.path.getsize("sybil_bayes_rf_model.pkl") / 1024, 2)

# Output results
print(f"✅ Model Accuracy: {accuracy:.4f}\n")
print(f"📊 Classification Report:\n{report}")
print(f"📦 Model Size: {model_size_kb} KB")
print(f"⏱️ Total Prediction Time: {end - start:.6f} seconds")
print(f"⚡ Average Prediction Time per Sample: {(end - start) / len(X_test):.9f} seconds")
