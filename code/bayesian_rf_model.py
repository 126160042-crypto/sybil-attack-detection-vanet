import pandas as pd
import time
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from skopt import BayesSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import warnings

# Ignore unnecessary warnings
warnings.filterwarnings("ignore")

print("📥 Loading dataset...")
# Load your dataset
data = pd.read_excel("synthetic_75000_combined_traffic_data.xlsx")

print("🧹 Preprocessing...")
# Encode categorical columns using LabelEncoder
label_encoders = {}
for col in ['Vehicle Type', 'Edge ID']:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le  # Save encoder for future use

# Updated ACO-selected features with Position X added
features = ['Vehicle Type', 'Speed (m/s)', 'Position X', 'Position Y', 'Edge ID']
X = data[features]
y = data['Sybil']

# Split data into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

# Save test data to CSV
X_test.to_csv("X_test_bayes_rf.csv", index=False)
y_test.to_csv("y_test_bayes_rf.csv", index=False)
print("💾 Test data saved as 'X_test_bayes_rf.csv' and 'y_test_bayes_rf.csv'")

# Define Bayesian Search space
param_space = {
    'n_estimators': (50, 300),
    'max_depth': (4, 32),
    'min_samples_split': (2, 10),
    'min_samples_leaf': (1, 10),
    'bootstrap': [True, False],
}

# Initialize Bayesian Optimizer
opt = BayesSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    search_spaces=param_space,
    n_iter=25,
    cv=3,
    n_jobs=-1,
    random_state=42,
    verbose=0
)

print("🚀 Starting Bayesian optimization and training...")
# Train and time the model
start_time = time.time()
opt.fit(X_train, y_train)
training_time = time.time() - start_time

# Get the best model
model = opt.best_estimator_

# Evaluate on test set
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\n✅ Training complete!")
print(f"🏆 Best Parameters: {opt.best_params_}")
print(f"🎯 Accuracy on Test Set: {accuracy:.4f}")
print("\n📊 Classification Report:\n")
print(classification_report(y_test, y_pred))

# Save the trained model
model_filename = "sybil_bayes_rf_model.pkl"
joblib.dump(model, model_filename)
print(f"\n💾 Model saved as '{model_filename}'")
print(f"⏱️ Training Time: {training_time:.4f} seconds")

# Optional: Save label encoders
joblib.dump(label_encoders, "label_encoders.pkl")
print("💾 Label encoders saved as 'label_encoders.pkl'")
