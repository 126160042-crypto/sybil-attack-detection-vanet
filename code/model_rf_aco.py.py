import numpy as np
import pandas as pd
import joblib
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# ------------------------ Load Dataset ------------------------
df = pd.read_excel("synthetic_75000_combined_traffic_data.xlsx", sheet_name="Sheet1")
df["Sybil"] = df["Sybil"].map({"Yes": 1, "No": 0})
df = df.dropna(subset=["Sybil"])

print("🔍 Class Balance:\n", df["Sybil"].value_counts(normalize=True))

categorical_cols = ["Vehicle ID", "Vehicle Type", "Edge ID"]
for col in categorical_cols:
    df[col] = LabelEncoder().fit_transform(df[col])
df = df.drop(columns=["Timestamp"], errors="ignore")

# Correlation Check
corrs = df.corr(numeric_only=True)["Sybil"].abs().sort_values(ascending=False)
print("\n📈 Feature Correlation with 'Sybil':\n", corrs)

# Drop features highly correlated (for testing realism)
df = df.drop(columns=[col for col in corrs.index if col != "Sybil" and corrs[col] > 0.97])

X = df.drop(columns=["Sybil"])
y = df["Sybil"]
feature_names = list(X.columns)

# ------------------------ ACO Parameters ------------------------
num_ants = 10
num_features = X.shape[1]
num_iterations = 20
evaporation_rate = 0.2
pheromone = np.ones(num_features) * 0.5
alpha, beta = 1, 2

# ------------------------ ACO Fitness ------------------------
def evaluate_solution(selected_features):
    if np.sum(selected_features) == 0:
        return 0
    selected_cols = [f for f, s in zip(feature_names, selected_features) if s == 1]
    X_sel = X[selected_cols]
    X_train, X_test, y_train, y_test = train_test_split(X_sel, y, test_size=0.2, stratify=y, random_state=42)
    
    # Add heavier noise
    noise_factor = 0.01
    X_train = X_train + np.random.normal(0, noise_factor, X_train.shape)
    X_test = X_test + np.random.normal(0, noise_factor, X_test.shape)

    clf = RandomForestClassifier(n_estimators=50, max_depth=8, class_weight='balanced', random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return accuracy_score(y_test, y_pred)

# ------------------------ ACO Loop ------------------------
best_solution = None
best_score = -1
start_train = time.time()

for iteration in range(num_iterations):
    all_solutions, scores = [], []
    for ant in range(num_ants):
        prob = (pheromone ** alpha) * ((1.0 / (pheromone + 1e-6)) ** beta)
        prob /= prob.sum()
        solution = np.random.rand(num_features) < prob
        all_solutions.append(solution)
        score = evaluate_solution(solution)
        scores.append(score)
        if score > best_score:
            best_score = score
            best_solution = solution

    pheromone *= (1 - evaporation_rate)
    for i in range(num_ants):
        pheromone += scores[i] * all_solutions[i]

end_train = time.time()

selected_features = [f for f, s in zip(feature_names, best_solution) if s == 1]

# ------------------------ Final Model ------------------------
X_selected = X[selected_features]
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, stratify=y, random_state=42)
X_train = X_train + np.random.normal(0, 0.01, X_train.shape)
X_test = X_test + np.random.normal(0, 0.01, X_test.shape)

final_model = RandomForestClassifier(n_estimators=50, max_depth=8, class_weight='balanced', random_state=42)
final_model.fit(X_train, y_train)

y_train_pred = final_model.predict(X_train)
y_test_pred = final_model.predict(X_test)

train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)

# ------------------------ Save Outputs ------------------------
joblib.dump(final_model, "sybil_aco_rf_model.pkl")
X_test.to_csv("X_test_aco.csv", index=False)
y_test.to_csv("y_test_aco.csv", index=False)

# ------------------------ Output (Styled like Bayesian RF) ------------------------
print(f"\n✅ ACO RF Model saved as 'sybil_aco_rf_model.pkl'")
print(f"📌 Selected Features ({len(selected_features)}): {selected_features}")
print(f"🎯 Training Accuracy: {train_acc * 100:.4f}%")
print(f"🎯 Testing Accuracy : {test_acc * 100:.4f}%")
print(f"🕒 Training Time    : {end_train - start_train:.4f} seconds")
