import pandas as pd
import numpy as np
import joblib
import time
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# ------------------------- WOA PARAMETERS -------------------------
SearchAgents = 10
MaxIter = 20

# ------------------------- LOAD AND PREPARE DATA -------------------------
df = pd.read_excel("synthetic_75000_combined_traffic_data.xlsx", sheet_name="Sheet1")
df["Sybil"] = df["Sybil"].map({"Yes": 1, "No": 0})
df = df.dropna(subset=["Sybil"])

categorical_cols = ["Vehicle ID", "Vehicle Type", "Edge ID"]
for col in categorical_cols:
    df[col] = LabelEncoder().fit_transform(df[col])

df = df.drop(columns=["Timestamp"], errors="ignore")
X = df.drop(columns=["Sybil"])
y = df["Sybil"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# ------------------------- OBJECTIVE FUNCTION -------------------------
def fitness_function(params):
    n_estimators = int(params[0])
    max_depth = int(params[1])

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        class_weight='balanced'
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return 1 - acc  # minimize loss

# ------------------------- WOA ALGORITHM -------------------------
def WOA(fitness_func, lb, ub, dim, num_agents, max_iter):
    positions = np.random.uniform(lb, ub, (num_agents, dim))
    best_score = float("inf")
    best_pos = None

    for t in range(max_iter):
        for i in range(num_agents):
            score = fitness_func(positions[i])
            if score < best_score:
                best_score = score
                best_pos = positions[i]

        a = 2 - t * (2 / max_iter)
        for i in range(num_agents):
            r = np.random.rand()
            A = 2 * a * r - a
            C = 2 * r
            D = abs(C * best_pos - positions[i])
            positions[i] = best_pos - A * D

            positions[i] = np.clip(positions[i], lb, ub)
    
    return best_pos, 1 - best_score  # return accuracy

# ------------------------- RUN WOA -------------------------
start_train = time.time()
best_params, best_accuracy = WOA(
    fitness_func=fitness_function,
    lb=[10, 3],  # n_estimators, max_depth
    ub=[200, 20],
    dim=2,
    num_agents=SearchAgents,
    max_iter=MaxIter
)
end_train = time.time()

n_estimators = int(best_params[0])
max_depth = int(best_params[1])

# Final model
best_model = RandomForestClassifier(
    n_estimators=n_estimators,
    max_depth=max_depth,
    class_weight="balanced",
    random_state=42
)
best_model.fit(X_train, y_train)

joblib.dump(best_model, "sybil_woa_rf_model.pkl")
X_test.to_csv("X_test_woa.csv", index=False)
y_test.to_csv("y_test_woa.csv", index=False)

print(f"\n✅ WOA RF Model saved as 'sybil_woa_rf_model.pkl'")
print(f"📊 Best Parameters: n_estimators={n_estimators}, max_depth={max_depth}")
print(f"🕒 Training Time: {end_train - start_train:.4f} seconds")
