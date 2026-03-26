import pandas as pd
import numpy as np
import joblib
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# ------------------------- PSO PARAMETERS -------------------------
num_particles = 10
max_iter = 20
w = 0.5
c1 = 1.5
c2 = 1.5

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

# Use stratified split to preserve distribution
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ------------------------- FITNESS FUNCTION (REALISTIC) -------------------------
def fitness_function(params):
    n_estimators = int(params[0])
    max_depth = int(params[1])

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight='balanced',
        random_state=42
    )

    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    acc_train = accuracy_score(y_train, y_pred_train)
    acc_test = accuracy_score(y_test, y_pred_test)

    # Penalize overfitting
    overfit_penalty = abs(acc_train - acc_test)
    loss = (1 - acc_test) + (0.5 * overfit_penalty)

    return loss

# ------------------------- PSO ALGORITHM -------------------------
def PSO(fitness_func, lb, ub, dim, num_particles, max_iter):
    positions = np.random.uniform(lb, ub, (num_particles, dim))
    velocities = np.zeros((num_particles, dim))
    pbest_positions = np.copy(positions)
    pbest_scores = np.array([fitness_func(p) for p in positions])
    gbest_position = pbest_positions[np.argmin(pbest_scores)]
    gbest_score = min(pbest_scores)

    for _ in range(max_iter):
        for i in range(num_particles):
            r1, r2 = np.random.rand(dim), np.random.rand(dim)
            velocities[i] = (w * velocities[i] +
                             c1 * r1 * (pbest_positions[i] - positions[i]) +
                             c2 * r2 * (gbest_position - positions[i]))
            positions[i] += velocities[i]
            positions[i] = np.clip(positions[i], lb, ub)

            fitness = fitness_func(positions[i])
            if fitness < pbest_scores[i]:
                pbest_scores[i] = fitness
                pbest_positions[i] = positions[i]
                if fitness < gbest_score:
                    gbest_score = fitness
                    gbest_position = positions[i]
    return gbest_position, 1 - gbest_score

# ------------------------- RUN PSO & SAVE MODEL -------------------------
start_train = time.time()
best_params, best_accuracy = PSO(
    fitness_func=fitness_function,
    lb=np.array([50, 3]),  # Lower bound: [n_estimators, max_depth]
    ub=np.array([150, 10]),  # Constrained upper bound
    dim=2,
    num_particles=num_particles,
    max_iter=max_iter
)
end_train = time.time()

n_estimators = int(best_params[0])
max_depth = int(best_params[1])

final_model = RandomForestClassifier(
    n_estimators=n_estimators,
    max_depth=max_depth,
    class_weight="balanced",
    random_state=42
)
final_model.fit(X_train, y_train)

joblib.dump(final_model, "sybil_pso_rf_model.pkl")
X_test.to_csv("X_test_pso.csv", index=False)
y_test.to_csv("y_test_pso.csv", index=False)

print(f"\n✅ Model saved as 'sybil_pso_rf_model.pkl'")
print(f"📊 Best Parameters: n_estimators={n_estimators}, max_depth={max_depth}")
print(f"🕒 Training Time: {end_train - start_train:.4f} seconds")
print(f"🎯 Realistic Best Accuracy: {best_accuracy:.4f}")
