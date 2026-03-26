import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_excel("sybil_attack_predictions.xlsx")

# Ensure "Predicted Sybil" is an integer after handling NaN values
df["Predicted Sybil"] = df["Predicted Sybil"].fillna(0).astype(int)

# Define marker sizes for better visualization
marker_sizes = df["Predicted Sybil"].map({0: 10, 1: 50}).fillna(10)  # Default size for unknown cases

# --------------------------
# Plot 1: Scatter Plot - Vehicle Positions (Sybil vs Normal)
# --------------------------
plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=df, x="Position X", y="Position Y",
    hue="Predicted Sybil",
    palette={0: "blue", 1: "red"},  # Normal = Blue, Sybil = Red
    alpha=0.6,
    size=df["Predicted Sybil"],  # Adjust size based on Sybil status
    sizes={0: 20, 1: 80}  # Larger for Sybil vehicles
)
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.title("Vehicle Positions: Sybil vs Normal")
plt.legend(title="Vehicle Type", labels=["Normal", "Sybil"])
plt.show()

# --------------------------
# Plot 2: Pie Chart - Sybil vs Normal Vehicles
# --------------------------
sybil_counts = df["Predicted Sybil"].value_counts()

# Dynamically set labels based on presence of Sybil vehicles
labels = ["Normal Vehicles"] if 1 not in sybil_counts.index else ["Normal Vehicles", "Sybil Vehicles"]
colors = ["blue"] if 1 not in sybil_counts.index else ["blue", "red"]

plt.figure(figsize=(6, 6))
plt.pie(sybil_counts, labels=labels, autopct="%1.1f%%", colors=colors, startangle=90)
plt.title("Proportion of Sybil vs Normal Vehicles")
plt.show()
