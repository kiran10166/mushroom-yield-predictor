from pathlib import Path
import json
import joblib
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

# -------------------------
# Paths
# -------------------------

PROCESSED = Path("data/processed")
MODELS = Path("models")
REPORTS = Path("reports")
FIGURES = REPORTS / "figures"

FIGURES.mkdir(parents=True, exist_ok=True)

# -------------------------
# Load train/test data
# -------------------------

X_train = np.load(PROCESSED / "X_train.npy")
X_test = np.load(PROCESSED / "X_test.npy")

y_train = np.load(PROCESSED / "y_train.npy")
y_test = np.load(PROCESSED / "y_test.npy")

# -------------------------
# Train Random Forest
# -------------------------

rf = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

# -------------------------
# Predictions
# -------------------------

pred = rf.predict(X_test)

# -------------------------
# Metrics
# -------------------------

mae = mean_absolute_error(y_test, pred)

rmse = np.sqrt(
    mean_squared_error(y_test, pred)
)

r2 = r2_score(y_test, pred)

print("\n===== RANDOM FOREST RESULTS =====")

print(f"MAE  : {mae:.3f} kg")
print(f"RMSE : {rmse:.3f} kg")
print(f"R²   : {r2:.3f}")

# -------------------------
# Save metrics
# -------------------------

metrics = {
    "model": "RandomForestRegressor",
    "mae": float(mae),
    "rmse": float(rmse),
    "r2": float(r2)
}

with open(
    REPORTS / "metrics_random_forest.json",
    "w"
) as f:
    json.dump(metrics, f, indent=4)

# -------------------------
# Feature Importance
# -------------------------

labels = [
    "temperature_c",
    "humidity_pct",
    "co2_ppm"
]

importances = rf.feature_importances_

for name, importance in zip(labels, importances):
    print(
        f"{name}: {importance:.3f}"
    )

plt.figure(figsize=(6, 4))

plt.barh(
    labels,
    importances
)

plt.xlabel("Importance")
plt.title(
    "Random Forest Feature Importance"
)

plt.tight_layout()

plt.savefig(
    FIGURES / "rf_importance.png",
    dpi=150
)

plt.close()

# -------------------------
# Save model
# -------------------------

joblib.dump(
    rf,
    MODELS / "random_forest.joblib"
)

print(
    "\nModel saved:"
)
print(
    MODELS / "random_forest.joblib"
)

print(
    "\nFeature importance plot saved:"
)
print(
    FIGURES / "rf_importance.png"
)