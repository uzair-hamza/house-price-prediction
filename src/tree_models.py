"""Tree models: Random Forest and Gradient Boosting.

Same train/test split (seed 42) as the baselines, so results are directly
comparable. Hyperparameters are picked on the TRAINING set only; the final
numbers are measured once on the held-out test set. All four models'
metrics end up in results/metrics.json, plus a predicted-vs-actual plot
for the best model.
"""
import json
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE, "data", "housing.csv")
RESULTS = os.path.join(BASE, "results")
METRICS_PATH = os.path.join(RESULTS, "metrics.json")
os.makedirs(RESULTS, exist_ok=True)

FEATURES = ["MedInc", "HouseAge", "AveRooms", "AveBedrms",
            "Population", "AveOccup", "Latitude", "Longitude"]
SEED = 42


def evaluate(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    metrics = {
        "rmse": float(mean_squared_error(y_test, pred) ** 0.5),
        "mae": float(mean_absolute_error(y_test, pred)),
        "r2": float(r2_score(y_test, pred)),
    }
    print(f"{name}: RMSE={metrics['rmse']:.4f}  "
          f"MAE={metrics['mae']:.4f}  R2={metrics['r2']:.4f}")
    return metrics, pred


def pick_on_train(candidates, X_train, y_train):
    best_name, best_model, best_r2 = None, None, -1e9
    for name, model in candidates:
        model.fit(X_train, y_train)
        r2 = r2_score(y_train, model.predict(X_train))
        print(f"  train pick - {name}: train R2={r2:.4f}")
        if r2 > best_r2:
            best_name, best_model, best_r2 = name, model, r2
    print(f"  -> picked {best_name}")
    return best_name, best_model


def main():
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURES]
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED)

    with open(METRICS_PATH) as f:
        results = json.load(f)

    print("Random Forest candidates:")
    rf_name, rf = pick_on_train(
        [(f"RandomForest(n={n})",
          RandomForestRegressor(n_estimators=n, random_state=SEED, n_jobs=-1))
         for n in (100, 200)],
        X_train, y_train)
    results[rf_name], rf_pred = evaluate(rf_name, rf,
                                         X_train, X_test, y_train, y_test)

    print("Gradient Boosting candidates:")
    gb_name, gb = pick_on_train(
        [(f"HistGradientBoosting(iter={i})",
          HistGradientBoostingRegressor(max_iter=i, random_state=SEED))
         for i in (100, 200)],
        X_train, y_train)
    results[gb_name], gb_pred = evaluate(gb_name, gb,
                                         X_train, X_test, y_train, y_test)

    with open(METRICS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nMetrics saved to {METRICS_PATH}")

    # Predicted vs actual for the best model on test R2
    best = max(results.items(), key=lambda kv: kv[1]["r2"])[0]
    pred = rf_pred if best == rf_name else gb_pred
    plt.figure(figsize=(7, 7))
    plt.scatter(y_test, pred, s=2, alpha=0.3)
    lo, hi = y_test.min(), y_test.max()
    plt.plot([lo, hi], [lo, hi], "r--", label="perfect prediction")
    plt.xlabel("Actual median house value (in $100,000s)")
    plt.ylabel("Predicted median house value (in $100,000s)")
    plt.title(f"Predicted vs actual - {best}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, "predicted_vs_actual.png"))
    plt.close()
    print(f"Best model on test R2: {best}; plot saved to "
          "results/predicted_vs_actual.png")


if __name__ == "__main__":
    main()
