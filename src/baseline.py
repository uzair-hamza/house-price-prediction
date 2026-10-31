"""Baseline models: Linear Regression and Ridge.

Same train/test split (seed 42) is used by every model script so results
are directly comparable. Features are standardized inside a pipeline.
Metrics are written to results/metrics.json.
"""
import json
import os
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

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
    return metrics


def main():
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURES]
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED)

    results = {}

    results["LinearRegression"] = evaluate(
        "LinearRegression",
        make_pipeline(StandardScaler(), LinearRegression()),
        X_train, X_test, y_train, y_test)

    # Ridge with a small alpha grid, picked on the TRAINING set only
    best_alpha, best_r2 = None, -1e9
    for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
        model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        model.fit(X_train, y_train)
        r2 = r2_score(y_train, model.predict(X_train))
        if r2 > best_r2:
            best_r2, best_alpha = r2, alpha
    print(f"Ridge: best alpha on train = {best_alpha}")
    results["Ridge"] = evaluate(
        f"Ridge(alpha={best_alpha})",
        make_pipeline(StandardScaler(), Ridge(alpha=best_alpha)),
        X_train, X_test, y_train, y_test)

    with open(METRICS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nMetrics saved to {METRICS_PATH}")


if __name__ == "__main__":
    main()
