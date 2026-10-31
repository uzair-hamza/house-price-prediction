"""Load the California Housing dataset and save a local copy as CSV.

Source: sklearn.datasets.fetch_california_housing (20,640 rows, 8 features).
The CSV copy in data/housing.csv makes the project reproducible without
re-downloading the dataset.
"""
import os
import pandas as pd
from sklearn.datasets import fetch_california_housing

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "housing.csv")


def main():
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame
    df = df.rename(columns={"MedHouseVal": "target"})
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    df.to_csv(DATA_PATH, index=False)
    print(f"Saved {df.shape[0]} rows x {df.shape[1]} columns to {DATA_PATH}")
    print("\nColumns:", list(df.columns))
    print("\nFirst rows:")
    print(df.head())
    print("\nSummary statistics:")
    print(df.describe())


if __name__ == "__main__":
    main()
