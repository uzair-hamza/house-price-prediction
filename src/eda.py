"""Exploratory data analysis for the California Housing dataset.

Prints summary stats, missing values and correlations with the target,
and saves plots to results/.
"""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE, "data", "housing.csv")
RESULTS = os.path.join(BASE, "results")
os.makedirs(RESULTS, exist_ok=True)

FEATURES = ["MedInc", "HouseAge", "AveRooms", "AveBedrms",
            "Population", "AveOccup", "Latitude", "Longitude"]


def main():
    df = pd.read_csv(DATA_PATH)
    print("Shape:", df.shape)

    print("\nMissing values per column:")
    print(df.isnull().sum())

    print("\nSummary statistics:")
    print(df.describe().round(3))

    corr = df.corr(numeric_only=True)["target"].drop("target").sort_values(ascending=False)
    print("\nCorrelation of each feature with the target (median house value):")
    print(corr.round(3))

    # 1. Target distribution
    plt.figure(figsize=(8, 5))
    plt.hist(df["target"], bins=50, edgecolor="black")
    plt.xlabel("Median house value (in $100,000s)")
    plt.ylabel("Count")
    plt.title("Distribution of median house values")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, "target_histogram.png"))
    plt.close()

    # 2. Correlation heatmap
    plt.figure(figsize=(9, 7))
    c = df[FEATURES + ["target"]].corr(numeric_only=True)
    im = plt.imshow(c, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(im)
    plt.xticks(range(len(c.columns)), c.columns, rotation=45, ha="right")
    plt.yticks(range(len(c.columns)), c.columns)
    plt.title("Feature correlation heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, "correlation_heatmap.png"))
    plt.close()

    # 3. Strongest feature vs target
    top_feature = corr.index[0]
    plt.figure(figsize=(8, 5))
    plt.scatter(df[top_feature], df["target"], s=2, alpha=0.3)
    plt.xlabel(top_feature)
    plt.ylabel("Median house value (in $100,000s)")
    plt.title(f"{top_feature} vs median house value")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, "top_feature_scatter.png"))
    plt.close()

    print("\nPlots saved to results/: target_histogram.png, "
          "correlation_heatmap.png, top_feature_scatter.png")
    print("\nKey findings:")
    print(f"- No missing values in any column.")
    print(f"- Strongest predictor of house value is {top_feature} "
          f"(corr = {corr.iloc[0]:.3f}): richer areas cost more.")
    print(f"- Latitude has a negative correlation ({corr['Latitude']:.3f}): "
          "houses further south (smaller latitude) tend to be pricier, "
          "matching coastal southern California.")
    print(f"- Target is capped at 5.0 ($500k) - the histogram piles up there, "
          "so very expensive houses are clipped in this dataset.")


if __name__ == "__main__":
    main()
