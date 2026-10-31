# House Price Prediction (California Housing)

Predicting median house values from census data, using the California
Housing dataset (20,640 districts, 8 features like median income, house
age, location). I wanted to see how far simple models go on a real
regression problem, and where they break.

## Dataset

- Source: `sklearn.datasets.fetch_california_housing`
- 20,640 rows, 8 features + 1 target (median house value, in $100,000s)
- A local copy is saved at `data/housing.csv` so the project runs without
  re-downloading
- Train/test split: 80/20, seed 42, same split for every model

What the data showed (see `src/eda.py`):
- No missing values anywhere
- Median income (`MedInc`) is by far the strongest predictor (correlation 0.688)
- The target is **capped at 5.0** ($500,000). Expensive houses are clipped,
  so no model can predict above 5.0. You can see the pile-up at 5.0 in the
  histogram and in the predicted-vs-actual plot.

## Results (held-out test set)

| Model | RMSE | MAE | R2 |
|---|---|---|---|
| Linear Regression | 0.7456 | 0.5332 | 0.5758 |
| Ridge (alpha=0.01) | 0.7456 | 0.5332 | 0.5758 |
| Random Forest (200 trees) | 0.5043 | 0.3268 | 0.8059 |
| Hist Gradient Boosting (200 iters) | 0.4541 | 0.3015 | 0.8427 |

The best model (gradient boosting) is off by about $45,400 on average
(RMSE 0.4541 in $100k units).

Hyperparameters were picked on the **training** set only. The test set was
used exactly once per model, at the end, so these numbers are honest
held-out results.

## What I tried and what I learned

- Linear regression only reached R2 0.58. House prices do not follow a
  straight line from these features (location effects are non-linear), so
  a linear model underfits.
- Ridge with a small alpha grid ended up identical to plain linear
  regression. Regularization was not the problem here, non-linearity was,
  so Ridge added nothing.
- Random Forest jumped to R2 0.81, but it overfits: train R2 0.974 vs
  test R2 0.806. It memorizes the training districts.
- Gradient boosting won (R2 0.84) and generalizes better: train R2 0.899
  vs test R2 0.843. Boosting corrects its own mistakes instead of
  memorizing, which suits this data.
- The 5.0 target cap is a hard ceiling on accuracy. A chunk of the
  remaining error comes from houses worth more than $500k that the
  dataset records as exactly 5.0.

## Project structure

```
house-price-prediction/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── housing.csv            # local copy of the dataset
├── src/
│   ├── load_data.py           # downloads dataset, saves CSV, prints overview
│   ├── eda.py                 # stats, correlations, plots
│   ├── baseline.py            # Linear Regression + Ridge
│   └── tree_models.py         # Random Forest + Gradient Boosting
└── results/
    ├── metrics.json           # all test metrics
    ├── target_histogram.png
    ├── correlation_heatmap.png
    ├── top_feature_scatter.png
    └── predicted_vs_actual.png
```

## How to run

```bash
pip install -r requirements.txt
python src/load_data.py    # downloads the dataset once, saves data/housing.csv
python src/eda.py          # analysis + plots in results/
python src/baseline.py     # linear models, writes results/metrics.json
python src/tree_models.py  # tree models, updates metrics, saves plot
```

`load_data.py` needs internet the first time (it fetches the dataset from
sklearn). Everything after that reads the local CSV.
