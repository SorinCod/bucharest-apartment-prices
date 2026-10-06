# Bucharest Apartment Price Estimator

A machine learning project that estimates apartment prices in Bucharest from basic listing details: area, number of rooms, floor, sector and zone score. It covers the full workflow, from exploratory analysis and model comparison to a small Streamlit app.

![App screenshot](assets/app.png)

## Results

Final model: **Random Forest with a log-transformed target**, evaluated on a held-out test set of 671 apartments.

| Metric | Value |
|---|---|
| MAE | €16,031 |
| MAPE | 16.1% |
| R² | 0.813 |

For reference, a baseline that always predicts the median price has an MAE of about €39,000.

### Model comparison (5-fold cross-validation on the training set)

| Model | MAE (EUR) | MAPE | R² |
|---|---|---|---|
| Linear regression | 20,809 ± 1,148 | 21.7% | 0.754 |
| Linear regression, log target | 19,272 ± 1,360 | 18.3% | 0.764 |
| Random Forest | 17,725 ± 927 | 17.2% | 0.778 |
| **Random Forest, log target** | 17,811 ± 1,024 | **16.7%** | 0.773 |
| Gradient Boosting | 18,485 ± 956 | 18.4% | 0.783 |
| Gradient Boosting, log target | 18,218 ± 1,142 | 17.3% | 0.785 |

The tree-based models clearly beat the linear ones. The differences among the tree-based models are smaller than the variation between folds, so they perform comparably. Random Forest with a log target was chosen for its lowest percentage error.

### Error by price range (test set)

| Price range | Apartments | MAE (EUR) | MAPE |
|---|---|---|---|
| under 60k | 195 | 6,982 | 15% |
| 60k to 100k | 277 | 12,655 | 16% |
| 100k to 150k | 122 | 19,866 | 16% |
| over 150k | 77 | 45,017 | 20% |

## Key findings

- **Area is by far the strongest predictor**, followed by zone score and sector.
- **Log-transforming the price** lowered the percentage error for every model, because price effects are multiplicative rather than additive.
- **Hyperparameter tuning brought no meaningful gain** (about €150 in MAE). Performance is limited by the available features, not by the algorithm.
- **Engineered floor features** (ground floor, top floor) showed a small effect in the exploratory analysis but added no measurable value to the model.

## Limitations

- The data is from **March 2019**, so estimates reflect 2019 price levels.
- The dataset has only seven columns. Important price drivers such as build year, condition and exact location are missing.
- The model is **less reliable above €150,000** and tends to underestimate those apartments.
- 175 duplicate rows were removed on the assumption that they were repeated listings.
- The `score` column is not documented in the source. It was interpreted from the data as a zone rating in which 1 is the most expensive zone.

## Project structure

```
bucharest-apartment-prices/
├── app.py                      # Streamlit app
├── requirements.txt
├── data/
│   └── raw/                    # dataset goes here (not included)
├── models/
│   └── price_model.joblib      # trained model
├── notebooks/
│   ├── 01_exploration.ipynb    # exploratory data analysis
│   └── 02_modeling.ipynb       # model comparison, tuning, evaluation
└── src/
    ├── features.py             # shared feature engineering
    └── train.py                # reproducible training script
```

## How to run

```bash
git clone https://github.com/SorinCod/bucharest-apartment-prices.git
cd bucharest-apartment-prices

python -m venv .venv
source .venv/Scripts/activate    # Windows (Git Bash)
# source .venv/bin/activate      # macOS / Linux

pip install -r requirements.txt
streamlit run app.py
```

### Retraining the model

1. Download the dataset from Kaggle (link below).
2. Save the CSV as `data/raw/bucharest_house_prices.csv`.
3. Run:

```bash
python -m src.train
```

The script prints the test metrics and saves the model to `models/price_model.joblib`. With the fixed random seeds, it reproduces the results above.

## Data

[Bucharest House Price Dataset](https://www.kaggle.com/datasets/denisadutca/bucharest-house-price-dataset) on Kaggle: 3,529 listings from March 2019. The dataset is not included in this repository.

## Tech stack

Python, pandas, scikit-learn, matplotlib, seaborn, Streamlit