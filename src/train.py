"""Train the final price model and save it to models/price_model.joblib."""
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (mean_absolute_error,
                             mean_absolute_percentage_error, r2_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.features import COLUMN_NAMES, FEATURES, add_features

DATA_PATH = "data/raw/bucharest_house_prices.csv"
MODEL_PATH = "models/price_model.joblib"


def main():
    # Load and clean
    df = pd.read_csv(DATA_PATH).rename(columns=COLUMN_NAMES)
    df = df.drop_duplicates().reset_index(drop=True)
    df = add_features(df)

    X = df[FEATURES]
    y = df["price"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Best configuration found in notebooks/02_modeling.ipynb
    preprocessor = ColumnTransformer(
        transformers=[
            ("onehot", OneHotEncoder(handle_unknown="ignore"), ["sector", "score"]),
        ],
        remainder="passthrough",
    )
    model = TransformedTargetRegressor(
        regressor=Pipeline(steps=[
            ("preprocess", preprocessor),
            ("model", RandomForestRegressor(
                n_estimators=300, max_features=0.5,
                random_state=42, n_jobs=-1,
            )),
        ]),
        func=np.log1p,
        inverse_func=np.expm1,
    )
    model.fit(X_train, y_train)

    # Evaluate on the held-out test set
    pred = model.predict(X_test)
    print(f"MAE:  {mean_absolute_error(y_test, pred):,.0f} EUR")
    print(f"MAPE: {mean_absolute_percentage_error(y_test, pred) * 100:.1f}%")
    print(f"R2:   {r2_score(y_test, pred):.3f}")

    joblib.dump(model, MODEL_PATH, compress=3)
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()