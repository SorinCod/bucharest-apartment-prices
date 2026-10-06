import pandas as pd

COLUMN_NAMES = {
    "Nr Camere": "rooms",
    "Suprafata": "area",
    "Etaj": "floor",
    "Total Etaje": "total_floors",
    "Sector": "sector",
    "Scor": "score",
    "Pret": "price",
}

FEATURES = ["rooms", "area", "floor", "total_floors",
            "sector", "score", "ground_floor", "top_floor"]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered floor features. Used by both training and the app."""
    df = df.copy()
    df["ground_floor"] = (df["floor"] <= 0).astype(int)
    df["top_floor"] = (df["floor"] == df["total_floors"]).astype(int)
    return df