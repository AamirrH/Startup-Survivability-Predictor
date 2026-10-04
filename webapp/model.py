from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
NUMERIC_COLUMNS = [
    "funding_total_usd", "funding_rounds", "founded_at_year",
    "first_funding_at_year", "last_funding_at_year"
]
CATEGORICAL_COLUMNS = ["country_code", "first_category"]
FEATURES = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS


def log_funding(data):
    data = data.copy()
    data["funding_total_usd"] = np.log1p(data["funding_total_usd"])
    return data


def load_model():
    model = joblib.load(ARTIFACTS / "random_forest.joblib")
    metadata = json.loads((ARTIFACTS / "metadata.json").read_text(encoding="utf-8"))
    return model, metadata


def predict_startup(model, features):
    data = pd.DataFrame([features], columns=FEATURES)
    probabilities = model.predict_proba(data)[0]
    prediction = int(model.predict(data)[0])
    return {
        "prediction": prediction,
        "outcome": "Success pattern" if prediction == 1 else "Failure pattern",
        "recorded_outcome": "Acquired" if prediction == 1 else "Closed",
        "acquired_score": float(probabilities[1]),
        "closed_score": float(probabilities[0])
    }
