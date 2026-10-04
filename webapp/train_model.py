import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pycountry
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from webapp.model import ARTIFACTS, NUMERIC_COLUMNS, CATEGORICAL_COLUMNS, FEATURES, log_funding


def main():
    root = Path(__file__).resolve().parents[1]
    data_path = root / "data" / "cleaned_startup_data.csv"
    df = pd.read_csv(data_path).sort_values("startup_id").reset_index(drop=True)
    assert df["startup_id"].is_unique and df["startup_id"].notna().all()
    assert df["target"].isin([0, 1]).all()
    X_train, X_test, y_train, y_test = train_test_split(
        df[FEATURES], df["target"], test_size=0.2, stratify=df["target"], random_state=42
    )

    numeric = Pipeline([("imputer", SimpleImputer(strategy="median", add_indicator=True))])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("encoder", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20))
    ])
    preprocessing = ColumnTransformer([
        ("numeric", numeric, NUMERIC_COLUMNS),
        ("categorical", categorical, CATEGORICAL_COLUMNS)
    ])
    model = Pipeline([
        ("log_funding", FunctionTransformer(log_funding, validate=False)),
        ("preprocessor", preprocessing),
        ("model", RandomForestClassifier(
            n_estimators=100, max_depth=None, min_samples_leaf=5, random_state=42, n_jobs=1
        ))
    ])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    scores = model.predict_proba(X_test)[:, 1]

    reference_path = root / "results" / "random_forest_predictions.csv"
    if reference_path.exists():
        reference = pd.read_csv(reference_path)
        np.testing.assert_array_equal(reference["startup_id"], df.loc[X_test.index, "startup_id"])
        np.testing.assert_array_equal(reference["predicted"], predictions)
        np.testing.assert_allclose(reference["score"], scores, atol=1e-12, rtol=0)

    countries = []
    for code in sorted(X_train["country_code"].dropna().unique()):
        country = pycountry.countries.get(alpha_3=code)
        countries.append({"code": code, "name": country.name if country else code})
    countries.sort(key=lambda country: country["name"])

    metadata = {
        "model": "Random Forest", "sklearn_version": sklearn.__version__,
        "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "train_rows": len(X_train), "test_rows": len(X_test),
        "accuracy": accuracy_score(y_test, predictions),
        "macro_f1": f1_score(y_test, predictions, average="macro"),
        "roc_auc": roc_auc_score(y_test, scores),
        "countries": countries,
        "categories": sorted(X_train["first_category"].dropna().unique()),
        "max_funding": float(X_train["funding_total_usd"].max()),
        "max_rounds": int(X_train["funding_rounds"].max()),
        "last_training_year": 2015,
        "parameters": {"n_estimators": 100, "max_depth": None, "min_samples_leaf": 5, "random_state": 42}
    }
    ARTIFACTS.mkdir(exist_ok=True)
    joblib.dump(model, ARTIFACTS / "random_forest.joblib", compress=3)
    (ARTIFACTS / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print("Saved Random Forest and preprocessing pipeline to", ARTIFACTS)
    print("Test accuracy:", round(metadata["accuracy"], 4))
    print("Notebook predictions verified." if reference_path.exists() else "No reference predictions available.")


if __name__ == "__main__":
    main()
