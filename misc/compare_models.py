from pathlib import Path
from time import perf_counter
import json
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score, classification_report
)
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


def evaluate(actual, predicted, scores):
    metrics = {}
    metrics["accuracy"] = accuracy_score(actual, predicted)
    metrics["balanced_accuracy"] = balanced_accuracy_score(actual, predicted)
    metrics["precision"] = precision_score(actual, predicted, zero_division=0)
    metrics["recall"] = recall_score(actual, predicted, zero_division=0)
    metrics["f1"] = f1_score(actual, predicted, zero_division=0)
    metrics["f1_macro"] = f1_score(actual, predicted, average="macro", zero_division=0)
    metrics["roc_auc"] = roc_auc_score(actual, scores)
    metrics["average_precision"] = average_precision_score(actual, scores)
    return metrics


def main():
    root = Path(__file__).resolve().parent
    results_dir = root / "results"
    output_dir = results_dir / "comparison"
    output_dir.mkdir(exist_ok=True, parents=True)

    df = pd.read_csv(root / "data" / "cleaned_startup_data.csv")
    df = df.sort_values("startup_id").reset_index(drop=True)
    assert df["startup_id"].is_unique
    assert df["startup_id"].notna().all()
    assert df["target"].isin([0, 1]).all()

    numeric_columns = [
        "funding_total_usd", "funding_rounds", "founded_at_year",
        "first_funding_at_year", "last_funding_at_year"
    ]
    categorical_columns = ["country_code", "first_category"]
    X = df[numeric_columns + categorical_columns].copy()
    X["funding_total_usd"] = np.log1p(X["funding_total_usd"])
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scoring = [
        "accuracy", "balanced_accuracy", "precision", "recall",
        "f1", "f1_macro", "roc_auc", "average_precision"
    ]

    model_results = []
    for filename in ["random_forest", "svm"]:
        saved_predictions = pd.read_csv(results_dir / f"{filename}_predictions.csv")
        np.testing.assert_array_equal(saved_predictions["startup_id"], df.loc[X_test.index, "startup_id"])
        np.testing.assert_array_equal(saved_predictions["actual"], y_test)
        saved_metrics = pd.read_csv(results_dir / f"{filename}_metrics.csv")
        model_results.append(saved_metrics)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=3000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42)
    }
    parameter_grids = {
        "Logistic Regression": {
            "model__C": [0.1, 1, 10],
            "model__class_weight": [None, "balanced"]
        },
        "Decision Tree": {
            "model__max_depth": [3, 7, None],
            "model__min_samples_leaf": [1, 50]
        }
    }

    for model_name, estimator in models.items():
        print("Evaluating", model_name, flush=True)
        numeric_steps = [("imputer", SimpleImputer(strategy="median", add_indicator=True))]
        if model_name == "Logistic Regression":
            numeric_steps.append(("scaler", StandardScaler()))

        numeric_transformer = Pipeline(numeric_steps)
        categorical_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("encoder", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20))
        ])
        preprocessor = ColumnTransformer([
            ("numeric", numeric_transformer, numeric_columns),
            ("categorical", categorical_transformer, categorical_columns)
        ])
        model = Pipeline([("preprocessor", preprocessor), ("model", estimator)])

        baseline = cross_validate(
            model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=1, error_score="raise"
        )
        search = GridSearchCV(
            model, parameter_grids[model_name], scoring=scoring, refit="f1_macro",
            cv=cv, n_jobs=1, error_score="raise", return_train_score=True
        )
        start = perf_counter()
        search.fit(X_train, y_train)
        search_seconds = perf_counter() - start
        best_model = search.best_estimator_

        start = perf_counter()
        predicted = best_model.predict(X_test)
        prediction_seconds = perf_counter() - start
        scores = best_model.predict_proba(X_test)[:, 1]
        test_metrics = evaluate(y_test, predicted, scores)

        summary = {}
        summary["model"] = model_name
        summary["baseline_cv_f1_macro"] = baseline["test_f1_macro"].mean()
        summary["cv_f1_macro"] = search.best_score_
        summary["cv_f1_macro_std"] = search.cv_results_["std_test_f1_macro"][search.best_index_]
        for metric in scoring:
            summary["test_" + metric] = test_metrics[metric]
        summary["search_seconds"] = search_seconds
        summary["refit_seconds"] = search.refit_time_
        summary["prediction_seconds"] = prediction_seconds
        summary["best_parameters"] = json.dumps(search.best_params_)
        summary["train_rows"] = len(X_train)
        summary["test_rows"] = len(X_test)

        result_name = model_name.lower().replace(" ", "_")
        metrics_table = pd.DataFrame([summary])
        metrics_table.to_csv(output_dir / f"{result_name}_metrics.csv", index=False)
        model_results.append(metrics_table)

        predictions = pd.DataFrame()
        predictions["startup_id"] = df.loc[X_test.index, "startup_id"].to_numpy()
        predictions["actual"] = y_test.to_numpy()
        predictions["predicted"] = predicted
        predictions["score"] = scores
        predictions.to_csv(output_dir / f"{result_name}_predictions.csv", index=False)

        cv_results = pd.DataFrame(search.cv_results_).sort_values("rank_test_f1_macro")
        cv_results.to_csv(output_dir / f"{result_name}_search.csv", index=False)
        cv_summary = pd.DataFrame(index=scoring, dtype=float)
        for metric in scoring:
            cv_summary.loc[metric, "baseline_mean"] = baseline["test_" + metric].mean()
            cv_summary.loc[metric, "baseline_std"] = baseline["test_" + metric].std()
            cv_summary.loc[metric, "tuned_mean"] = search.cv_results_["mean_test_" + metric][search.best_index_]
            cv_summary.loc[metric, "tuned_std"] = search.cv_results_["std_test_" + metric][search.best_index_]
        cv_summary.index.name = "metric"
        cv_summary.to_csv(output_dir / f"{result_name}_cv.csv")

        report = classification_report(
            y_test, predicted, target_names=["Closed", "Acquired"],
            output_dict=True, zero_division=0
        )
        pd.DataFrame(report).T.to_csv(output_dir / f"{result_name}_classification_report.csv")

        train_predictions = best_model.predict(X_train)
        train_test = pd.DataFrame(index=["Training", "Test"], dtype=float)
        train_test.loc["Training", "accuracy"] = accuracy_score(y_train, train_predictions)
        train_test.loc["Test", "accuracy"] = test_metrics["accuracy"]
        train_test.loc["Training", "f1_macro"] = f1_score(y_train, train_predictions, average="macro")
        train_test.loc["Test", "f1_macro"] = test_metrics["f1_macro"]
        train_test.to_csv(output_dir / f"{result_name}_train_test.csv")
        print(model_name, search.best_params_, "CV macro F1:", round(search.best_score_, 4), flush=True)

    comparison = pd.concat(model_results, ignore_index=True)
    comparison = comparison.sort_values("cv_f1_macro", ascending=False)
    comparison.to_csv(results_dir / "final_model_comparison.csv", index=False)
    print(comparison[["model", "cv_f1_macro", "test_accuracy", "test_f1_macro", "test_roc_auc"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
