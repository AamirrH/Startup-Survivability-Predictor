import json
from pathlib import Path
from textwrap import dedent

ROOT = Path(__file__).resolve().parents[2]


def markdown(source):
    return {"cell_type": "markdown", "metadata": {}, "source": dedent(source).strip().splitlines(True)}


def code(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": dedent(source).strip().splitlines(True)}


def save(name, cells):
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    for index, cell in enumerate(cells):
        cell["id"] = f"cell-{index:02d}"
    (ROOT / "notebooks" / name).write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")


save("Fail_Dataset_from_Crunchbase.ipynb", [
    markdown("""
    # Startup data: EDA and corrected cleaning

    Source: [Kaggle Crunchbase dataset](https://www.kaggle.com/datasets/yanmaksi/big-startup-secsees-fail-dataset-from-crunchbase).

    Run this notebook first to recreate `data/cleaned_startup_data.csv`. The original EDA's binary task is preserved: closed = 0, acquired = 1. Operating companies have unresolved outcomes and IPO companies are outside this particular comparison; neither is treated as failure.

    Corrections: parse funding as numeric instead of dropping it as high-cardinality text; retain country and the first listed category; remove the repeated date-conversion step; defer median imputation, encoding, and scaling to the training pipelines. Missing values in the exported CSV are intentional.
    """),
    code("""
    from pathlib import Path
    import io
    import urllib.request
    import zipfile
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from IPython.display import display

    data_path = Path("data/big_startup_secsees_dataset.csv")
    if not data_path.exists():
        url = "https://www.kaggle.com/api/v1/datasets/download/yanmaksi/big-startup-secsees-fail-dataset-from-crunchbase"
        archive_bytes = urllib.request.urlopen(url, timeout=90).read()
        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
            data_path.parent.mkdir(exist_ok=True)
            data_path.write_bytes(archive.read("big_startup_secsees_dataset.csv"))

    df = pd.read_csv(data_path)
    print("Rows and columns:", df.shape)
    display(df.head())
    """),
    code("""
    df.info()
    display(df.isna().sum().rename("missing_values").to_frame())
    display(df["status"].value_counts().rename("startups").to_frame())
    df["status"].value_counts().plot.bar(color="teal", rot=0)
    plt.title("Recorded startup outcomes")
    plt.ylabel("Startups")
    plt.tight_layout()
    plt.show()
    """),
    code("""
    print("Exact duplicate rows:", df.duplicated().sum())
    df = df.drop_duplicates().copy()
    df["status"] = df["status"].str.strip().str.lower()
    assert df["permalink"].notna().all()
    assert df["permalink"].is_unique

    df_clean = df[df["status"].isin(["acquired", "closed"])].copy()
    df_clean["target"] = df_clean["status"].map({"closed": 0, "acquired": 1})
    df_clean = df_clean.rename(columns={"permalink": "startup_id"})
    print("Labeled startups:", len(df_clean))
    print("Excluded operating startups:", df["status"].eq("operating").sum())
    print("Excluded IPO startups:", df["status"].eq("ipo").sum())
    display(df_clean["target"].value_counts().sort_index())
    """),
    markdown("""
    ## Numeric and date corrections

    Unknown funding (`-`) becomes missing rather than zero. Negative funding and non-positive funding-round counts are invalid. Date years outside 1800–2015 become missing: 2015 is an explicit historical cutoff inferred from this file's funding-date distribution, not today's year. This assumes a snapshot through 2015 and should be revisited for a newer dataset. Reversed first/last funding dates are removed; founding dates later than the last funding date are treated as inconsistent. No attempt is made to guess typo corrections.
    """),
    code("""
    funding = df_clean["funding_total_usd"].astype("string").str.replace(",", "", regex=False)
    df_clean["funding_total_usd"] = pd.to_numeric(funding, errors="coerce")
    df_clean.loc[df_clean["funding_total_usd"] < 0, "funding_total_usd"] = np.nan
    df_clean["funding_rounds"] = pd.to_numeric(df_clean["funding_rounds"], errors="coerce")
    df_clean.loc[df_clean["funding_rounds"] <= 0, "funding_rounds"] = np.nan

    date_columns = ["founded_at", "first_funding_at", "last_funding_at"]
    for column in date_columns:
        dates = pd.to_datetime(df_clean[column], format="%Y-%m-%d", errors="coerce")
        valid_year = dates.dt.year.between(1800, 2015)
        print(column, "invalid dates:", (df_clean[column].notna() & ~valid_year).sum())
        df_clean[column] = dates.where(valid_year)

    reversed_funding = df_clean["first_funding_at"] > df_clean["last_funding_at"]
    print("Reversed funding dates:", reversed_funding.sum())
    df_clean.loc[reversed_funding, ["first_funding_at", "last_funding_at"]] = pd.NaT
    late_founding = df_clean["founded_at"] > df_clean["last_funding_at"]
    print("Founding after last funding:", late_founding.sum())
    df_clean.loc[late_founding, "founded_at"] = pd.NaT

    for column in date_columns:
        df_clean[column + "_year"] = df_clean[column].dt.year
    """),
    markdown("""
    ## Modeling features

    The first listed category is a simple representation of a multi-category field; it is not necessarily the company's primary category. Country is retained, while redundant detailed locations and identifying fields are omitted. `startup_id` is exported only to verify that both notebooks use the same companies in the test set; it is never a predictor.
    """),
    code("""
    df_clean["country_code"] = df_clean["country_code"].str.strip().str.upper().replace("", np.nan)
    df_clean["first_category"] = df_clean["category_list"].str.split("|").str[0].str.strip().replace("", np.nan)
    columns = [
        "startup_id", "funding_total_usd", "funding_rounds",
        "founded_at_year", "first_funding_at_year", "last_funding_at_year",
        "country_code", "first_category", "target"
    ]
    df_final = df_clean[columns].sort_values("startup_id").reset_index(drop=True)
    assert df_final["target"].isin([0, 1]).all()
    assert df_final["startup_id"].is_unique
    df_final.to_csv("data/cleaned_startup_data.csv", index=False)
    print("Saved data/cleaned_startup_data.csv:", df_final.shape)
    display(df_final.head())
    display(df_final.isna().sum().rename("missing_before_training_imputation").to_frame())
    """),
    code("""
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    funding_groups = [
        np.log1p(df_final.loc[df_final["target"] == label, "funding_total_usd"].dropna())
        for label in [0, 1]
    ]
    axes[0].boxplot(funding_groups, showfliers=False)
    axes[0].set_xticks([1, 2], ["Closed", "Acquired"])
    axes[0].set_ylabel("log(1 + funding in USD)")
    axes[0].set_title("Funding by recorded outcome")
    correlations = df_final.select_dtypes(include="number").corr()["target"].drop("target")
    correlations.sort_values().plot.barh(ax=axes[1], color="teal")
    axes[1].set_title("Numeric feature correlations with target")
    plt.tight_layout()
    plt.show()
    """),
    markdown("""
    ## Interpretation limit

    These are retrospective outcome classifiers, not time-to-failure models. Funding totals and last funding dates describe the recorded company history and are not guaranteed to predate acquisition or closure. The dataset lacks the outcome dates and fixed observation window needed to establish prospective survival performance. A random holdout measures this historical classification task; deployment on operating startups requires separate validation and features measured before the prediction date.
    """),
])


for model_name, slug in [("Random Forest", "random_forest"), ("SVM", "svm")]:
    if slug == "random_forest":
        estimator_import = "from sklearn.ensemble import RandomForestClassifier"
        estimator = "RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1)"
        grid = '{"model__max_depth": [None, 10, 20], "model__min_samples_leaf": [1, 5]}'
        scaling = ""
        score_line = "y_score = best_model.predict_proba(X_test)[:, 1]"
    else:
        estimator_import = "from sklearn.svm import SVC"
        estimator = 'SVC(kernel="rbf", C=1.0, gamma="scale")'
        grid = '{"model__C": [0.1, 1, 10], "model__gamma": ["scale", 0.1]}'
        scaling = ',\n    ("scaler", StandardScaler())'
        score_line = "y_score = best_model.decision_function(X_test)"

    cells = [
        markdown(f"""
        # {model_name}: startup outcome prediction

        Uses `data/cleaned_startup_data.csv` from the corrected EDA notebook. Closed = 0; acquired = 1. Operating and IPO companies are excluded to preserve the original binary task. This is a historical acquired-versus-closed classifier, not a validated forecast of survival duration.

        Both models use the same seven features, stratified 80/20 split (seed 42), and three training-only cross-validation folds. Macro F1 is the selection metric because both outcomes matter. Baseline cross-validation is followed by six hyperparameter candidates per model. The held-out test set is used only after tuning.
        """),
        code(dedent(f"""
        from pathlib import Path
        from time import perf_counter
        import json
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        from IPython.display import display
        from sklearn.compose import ColumnTransformer
        from sklearn.dummy import DummyClassifier
        from sklearn.impute import SimpleImputer
        from sklearn.metrics import (
            accuracy_score, balanced_accuracy_score, precision_score, recall_score,
            f1_score, roc_auc_score, average_precision_score, classification_report,
            ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay
        )
        from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, GridSearchCV
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
        {estimator_import}

        model_name = "{model_name}"
        result_name = "{slug}"
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        """)),
        markdown("""
        ## Load and split

        Funding uses `log1p` to reduce its skew; this fixed transformation learns nothing from the data. Missing values are filled within the pipeline after splitting. IDs and labels are excluded from features. Rows are sorted by startup ID so the shared split is reproducible.
        """),
        code("""
        df = pd.read_csv("data/cleaned_startup_data.csv").sort_values("startup_id").reset_index(drop=True)
        assert df["startup_id"].notna().all() and df["startup_id"].is_unique
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
        assert set(X_train.index).isdisjoint(X_test.index)
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        split_counts = pd.DataFrame({
            "Train": y_train.value_counts(), "Test": y_test.value_counts()
        }).rename(index={0: "Closed", 1: "Acquired"})
        display(split_counts)
        print("Training rows:", len(X_train), "Test rows:", len(X_test))
        """),
        markdown("""
        ## Preprocessing and model

        Numeric values use training-fold medians. Categorical missing values become `Unknown`, then one-hot encoding groups categories occurring fewer than 20 times in that training fold. Unseen categories are handled during prediction. SVM additionally standardizes numeric features; trees do not need scaling. All learned preprocessing is fitted again within each cross-validation fold.
        """),
        code(f'''numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median", add_indicator=True)){scaling}
])
categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
    ("encoder", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20))
])
preprocessor = ColumnTransformer([
    ("numeric", numeric_transformer, numeric_columns),
    ("categorical", categorical_transformer, categorical_columns)
])
model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", {estimator})
])
scoring = {{
    "accuracy": "accuracy", "balanced_accuracy": "balanced_accuracy",
    "precision": "precision", "recall": "recall", "f1": "f1",
    "f1_macro": "f1_macro", "roc_auc": "roc_auc",
    "average_precision": "average_precision"
}}'''),
        markdown("## Baseline cross-validation"),
        code("""
        baseline_scores = cross_validate(
            model, X_train, y_train, cv=cv, scoring=scoring,
            n_jobs=1, error_score="raise"
        )
        baseline_summary = pd.DataFrame({
            "mean": {metric: baseline_scores["test_" + metric].mean() for metric in scoring},
            "std": {metric: baseline_scores["test_" + metric].std() for metric in scoring}
        })
        display(baseline_summary.round(4))
        """),
        markdown("""
        ## Small hyperparameter search

        Each algorithm receives six candidates and the same three folds. The default starting configuration is included in its grid. Select by mean validation macro F1, then refit the selected pipeline on all training rows. Search scores are selection estimates and may be optimistic; the holdout provides a separate assessment. Fold standard deviations are not confidence intervals.
        """),
        code(f'''parameter_grid = {grid}
search = GridSearchCV(
    model, parameter_grid, scoring=scoring, refit="f1_macro",
    cv=cv, n_jobs=1, error_score="raise", return_train_score=True
)
start = perf_counter()
search.fit(X_train, y_train)
search_seconds = perf_counter() - start
best_model = search.best_estimator_
print("Best parameters:", search.best_params_)
print("Best validation macro F1:", round(search.best_score_, 4))
print("Search time in seconds:", round(search_seconds, 2))

cv_results = pd.DataFrame(search.cv_results_).sort_values("rank_test_f1_macro")
cv_results.to_csv(results_dir / f"{{result_name}}_search.csv", index=False)
display(cv_results[[
    "params", "mean_train_f1_macro", "mean_test_f1_macro",
    "std_test_f1_macro", "mean_test_roc_auc", "mean_fit_time"
]].round(4))

cv_summary = baseline_summary.rename(columns={{"mean": "baseline_mean", "std": "baseline_std"}})
cv_summary["tuned_mean"] = [search.cv_results_["mean_test_" + metric][search.best_index_] for metric in scoring]
cv_summary["tuned_std"] = [search.cv_results_["std_test_" + metric][search.best_index_] for metric in scoring]
cv_summary.index.name = "metric"
cv_summary.to_csv(results_dir / f"{{result_name}}_cv.csv")
display(cv_summary.round(4))'''),
        markdown("""
        ## Held-out evaluation

        Precision, recall, F1, ROC AUC and average precision use acquired (1) as the positive class. Macro F1 averages both classes equally. The class report separately shows closed-startup performance. Average precision summarizes the precision-recall curve. SVM curves use decision scores, which are not probabilities. The majority-class dummy provides a simple reference.
        """),
        code(f'''start = perf_counter()
y_pred = best_model.predict(X_test)
prediction_seconds = perf_counter() - start
{score_line}

def evaluate(actual, predicted, scores):
    return {{
        "accuracy": accuracy_score(actual, predicted),
        "balanced_accuracy": balanced_accuracy_score(actual, predicted),
        "precision": precision_score(actual, predicted, zero_division=0),
        "recall": recall_score(actual, predicted, zero_division=0),
        "f1": f1_score(actual, predicted, zero_division=0),
        "f1_macro": f1_score(actual, predicted, average="macro", zero_division=0),
        "roc_auc": roc_auc_score(actual, scores),
        "average_precision": average_precision_score(actual, scores)
    }}

dummy = DummyClassifier(strategy="most_frequent")
dummy.fit(X_train, y_train)
test_metrics = evaluate(y_test, y_pred, y_score)
dummy_metrics = evaluate(y_test, dummy.predict(X_test), dummy.predict_proba(X_test)[:, 1])
display(pd.DataFrame([test_metrics, dummy_metrics], index=[model_name, "Majority baseline"]).round(4))
print(classification_report(y_test, y_pred, target_names=["Closed", "Acquired"], digits=4, zero_division=0))

train_pred = best_model.predict(X_train)
display(pd.DataFrame({{
    "accuracy": [accuracy_score(y_train, train_pred), test_metrics["accuracy"]],
    "f1_macro": [f1_score(y_train, train_pred, average="macro"), test_metrics["f1_macro"]]
}}, index=["Training", "Test"]).round(4))
print("Final training time in seconds:", round(search.refit_time_, 3))
print("Test prediction time in seconds:", round(prediction_seconds, 3))'''),
        code("""
        figure, axes = plt.subplots(1, 3, figsize=(16, 4))
        ConfusionMatrixDisplay.from_predictions(
            y_test, y_pred, display_labels=["Closed", "Acquired"],
            cmap="Blues", colorbar=False, ax=axes[0]
        )
        axes[0].set_title(model_name + ": confusion matrix")
        RocCurveDisplay.from_predictions(y_test, y_score, name=model_name, ax=axes[1])
        axes[1].plot([0, 1], [0, 1], "k--", label="Chance")
        axes[1].set_title("ROC curve: acquired is positive")
        axes[1].legend()
        PrecisionRecallDisplay.from_predictions(y_test, y_score, name=model_name, ax=axes[2])
        axes[2].axhline(y_test.mean(), color="black", linestyle="--", label="Acquired prevalence")
        axes[2].set_title("Precision-recall curve")
        axes[2].legend()
        plt.tight_layout()
        figure.savefig(results_dir / f"{result_name}_evaluation.png", dpi=150, bbox_inches="tight")
        plt.show()
        """),
        markdown("""
        ## Save results and compare models

        After both notebooks run, `model_comparison.csv` contains both results, ranked by training cross-validation macro F1. The test columns describe final holdout performance and must not guide repeated tuning. Run this last cell again to refresh the comparison after running the other notebook. Timing values are approximate wall-clock measurements on the current machine.
        """),
        code("""
        summary = {
            "model": model_name,
            "baseline_cv_f1_macro": baseline_summary.loc["f1_macro", "mean"],
            "cv_f1_macro": search.best_score_,
            "cv_f1_macro_std": search.cv_results_["std_test_f1_macro"][search.best_index_],
            **{"test_" + metric: value for metric, value in test_metrics.items()},
            "search_seconds": search_seconds,
            "refit_seconds": search.refit_time_,
            "prediction_seconds": prediction_seconds,
            "best_parameters": json.dumps(search.best_params_),
            "train_rows": len(X_train), "test_rows": len(X_test)
        }
        pd.DataFrame([summary]).to_csv(results_dir / f"{result_name}_metrics.csv", index=False)
        pd.DataFrame({
            "startup_id": df.loc[X_test.index, "startup_id"].to_numpy(),
            "actual": y_test.to_numpy(), "predicted": y_pred, "score": y_score
        }).to_csv(results_dir / f"{result_name}_predictions.csv", index=False)
        pd.DataFrame(classification_report(
            y_test, y_pred, target_names=["Closed", "Acquired"],
            output_dict=True, zero_division=0
        )).T.to_csv(results_dir / f"{result_name}_classification_report.csv")

        metric_files = [results_dir / "random_forest_metrics.csv", results_dir / "svm_metrics.csv"]
        comparison = pd.concat([pd.read_csv(path) for path in metric_files if path.exists()], ignore_index=True)
        comparison = comparison.sort_values("cv_f1_macro", ascending=False)
        comparison.to_csv(results_dir / "model_comparison.csv", index=False)
        display(comparison[[
            "model", "baseline_cv_f1_macro", "cv_f1_macro", "cv_f1_macro_std",
            "test_accuracy", "test_precision", "test_recall", "test_f1_macro", "test_roc_auc"
        ]].round(4))
        if len(comparison) == 2:
            print("Candidate selected by validation macro F1:", comparison.iloc[0]["model"])
        else:
            print("Run the other model notebook to complete the comparison.")
        """),
    ]
    save("Random_Forest.ipynb" if slug == "random_forest" else "SVM.ipynb", cells)
