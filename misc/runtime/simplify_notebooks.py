from pathlib import Path
import copy
import nbformat


def cell(source):
    return nbformat.v4.new_code_cell(source.strip())


for filename in ['Random_Forest.ipynb', 'SVM.ipynb']:
    path = Path(__file__).resolve().parents[2] / 'notebooks' / filename
    notebook = nbformat.read(path, as_version=4)
    original = copy.deepcopy(notebook)
    nbformat.write(original, Path(__file__).resolve().parent / ('before_' + filename))
    replacements = {}
    replacements[1] = [cell(notebook.cells[1].source)]
    if filename == 'Random_Forest.ipynb':
        replacements[1][0].source = replacements[1][0].source.replace('OneHotEncoder, StandardScaler', 'OneHotEncoder')
    replacements[3] = [cell('''
df = pd.read_csv("data/cleaned_startup_data.csv")
df = df.sort_values("startup_id").reset_index(drop=True)

assert df["startup_id"].notna().all()
assert df["startup_id"].is_unique
assert df["target"].isin([0, 1]).all()

numeric_columns = [
    "funding_total_usd", "funding_rounds", "founded_at_year",
    "first_funding_at_year", "last_funding_at_year"
]
categorical_columns = ["country_code", "first_category"]

X = df[numeric_columns + categorical_columns].copy()
X["funding_total_usd"] = np.log1p(X["funding_total_usd"])
y = df["target"]
'''), cell('''
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
assert set(X_train.index).isdisjoint(X_test.index)

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

split_counts = pd.DataFrame()
split_counts["Train"] = y_train.value_counts().sort_index()
split_counts["Test"] = y_test.value_counts().sort_index()
split_counts.index = ["Closed", "Acquired"]
display(split_counts)
print("Training rows:", len(X_train), "Test rows:", len(X_test))
''')]
    preprocessing, modeling = notebook.cells[5].source.split('model = Pipeline(', 1)
    model_source, scoring_source = ('model = Pipeline(' + modeling).split('scoring = ', 1)
    replacements[5] = [cell(preprocessing), cell(model_source), cell('''
scoring = [
    "accuracy", "balanced_accuracy", "precision", "recall",
    "f1", "f1_macro", "roc_auc", "average_precision"
]
''')]
    replacements[7] = [cell('''
baseline_scores = cross_validate(
    model, X_train, y_train, cv=cv, scoring=scoring,
    n_jobs=1, error_score="raise"
)

baseline_summary = pd.DataFrame(index=scoring, columns=["mean", "std"], dtype=float)
for metric in scoring:
    fold_scores = baseline_scores["test_" + metric]
    baseline_summary.loc[metric, "mean"] = fold_scores.mean()
    baseline_summary.loc[metric, "std"] = fold_scores.std()

display(baseline_summary.round(4))
''')]
    tuning, tuning_tables = notebook.cells[9].source.split('cv_results = ', 1)
    grid_results = 'cv_results = ' + tuning_tables.split('cv_summary = ', 1)[0]
    replacements[9] = [cell(tuning), cell(grid_results), cell('''
cv_summary = baseline_summary.rename(columns={"mean": "baseline_mean", "std": "baseline_std"})
best_index = search.best_index_

for metric in scoring:
    cv_summary.loc[metric, "tuned_mean"] = search.cv_results_["mean_test_" + metric][best_index]
    cv_summary.loc[metric, "tuned_std"] = search.cv_results_["std_test_" + metric][best_index]

cv_summary.index.name = "metric"
cv_summary.to_csv(results_dir / f"{result_name}_cv.csv")
display(cv_summary.round(4))
''')]
    predictions, evaluation = notebook.cells[11].source.split('def evaluate(', 1)
    training = notebook.cells[11].source.split('train_pred = ', 1)[1]
    replacements[11] = [cell(predictions), cell('''
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

test_metrics = evaluate(y_test, y_pred, y_score)
'''), cell('''
dummy = DummyClassifier(strategy="most_frequent")
dummy.fit(X_train, y_train)
dummy_pred = dummy.predict(X_test)
dummy_score = dummy.predict_proba(X_test)[:, 1]
dummy_metrics = evaluate(y_test, dummy_pred, dummy_score)

test_results = pd.DataFrame([test_metrics, dummy_metrics])
test_results.index = [model_name, "Majority baseline"]
display(test_results.round(4))

report = classification_report(
    y_test, y_pred, target_names=["Closed", "Acquired"], digits=4, zero_division=0
)
print(report)
'''), cell('''
train_pred = best_model.predict(X_train)

train_test_results = pd.DataFrame(index=["Training", "Test"], dtype=float)
train_test_results.loc["Training", "accuracy"] = accuracy_score(y_train, train_pred)
train_test_results.loc["Test", "accuracy"] = test_metrics["accuracy"]
train_test_results.loc["Training", "f1_macro"] = f1_score(y_train, train_pred, average="macro")
train_test_results.loc["Test", "f1_macro"] = test_metrics["f1_macro"]

display(train_test_results.round(4))
print("Final training time in seconds:", round(search.refit_time_, 3))
print("Test prediction time in seconds:", round(prediction_seconds, 3))
''')]
    replacements[14] = [cell('''
summary = {}
summary["model"] = model_name
summary["baseline_cv_f1_macro"] = baseline_summary.loc["f1_macro", "mean"]
summary["cv_f1_macro"] = search.best_score_
summary["cv_f1_macro_std"] = cv_summary.loc["f1_macro", "tuned_std"]

for metric in scoring:
    summary["test_" + metric] = test_metrics[metric]

summary["search_seconds"] = search_seconds
summary["refit_seconds"] = search.refit_time_
summary["prediction_seconds"] = prediction_seconds
summary["best_parameters"] = json.dumps(search.best_params_)
summary["train_rows"] = len(X_train)
summary["test_rows"] = len(X_test)

metrics_table = pd.DataFrame([summary])
metrics_table.to_csv(results_dir / f"{result_name}_metrics.csv", index=False)
'''), cell('''
predictions = pd.DataFrame()
predictions["startup_id"] = df.loc[X_test.index, "startup_id"].to_numpy()
predictions["actual"] = y_test.to_numpy()
predictions["predicted"] = y_pred
predictions["score"] = y_score
predictions.to_csv(results_dir / f"{result_name}_predictions.csv", index=False)

report = classification_report(
    y_test, y_pred, target_names=["Closed", "Acquired"],
    output_dict=True, zero_division=0
)
report_table = pd.DataFrame(report).T
report_table.to_csv(results_dir / f"{result_name}_classification_report.csv")
'''), cell('''
metric_files = ["random_forest_metrics.csv", "svm_metrics.csv"]
model_results = []

for filename in metric_files:
    path = results_dir / filename
    if path.exists():
        model_results.append(pd.read_csv(path))

comparison = pd.concat(model_results, ignore_index=True)
comparison = comparison.sort_values("cv_f1_macro", ascending=False)
comparison.to_csv(results_dir / "model_comparison.csv", index=False)

comparison_columns = [
    "model", "baseline_cv_f1_macro", "cv_f1_macro", "cv_f1_macro_std",
    "test_accuracy", "test_precision", "test_recall", "test_f1_macro", "test_roc_auc"
]
display(comparison[comparison_columns].round(4))

if len(comparison) == 2:
    print("Candidate selected by validation macro F1:", comparison.iloc[0]["model"])
else:
    print("Run the other model notebook to complete the comparison.")
''')]
    notebook.cells = []
    for index, original_cell in enumerate(original.cells):
        if index in replacements:
            notebook.cells.extend(replacements[index])
        else:
            notebook.cells.append(original_cell)
    for notebook_cell in notebook.cells:
        if notebook_cell.cell_type == 'code':
            notebook_cell.outputs = []
            notebook_cell.execution_count = None
    nbformat.validate(notebook)
    nbformat.write(notebook, path)
    old_markdown = [c.source for c in original.cells if c.cell_type == 'markdown']
    new_markdown = [c.source for c in notebook.cells if c.cell_type == 'markdown']
    assert old_markdown == new_markdown
    print(filename, 'all original Markdown preserved;', len(notebook.cells), 'shorter cells')
