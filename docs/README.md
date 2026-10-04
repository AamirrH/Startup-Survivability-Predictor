# Startup outcome model comparison

Random Forest is now finalized for the UI phase. The responsive Flask app lives in [webapp/](../webapp/README.md). Run `.\.venv\Scripts\python.exe -m webapp.app` from the repository root after installing `webapp/requirements.txt`; open `http://127.0.0.1:8000`. See the app README for phone access and deployment instructions.

Separate, executed Jupyter notebooks implement scikit-learn Random Forest and RBF SVM. All model code uses library implementations and has no code comments. Short Markdown explanations describe the methodology.

The [final four-model comparison](FINAL_MODEL_COMPARISON.md) now reviews the added Logistic Regression and Decision Tree notebooks and compares all four algorithms under one evaluation protocol. The supplied LR/tree notebooks retain their original code and saved outputs. The RF/SVM notebooks have been simplified while preserving their content and results. The no-code-comments statement above applies to the RF/SVM implementation; the submitted LR/tree notebooks contain their original comments.

After running RF and SVM, run `.\.venv\Scripts\python.exe misc/compare_models.py` from the repository root to reproduce the common LR/tree benchmark and update `results/final_model_comparison.csv`. The two-model results and original instructions below remain available for the RF/SVM experiment.

## Run

Use Python 3.12, install `misc/requirements.txt`, and open Jupyter from the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r misc/requirements.txt
.\.venv\Scripts\python.exe -m notebook
```

Run notebooks from top to bottom in this order:

1. [`Fail_Dataset_from_Crunchbase.ipynb`](../notebooks/Fail_Dataset_from_Crunchbase.ipynb): corrected EDA and export of `data/cleaned_startup_data.csv`.
2. `Random_Forest.ipynb`: baseline validation, tuning, and Random Forest evaluation.
3. `SVM.ipynb`: baseline validation, tuning, SVM evaluation, and the combined comparison.

The cleaned CSV is included, so either model notebook can run independently without rerunning the EDA. The EDA downloads the original public [Kaggle dataset](https://www.kaggle.com/datasets/yanmaksi/big-startup-secsees-fail-dataset-from-crunchbase) if `data/big_startup_secsees_dataset.csv` is absent. Alternatively, download and extract that file into `data/`. The original source CSV is excluded from Git.

## Cleaning corrections

The supplied folder contained the EDA notebook but no exported dataset. Its saved outputs showed 66,368 source records and 11,787 acquired/closed records; the downloaded source matches these counts.

- Convert `funding_total_usd` to numeric. The previous cleaning treated it as text and removed it because of its many unique values.
- Keep country and the first listed category instead of discarding all high-cardinality categorical fields. The first category is a simplification, not a claim that it is the primary category. Fine-grained location fields are omitted.
- Remove the duplicated date-processing step. Treat malformed dates and years outside 1800–2015 as missing, using an explicitly assumed historical cutoff based on this file's funding-date distribution. Do not substitute the current year or guess corrections to values such as 1015 or 2105.
- Treat reversed funding dates and founding dates later than last funding as missing in the affected fields. The latter is a conservative assumption: pre-incorporation funding can also explain such records, so these dates are not proven typos. In this dataset, five invalid date values and 411 founding/last-funding inconsistencies were marked missing, without dropping the companies.
- Check duplicate company identifiers before splitting. Retain IDs only for auditing predictions, never as features.
- Leave missing values in the cleaned CSV. Training pipelines learn medians, encodings, and SVM scaling separately inside each training fold, avoiding the previous full-data median-imputation leakage.

The binary labels remain **closed = 0, acquired = 1**, as in the original EDA. The 53,034 operating companies and 1,547 IPO companies are excluded. Operating companies are not failures; IPO exclusion preserves this experiment's original scope.

## Evaluation design

Both algorithms use the same seven features and the same stratified 80/20 split with seed 42: 9,429 training rows and 2,358 test rows. Funding uses a fixed `log1p` transformation. Numeric missing indicators are retained. One-hot encoding groups categories with fewer than 20 training-fold examples; unseen categories are supported. Numeric features are standardized for SVM.

Each algorithm receives three-fold stratified cross-validation and a six-candidate `GridSearchCV`, using identical folds and **macro F1** for selection. The starting configuration is included in each grid. Random Forest searches maximum tree depth and minimum leaf size; RBF SVM searches `C` and `gamma`. This is a small initial search, not exhaustive optimization.

Each notebook includes:

- Baseline and tuned cross-validation means and standard deviations.
- Holdout accuracy, balanced accuracy, precision, recall, F1, macro F1, ROC AUC, and average precision.
- A classification report for both classes, confusion matrix, ROC curve, and precision-recall curve.
- A majority-class reference and training-versus-test scores to help identify overfitting.
- Approximate search, final-fit, and prediction timings.
- Saved search results, metrics, class reports, per-company predictions, and evaluation plots under `results/`.

Precision, recall, and binary F1 use acquired as the positive class. Macro F1 weighs both classes equally. SVM ROC/PR metrics use decision scores, not estimated probabilities. Random Forest uses positive-class probability scores. Timings are machine-dependent, and prediction time measures class prediction only. Cross-validation standard deviations are not confidence intervals.

`results/model_comparison.csv` ranks models by training cross-validation macro F1, alongside their test metrics. The last cell of either model notebook refreshes this table after both have run. A tuned candidate's selection score can be optimistic; it is not an independent generalization estimate.

## Measured results

| Metric | Random Forest | RBF SVM |
| --- | ---: | ---: |
| Baseline validation macro F1 | 0.7139 | 0.7255 |
| Tuned validation macro F1 | 0.7273 | 0.7263 |
| Validation macro F1 standard deviation | 0.0055 | 0.0083 |
| Test accuracy | 0.7426 | 0.7366 |
| Test balanced accuracy | 0.7410 | 0.7353 |
| Test precision (acquired) | 0.7322 | 0.7237 |
| Test recall (acquired) | 0.7144 | 0.7126 |
| Test F1 (acquired) | 0.7232 | 0.7181 |
| Test macro F1 | 0.7413 | 0.7355 |
| Test ROC AUC | 0.8266 | 0.8148 |
| Test average precision | 0.7907 | 0.7711 |

Random Forest narrowly leads the chosen validation metric, and also performs better on this holdout. The validation difference is only about 0.001, smaller than either model's fold-to-fold standard deviation; this is a provisional preference, not evidence of a decisive or statistically significant advantage. The majority-class reference achieves 52.93% test accuracy.

Selected settings: Random Forest uses 100 trees, unrestricted maximum depth, and `min_samples_leaf=5`; SVM uses an RBF kernel with `C=1` and `gamma=0.1`.

## When to tune

Tune **each candidate algorithm before finalizing the choice**, using a modest, comparable search budget. Comparing only defaults can disadvantage SVM. Choose the algorithm using training cross-validation, then use the reserved test set for final assessment. These notebooks already follow that workflow. If further decisions are driven by the reported test results, obtain a fresh final holdout or use nested cross-validation; repeatedly tuning against this same test set would make its estimate optimistic. See the official [scikit-learn model-selection guide](https://scikit-learn.org/stable/modules/grid_search.html) and [data-leakage guidance](https://scikit-learn.org/stable/common_pitfalls.html).

## What this experiment can establish

This task classifies historical **acquired versus closed** outcomes. It does not estimate survival probability over a defined future period, and acquisition is only a proxy for success. Funding totals and last funding dates may contain information recorded after the relevant outcome; the source lacks outcome dates and a fixed observation window to verify otherwise. An early-warning model needs features frozen at a defined prediction date and later outcomes, followed by temporal validation. Operating companies are censored/unresolved observations rather than a labeled test set.

Random Forest and SVM are established classical baselines; implementing them alone does not establish a comparison with all state-of-the-art methods. The shared split and metric scheme can support additional algorithms in the wider project.
