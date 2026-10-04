# Startup Outcome Prediction: Final Model Comparison

## Recommendation

**Random Forest is the recommended candidate for the current acquired-versus-closed classification task.** It has the highest mean cross-validation macro F1, test macro F1, accuracy, ROC AUC, and average precision among the four models in the common evaluation.

SVM is a close alternative: its validation macro F1 is only **0.0010** lower. This difference is smaller than either model's fold-to-fold standard deviation, so the results do not establish a statistically significant advantage. Logistic Regression has the highest acquired-class recall and binary F1; the choice could change if identifying acquisitions were the primary objective. The agreed selection metric here is macro F1, which weighs both outcomes equally.

## 1. Review of the submitted notebooks

The repository contains [Logistic Regression](Logistic_regression.ipynb), [Decision Tree](Decision_tree.ipynb), [Random Forest](Random_Forest.ipynb), and [SVM](SVM.ipynb).

The Logistic Regression and Decision Tree notebooks demonstrate default models, tuning, classification metrics, confusion matrices, ROC curves, and feature analysis. The Decision Tree's saved results also clearly demonstrate reduced overfitting after tuning.

Several differences prevent directly ranking their original saved scores against RF/SVM:

| Item | Submitted LR / Decision Tree | RF / SVM |
| --- | --- | --- |
| Saved input schema | 7 columns: six numeric predictors and target | 9 columns: seven predictors, company ID, target |
| Funding representation | Precomputed `log_funding` and `funding_missing` | Funding transformed with `log1p`; missing indicators fitted inside the pipeline |
| Categorical predictors | None | Country and first listed category |
| Validation | 5 folds; tuning by ROC AUC | 3 shuffled stratified folds; tuning by macro F1 |
| Search configurations | LR: 24; tree: 144 | 6 each |
| Data loading | Colab-specific `/content/` path and upload | Project-local CSV |

The same row count and random seed do not establish identical test companies when data preparation or row order may differ. The six-feature CSV used for the submitted LR/tree runs is not present in the current checkout, so its upstream cleaning cannot be verified from those notebooks alone.

In Logistic Regression, `StandardScaler` is fitted on all training rows before `GridSearchCV`. This lets validation-fold information influence scaling. It does **not** fit on the held-out test set, but the scaler should still be fitted separately inside each validation fold. Also, simply changing the Colab path would leave both notebooks incompatible with the current CSV: their `drop(target)` feature selection would include company IDs and categorical text, and they do not supply the needed imputation/encoding.

The submitted notebooks remain unchanged. A separate [comparison script](compare_models.py) reruns LR/tree under the common protocol below. Their original tuning grids were reduced to six configurations for this benchmark; the new scores describe that explicitly defined experiment.

### Original saved results, for traceability

These values come from the submitted notebook outputs and describe their earlier six-feature experiment. They are not the basis of the final four-model ranking.

| Model | Version | Accuracy | Precision | Recall | Binary F1 | ROC AUC |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | Default | 0.6972 | 0.6996 | 0.6252 | 0.6603 | 0.7606 |
| Logistic Regression | Tuned | 0.6925 | 0.6771 | 0.6631 | 0.6700 | 0.7606 |
| Decision Tree | Default | 0.6531 | 0.6460 | 0.5820 | 0.6123 | 0.6433 |
| Decision Tree | Tuned | 0.7129 | 0.7084 | 0.6631 | 0.6850 | 0.7820 |

## 2. Data cleaning and splitting — rubric: 3 marks

Source: [Crunchbase startup dataset on Kaggle](https://www.kaggle.com/datasets/yanmaksi/big-startup-secsees-fail-dataset-from-crunchbase). The corrected [EDA notebook](Fail_Dataset_from_Crunchbase.ipynb) exports the current [cleaned CSV](cleaned_startup_data.csv).

- **Labels:** closed = 0, acquired = 1. Keep 11,787 labeled companies: 6,238 closed and 5,549 acquired. Exclude 53,034 operating companies with unresolved outcomes and 1,547 IPO companies to preserve this binary task.
- **Cleaning:** parse funding numerically; keep unknown amounts missing; flag invalid dates; use the documented 1800–2015 date range; flag founding dates later than last funding conservatively. Pre-incorporation funding can explain some flagged dates, so these are not all proven typos.
- **Features:** log funding, funding rounds, founding year, first funding year, last funding year, country, and first listed category. The category representation is a simplification of a multi-category field.
- **Identity checks:** company IDs are unique, used to audit splits, and excluded from prediction features.
- **Split:** sort by company ID, then stratify 80/20 with seed 42. Training: **9,429** companies; test: **2,358**, comprising 1,248 closed and 1,110 acquired. All four test ID sequences and labels were checked for equality.
- **Preprocessing:** median imputation with missing indicators and categorical encoding are fitted inside each training fold. Rare categories have fewer than 20 training-fold occurrences. Unknown categories are supported. LR and SVM standardize numeric features; tree models do not require scaling.

Missing values in the cleaned CSV are intentional. Filling them inside training pipelines prevents information from validation/test rows from influencing learned preprocessing.

## 3. Common model comparison — rubric: 4 marks

All four models use the same data, features, split, and `StratifiedKFold(n_splits=3, shuffle=True, random_state=42)`. Each receives six candidate configurations, with its starting configuration included. `GridSearchCV` chooses the highest mean **validation macro F1**, then refits on all training rows. The test set supplies descriptive performance after selection.

### Hyperparameter search and selected settings

| Model | Six-candidate search | Selected settings |
| --- | --- | --- |
| Random Forest | Depth: None, 10, 20; minimum leaf size: 1, 5 | 100 trees; depth=None; minimum leaf size=5 |
| RBF SVM | C: 0.1, 1, 10; gamma: scale, 0.1 | C=1; gamma=0.1 |
| Logistic Regression | C: 0.1, 1, 10; class weight: None, balanced | C=0.1; balanced class weights; L2 regularization; lbfgs solver |
| Decision Tree | Depth: 3, 7, None; minimum leaf size: 1, 50 | Depth=None; minimum leaf size=50; Gini criterion |

These are small, comparable searches rather than exhaustive optimization. LR/tree retain their respective model families, while using the corrected common preprocessing and selection protocol.

### Validation results: basis for model selection

| Rank | Model | Baseline macro F1 | Tuned macro F1 | Fold standard deviation |
| ---: | --- | ---: | ---: | ---: |
| 1 | Random Forest | 0.7139 | **0.7273** | 0.0055 |
| 2 | SVM | 0.7255 | 0.7263 | 0.0083 |
| 3 | Logistic Regression | 0.7116 | 0.7151 | 0.0083 |
| 4 | Decision Tree | 0.6640 | 0.7054 | 0.0075 |

The tree benefits most from this tuning search. Baseline and tuned values above are validation scores, so they should not be confused with the earlier notebooks' default-versus-tuned test scores. Fold standard deviations are descriptive variability, not confidence intervals. Selecting the best candidate can also make its reported validation score optimistic.

### Test results on identical companies

Acquired (1) is the positive class for precision, recall, binary F1, ROC AUC, and average precision. Macro F1 averages the two class-specific F1 scores equally.

| Model | Accuracy | Balanced accuracy | Precision | Recall | Binary F1 | Macro F1 | ROC AUC | Average precision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Random Forest | **0.7426** | **0.7410** | **0.7322** | 0.7144 | 0.7232 | **0.7413** | **0.8266** | **0.7907** |
| SVM | 0.7366 | 0.7353 | 0.7237 | 0.7126 | 0.7181 | 0.7355 | 0.8148 | 0.7711 |
| Logistic Regression | 0.7316 | 0.7329 | 0.6986 | **0.7559** | **0.7261** | 0.7314 | 0.8121 | 0.7833 |
| Decision Tree | 0.7167 | 0.7167 | 0.6925 | 0.7162 | 0.7042 | 0.7162 | 0.7927 | 0.7456 |

The majority-class reference achieves 0.5293 accuracy, 0.5000 balanced accuracy, 0.3461 macro F1, and 0.5000 ROC AUC. All four models improve on it. Accuracy measures overall correct classification; balanced accuracy averages class recalls. ROC AUC assesses ranking across thresholds, and average precision summarizes the precision-recall curve. SVM decision scores are not probabilities.

### Practical interpretation

| Model | Finding in this experiment | Tradeoff |
| --- | --- | --- |
| Random Forest | Best validation macro F1 and strongest overall test results | An ensemble is less directly interpretable than one tree or a linear model |
| SVM | Closest competitor on validation macro F1 | Scaling is essential; scores require calibration before being presented as probabilities |
| Logistic Regression | Highest acquisition recall and binary F1; competitive ROC AUC | Lower precision and macro F1 at the current decision rule |
| Decision Tree | Substantial improvement after limiting leaf size | Lowest tuned validation macro F1 among these four candidates |

For the common rerun, LR training/test accuracy is 0.7253/0.7316 and tree training/test accuracy is 0.7330/0.7167. These gaps do not suggest severe overfitting in those selected configurations, although similar scores alone cannot prove generalization. Detailed RF/SVM training-versus-test tables remain in their notebooks. Wall-clock timings are saved in the metric CSVs; they are approximate and were not measured as an isolated hardware benchmark.

## 4. RF and SVM code simplification

Both notebooks now use smaller cells, straightforward loops, named intermediate tables, and explicit metric assignments. Dense dictionary comprehensions and dictionary unpacking were removed. The original explanatory Markdown, split, features, pipelines, model definitions, grids, eight metrics, classification reports, dummy reference, training/test checks, timings, plots, and saved-result formats are preserved.

Both simplified notebooks were executed from start to finish. Their company-level predictions, cross-validation summaries, selected parameters, and test metrics match the previously committed results. Timings naturally change between runs. No code comments were added, and no implementation was hidden in a helper module to simplify these two notebooks.

## 5. Scope and next decision

Select **Random Forest provisionally** for this project's current classification stage, while retaining SVM as the close comparison model. The four methods are established classical baselines; this experiment does not establish superiority over all state-of-the-art approaches.

Continue choosing hyperparameters and algorithms through training validation. The existing holdout was already examined in the earlier RF/SVM work; this report extends that benchmark. Additional tuning informed by these test results requires a fresh final holdout or nested validation for an independent assessment.

The present labels distinguish acquisition from closure. They do not measure survival over a defined future period. Funding totals and last funding dates may include history recorded after the outcome, and the dataset lacks outcome dates needed to verify feature timing. Before interpreting predictions for operating startups as future survival, define a prediction date and horizon, freeze earlier features, and evaluate later outcomes with temporal validation. IPO exclusion also limits the meaning of the positive class to acquisition.

## Reproduce and inspect

From the project folder, run both simplified notebooks, then run:

```powershell
.\.venv\Scripts\python.exe compare_models.py
```

The script trains LR/tree under the common protocol, verifies their test company sequence against RF/SVM, and builds the four-model CSV. Rerun it after changing either model notebook's results; this Markdown report describes the verified run accompanying it.

- [Four-model metrics](results/final_model_comparison.csv)
- [Logistic Regression validation](results/comparison/logistic_regression_cv.csv) and [predictions](results/comparison/logistic_regression_predictions.csv)
- [Decision Tree validation](results/comparison/decision_tree_cv.csv) and [predictions](results/comparison/decision_tree_predictions.csv)
- [Random Forest evaluation plots](results/random_forest_evaluation.png)
- [SVM evaluation plots](results/svm_evaluation.png)
- [Verification record](results/comparison/verification.json)

The original two-model `results/model_comparison.csv` is retained to preserve the RF/SVM notebooks' existing behavior. `results/final_model_comparison.csv` is the combined four-model result.
