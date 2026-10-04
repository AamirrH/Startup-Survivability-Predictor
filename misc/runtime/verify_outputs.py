import ast
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import platform
import tokenize
import nbformat
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split

root = Path(__file__).resolve().parents[2]
for path in (root / 'notebooks').glob('*.ipynb'):
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    for cell in notebook.cells:
        if cell.cell_type == 'code':
            ast.parse(cell.source)
            assert cell.execution_count is not None, path
            assert not any(token.type == tokenize.COMMENT for token in tokenize.generate_tokens(io.StringIO(cell.source).readline)), path
            assert not any(output.output_type == 'error' for output in cell.outputs), path
    print(path.name, 'executed, valid, no code comments or error outputs')

data = pd.read_csv(root / 'data' / 'cleaned_startup_data.csv').sort_values('startup_id').reset_index(drop=True)
train, test = train_test_split(data, test_size=0.2, stratify=data.target, random_state=42)
assert set(train.startup_id).isdisjoint(test.startup_id)
assert len(data) == 11787 and data.startup_id.is_unique
predictions = {}
for slug in ['random_forest', 'svm']:
    prediction = pd.read_csv(f'results/{slug}_predictions.csv')
    summary = pd.read_csv(f'results/{slug}_metrics.csv').iloc[0]
    np.testing.assert_array_equal(prediction.startup_id, test.startup_id)
    np.testing.assert_array_equal(prediction.actual, test.target)
    actual, predicted, score = prediction.actual, prediction.predicted, prediction.score
    recalculated = {
        'accuracy': accuracy_score(actual, predicted),
        'balanced_accuracy': balanced_accuracy_score(actual, predicted),
        'precision': precision_score(actual, predicted),
        'recall': recall_score(actual, predicted),
        'f1': f1_score(actual, predicted),
        'f1_macro': f1_score(actual, predicted, average='macro'),
        'roc_auc': roc_auc_score(actual, score),
        'average_precision': average_precision_score(actual, score)
    }
    for metric, value in recalculated.items():
        assert np.isclose(value, summary['test_' + metric]), (slug, metric)
    search = pd.read_csv(f'results/{slug}_search.csv')
    assert len(search) == 6
    assert np.isclose(search.mean_test_f1_macro.max(), summary.cv_f1_macro)
    predictions[slug] = prediction
    print(slug, 'all eight metrics independently reproduced; test IDs and targets match the shared split')

np.testing.assert_array_equal(predictions['random_forest'].startup_id, predictions['svm'].startup_id)
comparison = pd.read_csv('results/model_comparison.csv')
assert set(comparison.model) == {'Random Forest', 'SVM'}
assert comparison.cv_f1_macro.is_monotonic_decreasing
for slug, model in [('random_forest', 'Random Forest'), ('svm', 'SVM')]:
    saved = pd.read_csv(f'results/{slug}_metrics.csv').iloc[0]
    combined = comparison.set_index('model').loc[model]
    for column in ['cv_f1_macro', 'test_accuracy', 'test_f1_macro', 'test_roc_auc', 'search_seconds']:
        assert np.isclose(saved[column], combined[column])

metadata = {
    'verified_run_date': '2026-10-04',
    'python_version': platform.python_version(),
    'packages': {name: importlib.metadata.version(name) for name in ['pandas', 'numpy', 'scipy', 'scikit-learn', 'matplotlib', 'joblib', 'nbformat', 'nbclient', 'ipykernel']},
    'dataset_url': 'https://www.kaggle.com/datasets/yanmaksi/big-startup-secsees-fail-dataset-from-crunchbase',
    'raw_sha256': hashlib.sha256((root / 'data' / 'big_startup_secsees_dataset.csv').read_bytes()).hexdigest(),
    'cleaned_sha256': hashlib.sha256((root / 'data' / 'cleaned_startup_data.csv').read_bytes()).hexdigest(),
    'random_state': 42,
    'train_rows': len(train),
    'test_rows': len(test),
    'cross_validation_folds': 3,
    'search_candidates_per_model': 6,
    'selection_metric': 'f1_macro',
    'checks': ['all notebooks fully executed', 'no notebook error outputs', 'no code comments', 'identical stratified test companies and labels', 'all eight test metrics reproduced from saved predictions', 'highest validation macro F1 candidate selected', 'comparison matches individual saved metrics']
}
Path('results/run_metadata.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
print(comparison[['model', 'cv_f1_macro', 'test_accuracy', 'test_f1_macro', 'test_roc_auc']].to_string(index=False))
