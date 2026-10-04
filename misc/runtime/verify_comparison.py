import ast
import io
import json
import subprocess
import sys
import tokenize
from pathlib import Path
import nbformat
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root / 'misc'))
from compare_models import evaluate

for name in ['Random_Forest.ipynb', 'SVM.ipynb']:
    notebook = nbformat.read(root / 'notebooks' / name, as_version=4)
    original = nbformat.read(Path(__file__).resolve().parent / ('before_' + name), as_version=4)
    nbformat.validate(notebook)
    assert [c.source for c in notebook.cells if c.cell_type == 'markdown'] == [c.source for c in original.cells if c.cell_type == 'markdown']
    for cell in notebook.cells:
        if cell.cell_type == 'code':
            ast.parse(cell.source)
            assert cell.execution_count is not None
            assert not any(t.type == tokenize.COMMENT for t in tokenize.generate_tokens(io.StringIO(cell.source).readline))
            assert not any(o.output_type == 'error' for o in cell.outputs)
    code_cells = [c for c in notebook.cells if c.cell_type == 'code']
    print(name, 'passed: executed, no comments/errors, all Markdown preserved; largest code cell:', max(len(c.source.splitlines()) for c in code_cells))

data = pd.read_csv(root / 'data' / 'cleaned_startup_data.csv').sort_values('startup_id').reset_index(drop=True)
train, test = train_test_split(data, test_size=0.2, stratify=data.target, random_state=42)
assert set(train.startup_id).isdisjoint(test.startup_id)
paths = {
    'Random Forest': Path('results/random_forest'),
    'SVM': Path('results/svm'),
    'Logistic Regression': Path('results/comparison/logistic_regression'),
    'Decision Tree': Path('results/comparison/decision_tree')
}
all_results = []
for model, prefix in paths.items():
    predictions = pd.read_csv(str(prefix) + '_predictions.csv')
    metrics = pd.read_csv(str(prefix) + '_metrics.csv')
    np.testing.assert_array_equal(predictions.startup_id, test.startup_id)
    np.testing.assert_array_equal(predictions.actual, test.target)
    recomputed = evaluate(predictions.actual, predictions.predicted, predictions.score)
    for name, value in recomputed.items():
        assert np.isclose(metrics.iloc[0]['test_' + name], value)
    search = pd.read_csv(str(prefix) + '_search.csv')
    assert len(search) == 6
    assert np.isclose(search.mean_test_f1_macro.max(), metrics.iloc[0].cv_f1_macro)
    assert metrics.iloc[0].train_rows == len(train)
    assert metrics.iloc[0].test_rows == len(test)
    all_results.append(metrics)
    print(model, 'same test IDs/targets; eight metrics independently verified')

for slug in ['random_forest', 'svm']:
    for suffix in ['predictions', 'cv']:
        file = f'results/{slug}_{suffix}.csv'
        previous = subprocess.run(['git', 'show', 'HEAD:' + file], check=True, capture_output=True, text=True, encoding='utf-8').stdout
        pd.testing.assert_frame_equal(pd.read_csv(io.StringIO(previous)), pd.read_csv(file))
    file = f'results/{slug}_metrics.csv'
    previous = subprocess.run(['git', 'show', 'HEAD:' + file], check=True, capture_output=True, text=True, encoding='utf-8').stdout
    old = pd.read_csv(io.StringIO(previous))
    new = pd.read_csv(file)
    stable_columns = [c for c in old.columns if not c.endswith('_seconds')]
    pd.testing.assert_frame_equal(old[stable_columns], new[stable_columns])
    print(slug, 'predictions, validation results and test metrics unchanged after simplification')

for file, old_file in [
    ('notebooks/Logistic_regression.ipynb', 'Logistic_regression.ipynb'),
    ('notebooks/Decision_tree.ipynb', 'Decision_tree.ipynb'),
    ('data/cleaned_startup_data.csv', 'cleaned_startup_data.csv'),
]:
    previous = subprocess.run(['git', 'show', 'HEAD:' + old_file], check=True, capture_output=True, text=True, encoding='utf-8').stdout
    assert previous.replace('\r\n', '\n') == (root / file).read_text(encoding='utf-8').replace('\r\n', '\n')
    print(file, 'unchanged')

comparison = pd.concat(all_results, ignore_index=True).sort_values('cv_f1_macro', ascending=False)
comparison.to_csv('results/final_model_comparison.csv', index=False)
metadata = {
    'models': list(paths),
    'train_rows': len(train),
    'test_rows': len(test),
    'cross_validation_folds': 3,
    'candidates_per_model': 6,
    'selection_metric': 'f1_macro',
    'verification': [
        'RF/SVM notebooks fully executed without error outputs',
        'RF/SVM original Markdown preserved exactly',
        'RF/SVM predictions, CV scores and metrics match previous committed results',
        'four models use identical test company IDs and labels',
        'eight metrics independently recomputed for all four models',
        'friend notebooks and cleaned dataset unchanged'
    ]
}
Path('results/comparison/verification.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
print(comparison[['model', 'cv_f1_macro', 'test_accuracy', 'test_f1_macro', 'test_roc_auc']].round(4).to_string(index=False))
