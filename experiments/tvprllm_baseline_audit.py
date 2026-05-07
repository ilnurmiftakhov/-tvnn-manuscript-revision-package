import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

DATA_DIR = Path(__file__).resolve().parents[1] / 'repo_tvnn' / 'TVPRLLM'
OUT_DIR = Path(__file__).resolve().parents[1] / 'experiments' / 'results'
OUT_DIR.mkdir(parents=True, exist_ok=True)

NUMBER_RE = re.compile(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?')

# Reported values from Table 4 in the manuscript PDF.
PAPER_RESULTS = {
    'Proteomics_l': {'f1_paper': 0.969, 'auprc_paper': 0.943},
    'Proteomics_gl': {'f1_paper': 0.163, 'auprc_paper': 0.223},
    'gene_l': {'f1_paper': 0.781, 'auprc_paper': 0.659},
    'gene_gl': {'f1_paper': 0.577, 'auprc_paper': 0.479},
}


def parse_dataset(stem: str) -> Tuple[np.ndarray, np.ndarray]:
    X: List[List[float]] = []
    y: List[int] = []
    path = DATA_DIR / f'{stem}.jsonl'
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            row = json.loads(line)
            nums = [float(x) for x in NUMBER_RE.findall(row['input'])]
            X.append(nums)
            y.append(int(row['output']))
    return np.asarray(X, dtype=float), np.asarray(y, dtype=int)


def chronological_split(X: np.ndarray, y: np.ndarray, train_frac: float = 0.8):
    n = len(X)
    n_train = int(n * train_frac)
    return X[:n_train], X[n_train:], y[:n_train], y[n_train:]


def get_models(seed: int = 0):
    return {
        'dummy_majority': DummyClassifier(strategy='most_frequent'),
        'logreg': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(max_iter=5000, class_weight='balanced', random_state=seed))
        ]),
        'linear_svm': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', SVC(kernel='linear', probability=True, class_weight='balanced', random_state=seed))
        ]),
        'rbf_svm': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=seed))
        ]),
        'random_forest': RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=1,
            class_weight='balanced',
            random_state=seed,
            n_jobs=-1,
        ),
        'mlp': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', alpha=1e-4, learning_rate_init=1e-3, max_iter=2000, random_state=seed))
        ]),
    }


def evaluate_model(model, X_train, y_train, X_test, y_test) -> Dict[str, float]:
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    if hasattr(model, 'predict_proba'):
        score = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, 'decision_function'):
        score = model.decision_function(X_test)
    else:
        score = pred.astype(float)
    return {
        'macro_f1': float(f1_score(y_test, pred, average='macro', zero_division=0)),
        'weighted_f1': float(f1_score(y_test, pred, average='weighted', zero_division=0)),
        'binary_f1': float(f1_score(y_test, pred, average='binary', zero_division=0)),
        'balanced_accuracy': float(balanced_accuracy_score(y_test, pred)),
        'auprc': float(average_precision_score(y_test, score)),
        'positive_rate_pred': float(np.mean(pred)),
    }


def main():
    all_results = {}
    for stem in ['Proteomics_l', 'Proteomics_gl', 'gene_l', 'gene_gl']:
        X, y = parse_dataset(stem)
        X_train, X_test, y_train, y_test = chronological_split(X, y, train_frac=0.8)
        dataset_result = {
            'n_total': int(len(X)),
            'n_train': int(len(X_train)),
            'n_test': int(len(X_test)),
            'feature_dim': int(X.shape[1]),
            'train_positive_rate': float(np.mean(y_train)),
            'test_positive_rate': float(np.mean(y_test)),
            'paper': PAPER_RESULTS[stem],
            'models': {},
        }
        for name, model in get_models(seed=0).items():
            dataset_result['models'][name] = evaluate_model(model, X_train, y_train, X_test, y_test)
        all_results[stem] = dataset_result
    out_path = OUT_DIR / 'tvprllm_baseline_audit.json'
    out_path.write_text(json.dumps(all_results, indent=2), encoding='utf-8')
    print(json.dumps(all_results, indent=2))
    print(f'SAVED {out_path}')


if __name__ == '__main__':
    main()
