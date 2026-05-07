import json
import re
from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

DATA_DIR = Path(__file__).resolve().parents[1] / 'repo_tvnn' / 'TVPRLLM'
OUT_DIR = Path(__file__).resolve().parents[1] / 'experiments' / 'results'
OUT_DIR.mkdir(parents=True, exist_ok=True)
NUMBER_RE = re.compile(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?')


def parse_dataset(stem: str):
    X=[]; y=[]
    with open(DATA_DIR / f'{stem}.jsonl','r',encoding='utf-8') as f:
        for line in f:
            row=json.loads(line)
            X.append([float(x) for x in NUMBER_RE.findall(row['input'])])
            y.append(int(row['output']))
    return np.asarray(X,float), np.asarray(y,int)


def models(seed=0):
    return {
        'logreg': Pipeline([('scaler',StandardScaler()),('clf',LogisticRegression(max_iter=5000,class_weight='balanced',random_state=seed))]),
        'linear_svm': Pipeline([('scaler',StandardScaler()),('clf',SVC(kernel='linear',probability=True,class_weight='balanced',random_state=seed))]),
        'random_forest': RandomForestClassifier(n_estimators=400,class_weight='balanced',random_state=seed,n_jobs=-1),
    }


def score_model(model, X, y, cv):
    rows=[]
    for tr, te in cv.split(X,y):
        model.fit(X[tr], y[tr])
        pred=model.predict(X[te])
        if hasattr(model,'predict_proba'):
            score=model.predict_proba(X[te])[:,1]
        else:
            score=model.decision_function(X[te])
        rows.append({
            'macro_f1': float(f1_score(y[te], pred, average='macro', zero_division=0)),
            'weighted_f1': float(f1_score(y[te], pred, average='weighted', zero_division=0)),
            'binary_f1': float(f1_score(y[te], pred, average='binary', zero_division=0)),
            'balanced_accuracy': float(balanced_accuracy_score(y[te], pred)),
            'auprc': float(average_precision_score(y[te], score)),
        })
    return rows


def summarize(rows: List[Dict[str,float]]):
    out={}
    for k in rows[0]:
        vals=[r[k] for r in rows]
        out[k]={'mean':float(np.mean(vals)),'std':float(np.std(vals)),'min':float(np.min(vals)),'max':float(np.max(vals))}
    return out


def main():
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    payload={}
    for stem in ['Proteomics_l','Proteomics_gl','gene_l','gene_gl']:
        X,y=parse_dataset(stem)
        payload[stem]={'n_total':int(len(X)),'feature_dim':int(X.shape[1]),'positive_rate':float(np.mean(y)),'models':{}}
        for name,model in models().items():
            rows=score_model(model,X,y,cv)
            payload[stem]['models'][name]={'folds':rows,'summary':summarize(rows)}
    out=OUT_DIR/'tvprllm_baseline_cv.json'
    out.write_text(json.dumps(payload,indent=2),encoding='utf-8')
    print(json.dumps(payload,indent=2))
    print('SAVED',out)

if __name__=='__main__':
    main()
