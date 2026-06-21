from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import joblib


def load_metadata(path: Path):
    with path.open('r', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--embeddings', required=True)
    ap.add_argument('--metadata', required=True)
    ap.add_argument('--output_model', default='data/embeddings/local_state_classifier.joblib')
    args = ap.parse_args()
    X = np.load(args.embeddings)
    meta = load_metadata(Path(args.metadata))
    y = np.array([m['new_state'] for m in meta])
    labels, counts = np.unique(y, return_counts=True)
    print('[INFO] Label counts:', dict(zip(labels.tolist(), counts.tolist())))
    if len(labels) < 2:
        raise ValueError('Need at least two state classes to train classifier. Use more cases or toy data.')
    stratify = y if min(counts) >= 2 else None
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42, stratify=stratify)
    clf = LogisticRegression(max_iter=2000, class_weight='balanced')
    clf.fit(X_tr, y_tr)
    pred = clf.predict(X_te)
    print(classification_report(y_te, pred, zero_division=0))
    out = Path(args.output_model); out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, out)
    print(f'[OK] Saved classifier to {out}')

if __name__ == '__main__':
    main()
