from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import Counter

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder


KEEP_TRANSITIONS = {
    "introduced",
    "treated",
    "ruled_out",
    "improving",
    "resolved",
    "no_change",
    "recurrent",
}


def load_meta(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", default="data/embeddings/segment_embeddings.npy")
    parser.add_argument("--metadata", default="data/embeddings/segment_metadata.jsonl")
    parser.add_argument("--output_dir", default="experiments/transition_mlp")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    X_all = np.load(args.embeddings)
    meta_all = load_meta(Path(args.metadata))

    keep_idx = [
        i for i, m in enumerate(meta_all)
        if m["transition"] in KEEP_TRANSITIONS
    ]

    X = X_all[keep_idx]
    meta = [meta_all[i] for i in keep_idx]
    y_text = np.array([m["transition"] for m in meta])
    groups = np.array([m["case_id"] for m in meta])

    print("[INFO] Total examples:", len(meta_all))
    print("[INFO] Kept examples:", len(meta))
    print("[INFO] Transition counts:", dict(Counter(y_text)))

    le = LabelEncoder()
    y = le.fit_transform(y_text)

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    clf = LogisticRegression(
        max_iter=5000,
        class_weight="balanced",
        n_jobs=-1,
        solver="lbfgs",
    )
    clf.fit(X_train, y_train)

    pred = clf.predict(X_test)

    report = classification_report(
        y_test,
        pred,
        target_names=le.classes_,
        zero_division=0,
    )
    print(report)

    cm = confusion_matrix(y_test, pred)

    (output_dir / "classification_report.txt").write_text(report)
    np.savetxt(output_dir / "confusion_matrix.csv", cm, delimiter=",", fmt="%d")

    joblib.dump(clf, output_dir / "transition_mlp.joblib")
    joblib.dump(le, output_dir / "transition_label_encoder.joblib")

    with (output_dir / "test_predictions.jsonl").open("w", encoding="utf-8") as f:
        for idx, true_id, pred_id in zip(test_idx, y_test, pred):
            row = dict(meta[idx])
            row["gold_transition"] = le.inverse_transform([true_id])[0]
            row["pred_transition"] = le.inverse_transform([pred_id])[0]
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("[OK] Saved:", output_dir)


if __name__ == "__main__":
    main()
