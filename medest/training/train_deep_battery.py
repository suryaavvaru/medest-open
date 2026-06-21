from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import Counter, defaultdict

import joblib
import numpy as np

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import make_pipeline


FILTERED_TRANSITIONS = {
    "introduced",
    "treated",
    "ruled_out",
    "improving",
    "resolved",
    "no_change",
    "recurrent",
}


def load_meta(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def run_task(X_all, meta_all, label_key, outdir, mode):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if mode == "filtered_transition":
        keep = [i for i, m in enumerate(meta_all) if m[label_key] in FILTERED_TRANSITIONS]
    else:
        keep = list(range(len(meta_all)))

    X = X_all[keep]
    meta = [meta_all[i] for i in keep]
    y_text = np.array([m[label_key] for m in meta])
    groups = np.array([m["case_id"] for m in meta])

    counts = Counter(y_text)
    print(f"\n=== TASK {label_key} | MODE {mode} ===")
    print("Examples:", len(meta))
    print("Counts:", dict(counts))

    le = LabelEncoder()
    y = le.fit_transform(y_text)

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    models = {
        "dummy_most_frequent": DummyClassifier(strategy="most_frequent"),
        "logreg_balanced": LogisticRegression(max_iter=5000, class_weight="balanced", n_jobs=-1),
        "logreg_scaled_balanced": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=5000, class_weight="balanced", n_jobs=-1)
        ),
        "sgd_logloss_balanced": make_pipeline(
            StandardScaler(),
            SGDClassifier(loss="log_loss", class_weight="balanced", max_iter=3000, random_state=42)
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=600,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
            min_samples_leaf=2
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=400,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
            min_samples_leaf=2
        ),
    }

    results = []
    best = None

    for name, model in models.items():
        print(f"\n--- Training {name} ---")
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        macro = f1_score(y_test, pred, average="macro", zero_division=0)
        weighted = f1_score(y_test, pred, average="weighted", zero_division=0)
        acc = accuracy_score(y_test, pred)

        all_label_ids = list(range(len(le.classes_)))

        report = classification_report(
            y_test,
            pred,
            labels=all_label_ids,
            target_names=le.classes_,
            zero_division=0,
        )

        model_dir = outdir / name
        model_dir.mkdir(parents=True, exist_ok=True)

        (model_dir / "classification_report.txt").write_text(report, encoding="utf-8")
        cm = confusion_matrix(y_test, pred, labels=all_label_ids)
        np.savetxt(model_dir / "confusion_matrix.csv", cm, delimiter=",", fmt="%d")

        joblib.dump(model, model_dir / "model.joblib")
        joblib.dump(le, model_dir / "label_encoder.joblib")

        with (model_dir / "test_predictions.jsonl").open("w", encoding="utf-8") as f:
            for idx, true_id, pred_id in zip(test_idx, y_test, pred):
                row = dict(meta[idx])
                row["gold_label"] = le.inverse_transform([true_id])[0]
                row["pred_label"] = le.inverse_transform([pred_id])[0]
                row["correct"] = bool(true_id == pred_id)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        summary = {
            "task": label_key,
            "mode": mode,
            "model": name,
            "examples": len(meta),
            "accuracy": float(acc),
            "macro_f1": float(macro),
            "weighted_f1": float(weighted),
            "label_counts": dict(counts),
            "classes": list(le.classes_),
        }
        save_json(model_dir / "summary.json", summary)
        results.append(summary)

        print(report)
        print("SUMMARY:", summary)

        if best is None or macro > best["macro_f1"]:
            best = summary

    results_sorted = sorted(results, key=lambda x: x["macro_f1"], reverse=True)
    save_json(outdir / "all_results.json", results_sorted)
    save_json(outdir / "best_result.json", results_sorted[0])

    return results_sorted[0], results_sorted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    X = np.load(args.embeddings)
    meta = load_meta(args.metadata)

    print("Embeddings:", X.shape)
    print("Metadata:", len(meta))

    all_best = []

    for label_key, mode in [
        ("new_state", "all_state"),
        ("transition", "all_transition"),
        ("transition", "filtered_transition"),
    ]:
        best, _ = run_task(
            X_all=X,
            meta_all=meta,
            label_key=label_key,
            outdir=outdir / f"{label_key}_{mode}",
            mode=mode,
        )
        all_best.append(best)

    save_json(outdir / "BEST_SUMMARY.json", all_best)

    print("\n================ FINAL BEST SUMMARY ================")
    for row in all_best:
        print(row)


if __name__ == "__main__":
    main()
