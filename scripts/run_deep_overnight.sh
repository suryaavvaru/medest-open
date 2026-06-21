#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

RUN_ID="deep_$(date +%Y%m%d_%H%M%S)"
EXP_DIR="experiments/$RUN_ID"
LOG_DIR="$EXP_DIR/logs"
mkdir -p "$EXP_DIR" "$LOG_DIR"

echo "=================================================="
echo "MedEST Deep Overnight Run"
echo "RUN_ID: $RUN_ID"
echo "EXP_DIR: $EXP_DIR"
echo "START: $(date)"
echo "=================================================="

echo "[1/10] Preflight"
python - <<'PY'
import sys, torch, platform
print("Python:", sys.version)
print("Platform:", platform.platform())
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY

echo "[2/10] Data counts"
echo "TXT:" $(find data/raw/maccrobat -name "*.txt" | wc -l)
echo "ANN:" $(find data/raw/maccrobat -name "*.ann" | wc -l)

test "$(find data/raw/maccrobat -name '*.txt' | wc -l)" -gt 0
test "$(find data/raw/maccrobat -name '*.ann' | wc -l)" -gt 0

echo "[3/10] Rebuild processed data"
rm -rf "$EXP_DIR/data"
mkdir -p "$EXP_DIR/data"

python -m medest.data.parse_maccrobat \
  --input_dir data/raw/maccrobat \
  --output_jsonl "$EXP_DIR/data/maccrobat_cases.jsonl" \
  | tee "$LOG_DIR/01_parse.log"

python -m medest.annotation.weak_labeler \
  --input_jsonl "$EXP_DIR/data/maccrobat_cases.jsonl" \
  --output_jsonl "$EXP_DIR/data/maccrobat_medest_weak.jsonl" \
  | tee "$LOG_DIR/02_weak_label.log"

echo "[4/10] Create enhanced EpiDelta embedding script if missing"
cat > medest/models/embed_epidelta_inputs.py <<'PY'
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel


def load_examples(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name", default="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
    parser.add_argument("--max_length", type=int, default=384)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = load_examples(args.input_jsonl)

    texts = []
    for r in rows:
        text = (
            f"Task: Medical epistemic state transition prediction.\n"
            f"Old state: {r['old_state']}\n"
            f"Medical proposition: {r['proposition_text']}\n"
            f"Evidence sentence: {r['evidence_text']}\n"
            f"Predict the semantic transition and new epistemic state."
        )
        texts.append(text)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name).to(args.device)
    model.eval()

    vecs = []
    for i in tqdm(range(0, len(texts), args.batch_size), desc="EpiDelta embedding"):
        batch = texts[i:i+args.batch_size]
        toks = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt",
        )
        toks = {k: v.to(args.device) for k, v in toks.items()}
        out = model(**toks)
        cls = out.last_hidden_state[:, 0, :].detach().cpu().float().numpy()
        vecs.append(cls)

    X = np.concatenate(vecs, axis=0)
    np.save(outdir / "epidelta_input_embeddings.npy", X)

    with (outdir / "epidelta_metadata.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("[OK] Saved", outdir / "epidelta_input_embeddings.npy", X.shape)
    print("[OK] Saved", outdir / "epidelta_metadata.jsonl")


if __name__ == "__main__":
    main()
PY

echo "[5/10] Create deep experiment trainer"
cat > medest/training/train_deep_battery.py <<'PY'
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

        report = classification_report(
            y_test,
            pred,
            target_names=le.classes_,
            zero_division=0,
        )

        model_dir = outdir / name
        model_dir.mkdir(parents=True, exist_ok=True)

        (model_dir / "classification_report.txt").write_text(report, encoding="utf-8")
        cm = confusion_matrix(y_test, pred)
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
PY

echo "[6/10] Generate EpiDelta embeddings"
mkdir -p "$EXP_DIR/embeddings_epidelta"

python -m medest.models.embed_epidelta_inputs \
  --input_jsonl "$EXP_DIR/data/maccrobat_medest_weak.jsonl" \
  --output_dir "$EXP_DIR/embeddings_epidelta" \
  --model_name microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext \
  --max_length 384 \
  --batch_size 2 \
  --device cpu \
  2>&1 | tee "$LOG_DIR/03_epidelta_embeddings.log"

echo "[7/10] Deep battery on EpiDelta embeddings"
python -m medest.training.train_deep_battery \
  --embeddings "$EXP_DIR/embeddings_epidelta/epidelta_input_embeddings.npy" \
  --metadata "$EXP_DIR/embeddings_epidelta/epidelta_metadata.jsonl" \
  --output_dir "$EXP_DIR/deep_battery_epidelta" \
  2>&1 | tee "$LOG_DIR/04_deep_battery_epidelta.log"

echo "[8/10] Deep battery on original segment embeddings if available"
if [ -f "data/embeddings/segment_embeddings.npy" ] && [ -f "data/embeddings/segment_metadata.jsonl" ]; then
  python -m medest.training.train_deep_battery \
    --embeddings data/embeddings/segment_embeddings.npy \
    --metadata data/embeddings/segment_metadata.jsonl \
    --output_dir "$EXP_DIR/deep_battery_original" \
    2>&1 | tee "$LOG_DIR/05_deep_battery_original.log"
else
  echo "Original embeddings not found; skipping original battery." | tee "$LOG_DIR/05_deep_battery_original.log"
fi

echo "[9/10] Generate final report"
python - <<PY
import json
from pathlib import Path

exp = Path("$EXP_DIR")
report = []

report.append("# MedEST Deep Overnight Report")
report.append("")
report.append(f"Run: {exp.name}")
report.append("")

for p in sorted(exp.glob("deep_battery_*/BEST_SUMMARY.json")):
    report.append(f"## {p.parent.name}")
    rows = json.loads(p.read_text())
    for r in rows:
        report.append("")
        report.append(f"### Task: {r['task']} | Mode: {r['mode']}")
        report.append(f"- Best model: {r['model']}")
        report.append(f"- Examples: {r['examples']}")
        report.append(f"- Accuracy: {r['accuracy']:.4f}")
        report.append(f"- Macro F1: {r['macro_f1']:.4f}")
        report.append(f"- Weighted F1: {r['weighted_f1']:.4f}")
        report.append(f"- Classes: {', '.join(r['classes'])}")
    report.append("")

out = exp / "FINAL_REPORT.md"
out.write_text("\\n".join(report), encoding="utf-8")
print(out.read_text())
PY

echo "[10/10] Done"
echo "END: $(date)"
echo "Final report:"
echo "$EXP_DIR/FINAL_REPORT.md"
