#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source .venv/bin/activate 2>/dev/null || true

mkdir -p experiments/logs results docs

LATEST="$(ls -td experiments/deep_* 2>/dev/null | head -1 || true)"

if [ -z "$LATEST" ]; then
  echo "[INFO] No deep experiment found. Run scripts/run_deep_overnight.sh first."
  exit 1
fi

EMB="$LATEST/embeddings_epidelta/epidelta_input_embeddings.npy"
META="$LATEST/embeddings_epidelta/epidelta_metadata.jsonl"

if [ ! -f "$EMB" ] || [ ! -f "$META" ]; then
  echo "[FAIL] Missing EpiDelta embeddings or metadata under $LATEST"
  echo "       Expected: $EMB"
  echo "       Expected: $META"
  exit 1
fi

RUN_DIR="$LATEST/one_ten_research"
mkdir -p "$RUN_DIR"

echo "[1/4] Ablation matrix"
python -m medest.training.run_ablation_matrix \
  --metadata "$META" \
  --output_dir "$RUN_DIR/ablation_inputs" \
  > "$RUN_DIR/01_ablation_matrix.log" 2>&1 || true

echo "[2/4] Neural EpiDelta multitask"
python -m medest.training.train_epidelta_neural \
  --embeddings "$EMB" \
  --metadata "$META" \
  --output_dir "$RUN_DIR/neural_epidelta" \
  --epochs 120 \
  --batch_size 64 \
  --lr 0.0002 \
  > "$RUN_DIR/02_neural_epidelta.log" 2>&1 || true

echo "[3/4] Extract best results"
python scripts/extract_best_results.py \
  --experiment_dir "$LATEST" \
  --output_json "results/summary_from_experiment.json" || true

echo "[4/4] Repo hygiene"
python scripts/check_repo_clean.py || true

echo "[OK] 110 research run complete-ish. Inspect: $RUN_DIR"
