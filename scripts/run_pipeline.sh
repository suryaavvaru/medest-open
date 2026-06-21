#!/usr/bin/env bash
set -e

python -m medest.data.parse_maccrobat \
  --input_dir data/raw/maccrobat \
  --output_jsonl data/processed/maccrobat_cases.jsonl

python -m medest.annotation.weak_labeler \
  --input_jsonl data/processed/maccrobat_cases.jsonl \
  --output_jsonl data/processed/maccrobat_medest_weak.jsonl

python -m medest.models.embed_segments \
  --input_jsonl data/processed/maccrobat_medest_weak.jsonl \
  --output_dir data/embeddings \
  --model_name microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext \
  --max_length 256 \
  --batch_size 4

python -m medest.training.train_local_state_classifier \
  --embeddings data/embeddings/segment_embeddings.npy \
  --metadata data/embeddings/segment_metadata.jsonl \
  --output_model data/embeddings/local_state_classifier.joblib
