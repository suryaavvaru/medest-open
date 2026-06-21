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
