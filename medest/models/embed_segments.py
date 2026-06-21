from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel


def load_examples(path: Path) -> List[Dict[str, Any]]:
    with path.open('r', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]

@torch.no_grad()
def embed_texts(texts, model_name, max_length, batch_size, device):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    vecs = []
    for i in tqdm(range(0, len(texts), batch_size), desc='Embedding'):
        toks = tokenizer(texts[i:i+batch_size], padding=True, truncation=True, max_length=max_length, return_tensors='pt')
        toks = {k: v.to(device) for k, v in toks.items()}
        with torch.cuda.amp.autocast(enabled=device.startswith('cuda')):
            out = model(**toks)
            vec = out.last_hidden_state[:, 0, :]
        vecs.append(vec.detach().cpu().float().numpy())
    return np.concatenate(vecs, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input_jsonl', required=True)
    ap.add_argument('--output_dir', required=True)
    ap.add_argument('--model_name', default='microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext')
    ap.add_argument('--max_length', type=int, default=256)
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--device', default=None)
    args = ap.parse_args()
    examples = load_examples(Path(args.input_jsonl))
    if not examples:
        raise ValueError('No examples found. Weak labeler produced zero rows.')
    texts = [f"Proposition: {ex['proposition_text']}\nEvidence: {ex['evidence_text']}" for ex in examples]
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[INFO] Device: {device}')
    print(f'[INFO] Model: {args.model_name}')
    print(f'[INFO] Examples: {len(texts)}')
    emb = embed_texts(texts, args.model_name, args.max_length, args.batch_size, device)
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir/'segment_embeddings.npy', emb)
    with (out_dir/'segment_metadata.jsonl').open('w', encoding='utf-8') as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + '\n')
    print(f'[OK] Saved {out_dir / "segment_embeddings.npy"} shape={emb.shape}')
    print(f'[OK] Saved {out_dir / "segment_metadata.jsonl"}')

if __name__ == '__main__':
    main()
