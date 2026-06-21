from __future__ import annotations

import argparse, json
from pathlib import Path
from collections import Counter


def load_rows(path: Path):
    with path.open('r', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def make_text(r, mode):
    old = r.get('old_state','not_mentioned')
    prop = r.get('proposition_text','')
    ev = r.get('evidence_text','')
    if mode == 'evidence_only':
        return ev
    if mode == 'proposition_evidence':
        return f'Medical proposition: {prop}\nEvidence sentence: {ev}'
    if mode == 'oldstate_evidence':
        return f'Old state: {old}\nEvidence sentence: {ev}'
    if mode == 'epidelta_full':
        return f'Task: Medical epistemic state transition prediction.\nOld state: {old}\nMedical proposition: {prop}\nEvidence sentence: {ev}\nPredict the semantic transition and new epistemic state.'
    raise ValueError(mode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--metadata', required=True)
    ap.add_argument('--output_dir', required=True)
    args = ap.parse_args()

    rows = load_rows(Path(args.metadata))
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    modes = ['evidence_only', 'proposition_evidence', 'oldstate_evidence', 'epidelta_full']
    for mode in modes:
        p = out / f'{mode}.jsonl'
        with p.open('w', encoding='utf-8') as f:
            for r in rows:
                nr = dict(r)
                nr['ablation_mode'] = mode
                nr['model_input_text'] = make_text(r, mode)
                f.write(json.dumps(nr, ensure_ascii=False) + '\n')
        print('[OK]', mode, p)

    counts = Counter(r.get('transition') for r in rows)
    (out/'label_counts.json').write_text(json.dumps(counts, indent=2), encoding='utf-8')

if __name__ == '__main__':
    main()
