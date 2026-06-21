#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, random
from pathlib import Path

FIELDS = [
    'audit_label',
    'audit_notes',
    'case_id',
    'segment_id',
    'old_state',
    'transition',
    'new_state',
    'proposition_text',
    'evidence_text',
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input_jsonl', required=True)
    ap.add_argument('--output_tsv', required=True)
    ap.add_argument('--n', type=int, default=100)
    ap.add_argument('--seed', type=int, default=13)
    args = ap.parse_args()

    rows = [json.loads(line) for line in Path(args.input_jsonl).read_text(encoding='utf-8').splitlines() if line.strip()]
    random.seed(args.seed)
    sample = random.sample(rows, min(args.n, len(rows)))

    out = Path(args.output_tsv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', encoding='utf-8') as f:
        f.write('\t'.join(FIELDS) + '\n')
        for r in sample:
            vals = [
                '',
                '',
                r.get('case_id',''),
                r.get('segment_id',''),
                r.get('old_state',''),
                r.get('transition',''),
                r.get('new_state',''),
                r.get('proposition_text','').replace('\t',' ').replace('\n',' '),
                r.get('evidence_text','').replace('\t',' ').replace('\n',' '),
            ]
            f.write('\t'.join(vals) + '\n')
    print('[OK] wrote audit sheet:', out)
    print('Use audit_label values such as: good, bad_proposition, bad_transition, ambiguous, needs_context')

if __name__ == '__main__':
    main()
