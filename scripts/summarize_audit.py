#!/usr/bin/env python3
from __future__ import annotations

import argparse, csv
from collections import Counter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--audit_tsv', required=True)
    args = ap.parse_args()

    c = Counter()
    total = 0
    with open(args.audit_tsv, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f, delimiter='\t'):
            lab = (row.get('audit_label') or '').strip() or 'unlabeled'
            c[lab] += 1
            total += 1
    print('total', total)
    for k,v in c.most_common():
        print(f'{k}\t{v}\t{v/max(total,1):.3f}')

if __name__ == '__main__':
    main()
