#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, glob
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--experiment_dir', required=True)
    ap.add_argument('--output_json', default='results/summary_from_experiment.json')
    args = ap.parse_args()

    paths = glob.glob(str(Path(args.experiment_dir) / '**' / 'best_result.json'), recursive=True)
    results = []
    for p in paths:
        try:
            results.append(json.loads(Path(p).read_text(encoding='utf-8')))
        except Exception as e:
            print('[WARN]', p, e)

    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({'best_results': results}, indent=2, default=str), encoding='utf-8')
    print('[OK] wrote', out)
    for r in results:
        print(r.get('task'), r.get('mode'), r.get('model'), 'macro_f1=', r.get('macro_f1'))

if __name__ == '__main__':
    main()
