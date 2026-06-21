#!/usr/bin/env python3
from pathlib import Path

bad_parts = {'.venv', 'venv', 'data', 'downloads', 'experiments', '__pycache__'}
bad_suffixes = {'.npy', '.npz', '.joblib', '.pkl', '.pt', '.pth', '.bin', '.safetensors', '.log'}

root = Path('.')
problems = []
for p in root.rglob('*'):
    if '.git' in p.parts:
        continue
    if any(part in bad_parts for part in p.parts):
        problems.append(str(p))
    if p.is_file() and p.suffix in bad_suffixes:
        problems.append(str(p))

print('=== MedEST repo hygiene check ===')
if not problems:
    print('[OK] No obvious raw data/model/log artifacts found in working tree.')
else:
    print('[WARN] Potential artifacts found:')
    for x in problems[:100]:
        print(' ', x)
    if len(problems) > 100:
        print(' ...', len(problems)-100, 'more')
