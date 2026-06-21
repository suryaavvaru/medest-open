from __future__ import annotations
import argparse, json, re
from pathlib import Path
from typing import Dict, Any, Optional

STATE_PATTERNS = [
    ('ruled_out', [r'\bruled out\b', r'\bexcluded\b', r'\bnegative for\b', r'\bno evidence of\b', r'\bwithout evidence of\b', r'\bnot consistent with\b']),
    ('confirmed', [r'\bconfirmed\b', r'\bdiagnosed with\b', r'\bdiagnosis of\b', r'\bculture grew\b', r'\bbiopsy (showed|revealed|confirmed)\b', r'\bpathology (showed|revealed|confirmed)\b']),
    ('supported', [r'\bconsistent with\b', r'\bsuggestive of\b', r'\bsupport(ed|s)?\b', r'\bcompatible with\b', r'\bconcerning for\b']),
    ('suspected', [r'\bsuspected\b', r'\bconcern for\b', r'\bconsidered\b', r'\bdifferential diagnosis\b', r'\bworking diagnosis\b']),
    ('possible', [r'\bpossible\b', r'\bmay represent\b', r'\bmight represent\b', r'\bcould be\b', r'\bcannot rule out\b', r'\bcannot exclude\b']),
    ('treated', [r'\btreated with\b', r'\bstarted on\b', r'\binitiated\b', r'\breceived\b', r'\badministered\b']),
    ('resolved', [r'\bresolved\b', r'\bresolution of\b', r'\bcomplete recovery\b', r'\bdischarged.*(well|stable)\b']),
    ('improving', [r'\bimproved\b', r'\bimprovement\b', r'\bresponded to\b', r'\bclinical response\b']),
    ('recurrent', [r'\brecurred\b', r'\brecurrence\b', r'\brelapsed\b', r'\brelapse\b']),
    ('weakened', [r'\bless likely\b', r'\bunlikely\b', r'\blow suspicion\b', r'\bnot favored\b']),
]


def infer_state(text: str) -> Optional[str]:
    lowered = text.lower()
    for state, patterns in STATE_PATTERNS:
        if any(re.search(p, lowered) for p in patterns):
            return state
    return None


def infer_transition(old_state: str, new_state: str) -> str:
    if old_state in {None, 'not_mentioned'}:
        if new_state in {'possible', 'suspected', 'supported', 'confirmed'}:
            return 'introduced'
        return new_state
    if new_state == old_state:
        return 'no_change'
    return {
        'supported': 'strengthened', 'confirmed': 'confirmed', 'weakened': 'weakened',
        'ruled_out': 'ruled_out', 'treated': 'treated', 'improving': 'improved',
        'resolved': 'resolved', 'recurrent': 'recurred', 'possible': 'weakened', 'suspected': 'introduced'
    }.get(new_state, 'no_change')


def build_weak_examples(case: Dict[str, Any]):
    mention_by_id = {m['mention_id']: m for m in case.get('mentions', [])}
    prop_state: Dict[str, str] = {}
    examples = []
    for seg in case.get('segments', []):
        new_state = infer_state(seg['text'])
        if new_state is None:
            continue
        mention_ids = seg.get('mention_ids', [])
        if not mention_ids:
            mention_ids = [None]
        for mid in mention_ids:
            m = mention_by_id.get(mid) if mid else None
            mention_text = m['text'].strip() if m else 'unknown medical proposition'
            mention_label = m['label'] if m else None
            proposition_id = f'patient_has::{mention_text.lower()}'
            old_state = prop_state.get(proposition_id, 'not_mentioned')
            transition = infer_transition(old_state, new_state)
            prop_state[proposition_id] = new_state
            examples.append({
                'case_id': case['case_id'],
                'segment_id': seg['segment_id'],
                'segment_order': seg['order'],
                'proposition_id': proposition_id,
                'proposition_text': f'patient has {mention_text}',
                'mention_text': mention_text,
                'mention_label': mention_label,
                'old_state': old_state,
                'new_state': new_state,
                'transition': transition,
                'evidence_text': seg['text'],
                'confidence': 'weak',
            })
    return examples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input_jsonl', required=True)
    ap.add_argument('--output_jsonl', required=True)
    args = ap.parse_args()
    out = Path(args.output_jsonl); out.parent.mkdir(parents=True, exist_ok=True)
    n_cases = n_examples = 0
    with Path(args.input_jsonl).open('r', encoding='utf-8') as fin, out.open('w', encoding='utf-8') as fout:
        for line in fin:
            case = json.loads(line); n_cases += 1
            for ex in build_weak_examples(case):
                fout.write(json.dumps(ex, ensure_ascii=False) + '\n'); n_examples += 1
    print(f'[OK] Read {n_cases} cases')
    print(f'[OK] Wrote {n_examples} weak examples to {out}')

if __name__ == '__main__':
    main()
