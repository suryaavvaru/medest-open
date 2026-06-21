from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import Counter

import numpy as np

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except Exception as e:  # pragma: no cover
    raise SystemExit('PyTorch is required for train_epidelta_neural.py') from e

from sklearn.metrics import classification_report, f1_score, accuracy_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight


KEEP_TRANSITIONS = {
    'introduced', 'treated', 'ruled_out', 'improving', 'resolved', 'no_change', 'recurrent'
}


class EpiDeltaMLP(nn.Module):
    def __init__(self, dim: int, n_transitions: int, n_states: int, dropout: float = 0.25):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.transition_head = nn.Linear(256, n_transitions)
        self.state_head = nn.Linear(256, n_states)

    def forward(self, x):
        h = self.shared(x)
        return self.transition_head(h), self.state_head(h)


def load_meta(path: Path):
    rows = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--embeddings', required=True)
    ap.add_argument('--metadata', required=True)
    ap.add_argument('--output_dir', required=True)
    ap.add_argument('--epochs', type=int, default=120)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--seed', type=int, default=13)
    ap.add_argument('--device', default='auto')
    ap.add_argument('--all_transitions', action='store_true')
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    X_all = np.load(args.embeddings).astype('float32')
    meta_all = load_meta(Path(args.metadata))

    keep_idx = []
    for i, r in enumerate(meta_all):
        tr = r.get('transition')
        if args.all_transitions or tr in KEEP_TRANSITIONS:
            keep_idx.append(i)

    X = X_all[keep_idx]
    meta = [meta_all[i] for i in keep_idx]

    transitions = np.array([r['transition'] for r in meta])
    states = np.array([r['new_state'] for r in meta])
    groups = np.array([r.get('case_id', '') for r in meta])

    le_t = LabelEncoder().fit(transitions)
    le_s = LabelEncoder().fit(states)
    y_t = le_t.transform(transitions)
    y_s = le_s.transform(states)

    split = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=args.seed)
    train_idx, test_idx = next(split.split(X, y_t, groups))

    tr_idx, va_idx = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=args.seed+1).split(X[train_idx], y_t[train_idx], groups[train_idx]))
    train2 = train_idx[tr_idx]
    val = train_idx[va_idx]

    device = args.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    def make_loader(idx, shuffle=False):
        ds = TensorDataset(
            torch.from_numpy(X[idx]),
            torch.from_numpy(y_t[idx]).long(),
            torch.from_numpy(y_s[idx]).long(),
        )
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle)

    train_loader = make_loader(train2, True)
    val_loader = make_loader(val, False)
    test_loader = make_loader(test_idx, False)

    model = EpiDeltaMLP(X.shape[1], len(le_t.classes_), len(le_s.classes_)).to(device)

    wt_t = compute_class_weight('balanced', classes=np.arange(len(le_t.classes_)), y=y_t[train2])
    wt_s = compute_class_weight('balanced', classes=np.arange(len(le_s.classes_)), y=y_s[train2])
    loss_t = nn.CrossEntropyLoss(weight=torch.tensor(wt_t, dtype=torch.float32, device=device))
    loss_s = nn.CrossEntropyLoss(weight=torch.tensor(wt_s, dtype=torch.float32, device=device))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    best = {'macro_f1': -1, 'epoch': -1}
    patience = 18
    bad = 0

    def eval_loader(loader):
        model.eval()
        gold_t=[]; pred_t=[]; gold_s=[]; pred_s=[]
        with torch.no_grad():
            for xb, ytb, ysb in loader:
                xb=xb.to(device)
                ot, os = model(xb)
                pred_t.extend(ot.argmax(1).cpu().numpy().tolist())
                pred_s.extend(os.argmax(1).cpu().numpy().tolist())
                gold_t.extend(ytb.numpy().tolist())
                gold_s.extend(ysb.numpy().tolist())
        return {
            'transition_accuracy': accuracy_score(gold_t, pred_t),
            'transition_macro_f1': f1_score(gold_t, pred_t, average='macro', zero_division=0),
            'state_accuracy': accuracy_score(gold_s, pred_s),
            'state_macro_f1': f1_score(gold_s, pred_s, average='macro', zero_division=0),
            'gold_t': gold_t, 'pred_t': pred_t, 'gold_s': gold_s, 'pred_s': pred_s,
        }

    hist=[]
    for epoch in range(1, args.epochs+1):
        model.train()
        total=0.0
        for xb, ytb, ysb in train_loader:
            xb=xb.to(device); ytb=ytb.to(device); ysb=ysb.to(device)
            opt.zero_grad()
            ot, os = model(xb)
            loss = loss_t(ot, ytb) + 0.5 * loss_s(os, ysb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.item())
        valm = eval_loader(val_loader)
        row = {'epoch': epoch, 'loss': total/max(len(train_loader),1), **{k:v for k,v in valm.items() if not k.startswith(('gold','pred'))}}
        hist.append(row)
        print('EPOCH', row)
        if valm['transition_macro_f1'] > best['macro_f1']:
            best = {'macro_f1': valm['transition_macro_f1'], 'epoch': epoch}
            torch.save(model.state_dict(), out/'best_model.pt')
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                print('[INFO] early stopping')
                break

    model.load_state_dict(torch.load(out/'best_model.pt', map_location=device))
    testm = eval_loader(test_loader)

    report_t = classification_report(testm['gold_t'], testm['pred_t'], labels=list(range(len(le_t.classes_))), target_names=le_t.classes_, zero_division=0)
    report_s = classification_report(testm['gold_s'], testm['pred_s'], labels=list(range(len(le_s.classes_))), target_names=le_s.classes_, zero_division=0)

    summary = {
        'task': 'neural_multitask',
        'mode': 'all_transition' if args.all_transitions else 'filtered_transition',
        'examples': int(len(X)),
        'device': str(device),
        'best_epoch': best['epoch'],
        'transition_accuracy': testm['transition_accuracy'],
        'transition_macro_f1': testm['transition_macro_f1'],
        'state_accuracy': testm['state_accuracy'],
        'state_macro_f1': testm['state_macro_f1'],
        'transition_classes': le_t.classes_.tolist(),
        'state_classes': le_s.classes_.tolist(),
        'transition_counts': dict(Counter(transitions)),
    }

    (out/'training_history.jsonl').write_text('\n'.join(json.dumps(r) for r in hist)+'\n', encoding='utf-8')
    (out/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    (out/'transition_report.txt').write_text(report_t, encoding='utf-8')
    (out/'state_report.txt').write_text(report_s, encoding='utf-8')
    print('=== TEST SUMMARY ===')
    print(json.dumps(summary, indent=2))
    print('=== TRANSITION REPORT ===')
    print(report_t)

if __name__ == '__main__':
    main()
