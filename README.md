# MedEST-Open

**A weakly supervised benchmark for evidence-conditioned medical epistemic state tracking.**

MedEST-Open is an independent medical NLP research project by **Surya Teja Avvaru** and **Krishna Sai Pokala**. It studies how clinical propositions change state as new evidence appears in a medical case narrative.

Instead of only extracting static entities, events, assertion labels, or uncertainty cues, MedEST models a structured semantic update:

```text
old clinical state + proposition + evidence sentence -> transition operator -> new clinical state
```

Example trajectories:

```text
pulmonary embolism: suspected -> ruled_out
pneumonia: introduced -> treated -> improving -> resolved
rash: resolved -> recurrent
```

## Final headline result

The final high-end model is a fine-tuned PubMedBERT sequence classifier on the filtered seven-way MedEST transition task.

| Model | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| Fine-tuned PubMedBERT | **0.9061** | **0.9075** | **0.9062** |
| PubMedBERT EpiDelta embeddings + balanced logistic regression | 0.7369 | 0.6641 | 0.7387 |
| Neural multitask EpiDelta head | 0.6743 | 0.5796 | 0.6700 |
| New-state logistic regression | 0.6795 | 0.5562 | 0.6773 |
| All-transition logistic regression | 0.7072 | 0.4488 | 0.7162 |

**Paper headline:** Fine-tuned PubMedBERT achieves **0.908 macro F1** on seven-way Medical Epistemic State Transition prediction in the MedEST-Open weak-label benchmark.

## Task

Each example contains:

- `old_state`: previous epistemic state
- `proposition_text`: normalized medical proposition
- `evidence_text`: clinical sentence or segment
- `transition`: semantic update operator
- `new_state`: updated proposition state

Current filtered transition labels:

```text
introduced, treated, ruled_out, improving, resolved, recurrent, no_change
```

## Data

The current prototype uses **MACCROBAT** case-report text and annotations to build weak MedEST transition examples.

| Statistic | Value |
|---|---:|
| MACCROBAT cases | 200 |
| Segments | 5,480 |
| Mentions | 25,189 |
| Weak MedEST examples | 4,296 |
| Filtered seven-way transition examples | 4,207 |
| Held-out evaluation examples | 1,022 |

No private clinical data is included in this repository.

## Weak supervision

MedEST-Open is intentionally framed as a **weakly supervised benchmark prototype**, not an expert-gold clinical benchmark.

Weak labels are generated using transparent medical cue rules and consistency filters. Automatic heuristic audits estimate label quality and noise modes.

| Audit sample | Good | Ambiguous | Bad transition | Bad proposition |
|---|---:|---:|---:|---:|
| 100 examples | 84.0% | 4.0% | 4.0% | 8.0% |
| 500 examples | 76.0% | 10.6% | 6.8% | 6.6% |

The correct interpretation is:

> MedEST-Open demonstrates that evidence-conditioned medical transition labels are highly learnable under a transparent weak-label protocol. Expert-reviewed proposition trajectories are the next benchmark milestone.



## Repository contents

```text
medest/
  annotation/        weak MedEST label generation
  data/              parsing utilities
  models/            embedding utilities
  training/          model-training scripts

scripts/             pipeline, audit, reporting, and repo utilities
docs/                final results, audit notes, methodology, decisions
paper/               manuscript scaffold
results/             final result summaries
examples/            small synthetic examples only
preprint/            polished preprint PDF and source package
```

Large raw data, embeddings, trained models, logs, downloads, virtual environments, and checkpoints are intentionally excluded.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The full data pipeline expects MACCROBAT `.txt` and `.ann` files under:

```text
data/raw/maccrobat/
```

Generated data and experiment artifacts are ignored by git.

## Reproduce the prototype pipeline

```bash
bash scripts/run_pipeline.sh
```

For the extended research battery:

```bash
bash scripts/run_110_research.sh
```

For the final PubMedBERT fine-tuning result, see:

```text
docs/FINAL_GPU_RESULT.md
results/pubmedbert_gpu_final_summary.json
```

## Research framing

MedEST proposes a transition-centric view of medical NLP:

> textual evidence acts as a learned state-update operator over clinical propositions.

This framing is useful for clinical case reports, longitudinal notes, diagnostic reasoning traces, treatment response tracking, and timeline-aware clinical narrative understanding.

## Current limitations

- Labels are weakly supervised and heuristically audited, not expert-gold.
- The current benchmark uses case reports, not hospital EHR notes.
- The current best model predicts local transition labels, not full end-to-end expert-reviewed proposition trajectories.
- The system is a research prototype only and is not clinical decision support.

## Authors

- Surya Teja Avvaru - Independent Researcher - suryata1@umbc.edu
- Krishna Sai Pokala - Independent Researcher - ok50250@umbc.edu
