# MedEST-Open: Medical Epistemic State Tracking

MedEST-Open is an independent medical NLP research project for tracking how clinical propositions change state across a medical case narrative.

Instead of only extracting entities, events, or uncertainty cues, MedEST models **semantic state updates**:

> old clinical state + proposition + new textual evidence -> transition operator -> new clinical state

The core prototype model family is called **EpiDelta**.

## Why this matters

Clinical narratives are full of evolving beliefs. A diagnosis can be suspected, weakened by a test, confirmed by culture, treated, resolved, or later recur. Standard clinical NLP systems often identify entities or local assertion status, but they do not explicitly model the **change operation** connecting one state to the next.

Example trajectory:

```text
pulmonary embolism: not_mentioned -> suspected -> ruled_out / confirmed -> treated -> resolved
```

MedEST treats each sentence or segment as possible evidence that updates a proposition-level clinical state.

## Task formulation

Each example contains:

- `proposition_text`: clinical proposition, such as `patient has pulmonary embolism`
- `old_state`: previous epistemic state
- `evidence_text`: new sentence or segment
- `transition`: semantic update operator
- `new_state`: updated epistemic state

Example transition labels:

```text
introduced, confirmed, ruled_out, treated, improving, resolved, recurrent, no_change
```

## Data source

The current open prototype uses MACCROBAT, an openly available clinical case-report annotation dataset. The pipeline converts annotated case reports into weak MedEST examples.

No private clinical data is included in this repository.

## Current prototype results

Case-level split. PubMedBERT-based EpiDelta input embeddings. Classical model battery over transition and state labels.

| Task | Mode | Best model | Accuracy | Macro F1 | Weighted F1 |
|---|---|---|---:|---:|---:|
| New state prediction | all_state | Logistic Regression, scaled balanced | 0.6795 | 0.5562 | 0.6773 |
| Transition prediction | all_transition | Logistic Regression, balanced | 0.7072 | 0.4488 | 0.7162 |
| Transition prediction | filtered_transition | Logistic Regression, balanced | 0.7369 | 0.6641 | 0.7387 |

The strongest current result is the 7-way filtered transition task:

> **0.7369 accuracy, 0.6641 macro F1, 0.7387 weighted F1**

## Repository contents

```text
medest/
  annotation/        weak MedEST label generation
  data/              parsing utilities
  models/            embedding utilities
  training/          model-training scripts

scripts/             pipeline, audit, reporting, and repo utilities
docs/                method, results, roadmap, and error analysis notes
paper/               short-paper scaffold
examples/            small synthetic examples only
results/             small result summaries only
```

Large raw data, embeddings, trained models, logs, downloads, and virtual environments are intentionally excluded.

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

## Reproduce the current prototype pipeline

```bash
bash scripts/run_pipeline.sh
```

For the deep EpiDelta battery, use the research repo with local experiment artifacts and run:

```bash
bash scripts/run_110_research.sh
```

## Research framing

MedEST proposes a transition-centric view of medical NLP:

> textual evidence acts as a learned state-update operator over clinical propositions.

This framing is useful for case reports, longitudinal notes, diagnostic reasoning, treatment response tracking, and clinical timeline understanding.

## Current limitations

- Current labels are weak labels derived from case-report annotations and lexical cues.
- Manual validation is needed before strong scientific claims.
- MACCROBAT is small, so scaling to larger open-access case-report corpora is a key next step.
- Current best public result uses frozen PubMedBERT embeddings with classical classifiers; neural EpiDelta models are included as the next modeling step.

## Status

Active research prototype. Current work focuses on manual validation, ablation studies, neural EpiDelta training, and scale-up to larger open case-report corpora.

## Authors

- Surya Teja Avvaru — Independent Researcher — suryata1@umbc.edu
- Krishna Sai Pokala — Independent Researcher — ok50250@umbc.edu
