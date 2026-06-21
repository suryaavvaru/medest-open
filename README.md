# MedEST-Open: Medical Epistemic State Tracking

MedEST-Open is an independent medical NLP research project for tracking how clinical propositions change state across a case narrative.

Instead of only extracting entities or uncertainty cues, MedEST models semantic state updates:

old clinical state + new textual evidence -> transition operator -> new clinical state

The core prototype model family is called EpiDelta.

## Why this matters

Clinical narratives are full of evolving beliefs: a diagnosis may be suspected, weakened by a test, confirmed by culture, treated, resolved, or later recur.

Most clinical NLP systems identify entities, events, or uncertainty locally. MedEST focuses on the transition between states.

Example trajectory:

pulmonary embolism: not_mentioned -> suspected -> ruled_out / confirmed -> treated -> resolved

## Task formulation

Each example contains:

- proposition_text: clinical proposition, such as patient has pulmonary embolism
- old_state: previous epistemic state
- evidence_text: new sentence or segment
- transition: semantic update operator
- new_state: updated epistemic state

Example transition labels: introduced, confirmed, ruled_out, treated, improving, resolved, recurrent, no_change.

## Data source

The current open prototype uses MACCROBAT, an openly available clinical case-report annotation dataset.

No private clinical data is included in this repository.

## Current prototype results

Case-level split. PubMedBERT-based EpiDelta input embeddings. Classical model battery over transition and state labels.

| Task | Mode | Best model | Accuracy | Macro F1 | Weighted F1 |
|---|---|---|---:|---:|---:|
| New state prediction | all_state | Logistic Regression, scaled balanced | 0.6795 | 0.5562 | 0.6773 |
| Transition prediction | all_transition | Logistic Regression, balanced | 0.7072 | 0.4488 | 0.7162 |
| Transition prediction | filtered_transition | Logistic Regression, balanced | 0.7369 | 0.6641 | 0.7387 |

The strongest current result is the 7-way filtered transition task: 0.7369 accuracy, 0.6641 macro F1, 0.7387 weighted F1.

## Repository contents

- medest/data: parsing utilities
- medest/annotation: weak MedEST label generation
- medest/models: embedding utilities
- medest/training: model-training scripts
- scripts: pipeline helpers
- docs: concise research notes and result summaries
- results: small summary files only

Large raw data, generated embeddings, trained models, experiment logs, and virtual environments are intentionally excluded.

## Quick start

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

The full data pipeline expects MACCROBAT .txt and .ann files under data/raw/maccrobat/.

Generated data and experiment artifacts are ignored by git.

## Research framing

MedEST proposes a transition-centric view of medical NLP: textual evidence acts as a learned state-update operator over clinical propositions.

This framing is useful for case reports, longitudinal notes, diagnostic reasoning, treatment response tracking, and clinical timeline understanding.

## Status

This is an active research prototype. Current work focuses on improving weak-label quality, adding manual validation, scaling to larger open-access case-report corpora, and training neural EpiDelta transition models.
