# Method Summary

## Core idea

MedEST-Open frames clinical narrative understanding as proposition-level epistemic state tracking.

For each proposition, the model observes an old state and a new evidence segment, then predicts the transition operator and updated state.

```text
old_state + proposition_text + evidence_text -> transition -> new_state
```

## EpiDelta input format

The EpiDelta prototype creates natural-language model inputs of the form:

```text
Task: Medical epistemic state transition prediction.
Old state: suspected.
Medical proposition: patient has pulmonary embolism.
Evidence sentence: CT angiography showed no evidence of pulmonary embolism.
Predict the semantic transition and new epistemic state.
```

These inputs are embedded using PubMedBERT and used for downstream transition/state prediction.

## Data generation

The current seed dataset is MACCROBAT. The pipeline:

1. parses case-report text and brat-style annotation files;
2. creates sentence/segment-level clinical propositions;
3. assigns weak epistemic states and transitions using lexical and annotation-derived heuristics;
4. builds model-ready JSONL examples;
5. embeds each EpiDelta input with PubMedBERT;
6. evaluates models using case-level train/test splits.

## Evaluation split

The split is grouped by case ID. Examples from the same case report do not appear in both train and test sets.

## Models

Current public results use a classical model battery over frozen embeddings:

- dummy most-frequent baseline
- balanced logistic regression
- scaled balanced logistic regression
- SGD logistic classifier
- ExtraTrees
- RandomForest

The pack also includes a neural multitask EpiDelta trainer with shared layers and two heads:

- transition head
- new-state head

## Why transition prediction is the crux

Local state prediction can be inflated by direct lexical cues. Transition prediction is harder because it asks whether new evidence introduces, confirms, weakens, rules out, treats, resolves, or changes the state of a proposition.
