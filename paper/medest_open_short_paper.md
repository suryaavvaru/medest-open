# MedEST-Open: Medical Epistemic State Tracking with Evidence-Conditioned Semantic Delta Operators

## 1. Introduction

Clinical narratives are dynamic. A diagnosis may be suspected, weakened by a negative test, confirmed by a later result, treated, improved, resolved, or recurrent. Many clinical NLP systems identify entities, assertions, or temporal events, but they do not explicitly represent the update operation that new evidence applies to a clinical proposition.

MedEST-Open frames this problem as Medical Epistemic State Tracking. Given an old proposition state and a new evidence segment, the model predicts a semantic transition and updated state.

## 2. Task

Each example contains:

- old state
- clinical proposition
- evidence segment
- transition operator
- new state

The transition operator is the central prediction target.

## 3. Data

The prototype uses MACCROBAT, an open annotated clinical case-report dataset. A weak-labeling pipeline converts annotated mentions and evidence segments into MedEST examples.

## 4. Model

The EpiDelta input combines old state, proposition text, and evidence text into a single model input. Current results use PubMedBERT embeddings and classical classifiers. Neural multitask EpiDelta models are included as the next step.

## 5. Results

| Task | Mode | Best model | Accuracy | Macro F1 | Weighted F1 |
|---|---|---|---:|---:|---:|
| New state | all_state | logreg_scaled_balanced | 0.6795 | 0.5562 | 0.6773 |
| Transition | all_transition | logreg_balanced | 0.7072 | 0.4488 | 0.7162 |
| Transition | filtered_transition | logreg_balanced | 0.7369 | 0.6641 | 0.7387 |

## 6. Error analysis

The strongest confusions are introduced/treated and improving/resolved. These errors reflect both modeling challenges and weak-label noise.

## 7. Limitations

Labels are weak and require manual validation. MACCROBAT is small. Broader context may be necessary for some transitions.

## 8. Conclusion

MedEST-Open suggests that semantic state-update modeling is a useful framing for clinical NLP. The current prototype demonstrates learnable signal and establishes a path toward larger-scale open medical epistemic tracking.
