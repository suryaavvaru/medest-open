# MedEST-Open: Evidence-Conditioned Medical Epistemic State Tracking

## Abstract

Clinical narratives continually revise the status of medical propositions as new evidence appears. We introduce MedEST-Open, an open research prototype for Medical Epistemic State Tracking: predicting how textual evidence updates proposition-level clinical states. The task is formulated as an evidence-conditioned semantic update problem: old state plus proposition plus evidence maps to a transition operator and a new epistemic state. Using MACCROBAT-derived weak labels and PubMedBERT EpiDelta embeddings, balanced logistic regression achieves 0.664 macro F1 on a 7-way filtered transition task under case-level splitting. A small multitask neural head underperforms this linear baseline, suggesting that the current weak-label data regime favors stable linear classifiers over higher-capacity models. An automatic heuristic audit estimates that 76% of a 500-example sample is likely usable, with remaining errors mainly from ambiguous transitions and noisy proposition spans.

## 1. Introduction

Medical case narratives are not static lists of entities. They describe evolving diagnostic beliefs, treatment actions, clinical responses, and recurrences. Existing clinical NLP work often focuses on entity extraction, relation extraction, uncertainty detection, or temporal ordering. MedEST-Open instead focuses on proposition-level epistemic state updates.

The central question is:

> Given an old clinical proposition state and a new evidence sentence, what semantic transition occurs?

Examples include suspected to ruled out, suspected to confirmed, not mentioned to treated, and improving to resolved.

## 2. Task Definition

Each MedEST example contains:

- a clinical proposition,
- an old epistemic state,
- a textual evidence segment,
- a transition operator,
- a new epistemic state.

The main transition labels are:

- introduced
- treated
- ruled_out
- improving
- resolved
- no_change
- recurrent

The broader weak-label set also contains rare labels such as confirmed, improved, strengthened, recurred, and weakened.

## 3. Data

The current prototype uses MACCROBAT case reports and annotations. The processing pipeline produced:

- 200 cases
- 5,480 segments
- 25,189 mentions
- 4,296 weak MedEST examples

The filtered 7-way transition task contains 4,207 examples.

## 4. Model

EpiDelta represents each instance as an evidence-conditioned state update:

> old state + clinical proposition + evidence sentence -> transition operator + new state

Text inputs are embedded with PubMedBERT. Classical classifiers and a small neural multitask head are evaluated on top of these embeddings.

## 5. Results

### 5.1 Main Results

| Task | Mode | Best model | Examples | Accuracy | Macro F1 | Weighted F1 |
|---|---|---|---:|---:|---:|---:|
| New state prediction | all_state | Logistic Regression, scaled balanced | 4,296 | 0.6795 | 0.5562 | 0.6773 |
| Transition prediction | all_transition | Logistic Regression, balanced | 4,296 | 0.7072 | 0.4488 | 0.7162 |
| Transition prediction | filtered_transition | Logistic Regression, balanced | 4,207 | 0.7369 | 0.6641 | 0.7387 |
| Transition prediction | filtered_transition | Neural multitask head | 4,207 | 0.6743 | 0.5796 | 0.6700 |

The strongest result is balanced logistic regression over EpiDelta PubMedBERT embeddings on the 7-way filtered transition task.

### 5.2 Neural Result

The neural multitask head reached 0.5796 macro F1 on filtered transition prediction, below the 0.6641 macro F1 linear baseline. This suggests that the current weak-label regime benefits more from pretrained representations and stable linear decision boundaries than from higher-capacity classifiers.

## 6. Weak-Label Audit

An automatic heuristic audit was used to estimate likely label quality and noise modes. This is not a replacement for expert annotation.

### 100-example audit

- Good: 84%
- Bad proposition: 8%
- Bad transition: 4%
- Ambiguous: 4%

### 500-example audit

- Good: 76.0%
- Ambiguous: 10.6%
- Bad transition: 6.8%
- Bad proposition: 6.6%

The dominant noise modes are noisy proposition spans and ambiguous distinctions between introduction, treatment, improvement, and resolution.

## 7. Error Analysis

The strongest confusion patterns include:

- introduced -> treated
- treated -> introduced
- improving -> resolved
- resolved -> improving
- ruled_out -> introduced
- recurrent -> treated

These confusions are clinically plausible and reflect the difficulty of separating state introduction from actions, and partial improvement from full resolution.

## 8. Limitations

This prototype uses weak labels rather than expert adjudicated labels. MACCROBAT contains only 200 case reports, so rare transition classes remain underrepresented. The audit is heuristic and should be replaced by manual expert validation in future work.

## 9. Future Work

Future work should include:

- expert manual validation,
- larger-scale PMC open-access case report mining,
- improved proposition normalization,
- better rare-transition handling,
- sequence-aware state tracking,
- stronger neural models trained on larger data.

## 10. Conclusion

MedEST-Open demonstrates that evidence-conditioned clinical state-update prediction is feasible using fully open data. The current best result, 0.664 macro F1 on 7-way transition prediction, supports the central claim that semantic update operators over clinical propositions are learnable from medical case narratives.
