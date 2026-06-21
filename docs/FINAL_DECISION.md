# Final Experiment Decision

## Primary result

The primary MedEST-Open result is the filtered 7-way transition task using PubMedBERT EpiDelta embeddings and balanced logistic regression.

- Accuracy: 0.7369
- Macro F1: 0.6641
- Weighted F1: 0.7387

This is the current strongest result and should be treated as the main paper/repo headline.

## Neural EpiDelta result

A small multitask neural classifier was trained over the same EpiDelta embedding representation.

- Transition accuracy: 0.6743
- Transition macro F1: 0.5796
- State accuracy: 0.5778
- State macro F1: 0.4579

The neural model underperformed the balanced logistic-regression baseline. This suggests that, under the current weak-label data regime, pretrained embeddings plus stable linear classifiers outperform a higher-capacity multitask head.

## Weak-label audit

An automatic heuristic weak-label audit was run to estimate likely noise modes.

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

This should be described as an automatic heuristic audit, not a manual or expert annotation study.

## Paper framing

The paper should emphasize:

1. A new task formulation: Medical Epistemic State Tracking.
2. Evidence-conditioned transition prediction.
3. Fully open-data prototype based on MACCROBAT.
4. Strong linear separability of transition states from EpiDelta PubMedBERT embeddings.
5. Clear remaining challenges: weak-label noise, rare transitions, and resolved/recurrent ambiguity.
