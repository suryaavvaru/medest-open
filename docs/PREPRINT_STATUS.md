# MedEST-Open Preprint Status

Current final preprint package:

- Title: **MedEST-Open: A Weakly Supervised Benchmark for Evidence-Conditioned Medical Epistemic State Tracking**
- Authors: Surya Teja Avvaru and Krishna Sai Pokala
- Status: arXiv-style weakly supervised benchmark preprint
- Scientific caveat: not an expert-gold clinical benchmark and not clinical decision support

## Final result

| Model | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| Fine-tuned PubMedBERT | 0.9061 | 0.9075 | 0.9062 |

Best checkpoint: epoch 5.

## Submission caveat

This work should be submitted as a weakly supervised benchmark/prototype paper. A later journal-grade benchmark should add expert-gold proposition trajectories, repeated seeds, per-example predictions, and confusion-matrix analysis.
