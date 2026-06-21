# Final Experiment Decision

## Final primary result

The final winning MedEST-Open model is a fine-tuned PubMedBERT text transformer trained on the 7-way filtered transition task.

- Model: `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext`
- Device: CUDA
- GPU: NVIDIA GeForce RTX 3070 Ti Laptop GPU
- Examples: 4,207
- Best epoch: 5
- Accuracy: 0.9061
- Macro F1: 0.9075
- Weighted F1: 0.9062

## Baselines

| Model | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| Fine-tuned PubMedBERT | 0.9061 | 0.9075 | 0.9062 |
| PubMedBERT EpiDelta embeddings + balanced logistic regression | 0.7369 | 0.6641 | 0.7387 |
| Neural multitask EpiDelta head | 0.6743 | 0.5796 | 0.6700 |

## Weak-label audit

An automatic heuristic weak-label audit was used to estimate likely noise modes. This is not a replacement for expert annotation.

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

## Final paper framing

The final headline is:

> Fine-tuned PubMedBERT achieves 0.908 macro F1 on 7-way Medical Epistemic State Transition prediction in the MedEST-Open weak-label benchmark.

The claim should remain careful: this is a strong open-data weak-label prototype result, not a clinically validated expert-gold benchmark.
