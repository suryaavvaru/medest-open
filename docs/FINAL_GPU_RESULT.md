# Final GPU Transformer Result

## Primary result

The final MedEST-Open high-end text-transformer run fine-tuned PubMedBERT directly on the 7-way filtered transition task.

- Model: `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext`
- Task: 7-way MedEST transition prediction
- Examples: 4,207
- Device: CUDA
- GPU: NVIDIA GeForce RTX 3070 Ti Laptop GPU
- Batch size: 16
- Epochs: 8
- Best epoch: 5

## Final score

| Model | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| PubMedBERT fine-tuned text transformer | 0.9061 | 0.9075 | 0.9062 |
| PubMedBERT EpiDelta embeddings + balanced logistic regression | 0.7369 | 0.6641 | 0.7387 |
| Neural multitask EpiDelta head | 0.6743 | 0.5796 | 0.6700 |

## Interpretation

Fine-tuned PubMedBERT is the final winning model for the current weak-label MedEST-Open prototype. It substantially outperforms the previous linear-probe baseline on the 7-way filtered transition task.

The result should be framed carefully: this is a strong result on a weak-label MACCROBAT-derived benchmark, not a clinically validated expert-gold benchmark. The correct claim is that evidence-conditioned medical transition labels are highly learnable from biomedical language-model fine-tuning under the current open-data prototype.

## Paper headline

Fine-tuned PubMedBERT achieves 0.908 macro F1 on 7-way Medical Epistemic State Transition prediction in the MedEST-Open weak-label benchmark.

## Limitation

The benchmark remains weakly supervised. Future work should validate the task with expert-reviewed proposition trajectories and scale to larger PMC Open Access case-report corpora.
