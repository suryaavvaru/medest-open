# Results Summary

Current prototype results use MACCROBAT-derived weak MedEST examples, PubMedBERT EpiDelta embeddings, and case-level splitting.

## Best current results

| Task | Mode | Best model | Accuracy | Macro F1 | Weighted F1 |
|---|---|---|---:|---:|---:|
| New state | all_state | logreg_scaled_balanced | 0.6795 | 0.5562 | 0.6773 |
| Transition | all_transition | logreg_balanced | 0.7072 | 0.4488 | 0.7162 |
| Transition | filtered_transition | logreg_balanced | 0.7369 | 0.6641 | 0.7387 |

## Main result

The strongest result is the 7-way filtered transition task:

```text
Accuracy:    0.7369
Macro F1:    0.6641
Weighted F1: 0.7387
```

This is the current best evidence that EpiDelta-style inputs encode useful signal for semantic clinical state-update prediction.

## Interpretation

- The filtered transition task is more stable than the all-transition task because extremely rare labels are removed.
- The all-transition task remains useful for studying long-tail transition behavior.
- The new-state task is easier than transition prediction but still useful as an auxiliary head in multitask models.

## Next result targets

- filtered transition macro F1 >= 0.70
- all-transition macro F1 >= 0.50
- manual-label audit showing usable weak-label precision
- ablation showing the contribution of old state, proposition text, and evidence text
