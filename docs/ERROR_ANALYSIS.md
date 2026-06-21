# Error Analysis Summary

The current best filtered transition model reaches 0.7369 accuracy and 0.6641 macro F1.

Top observed confusion families:

1. `introduced` vs `treated`
2. `treated` vs `introduced`
3. `improving` vs `resolved`
4. `resolved` vs `improving`
5. `ruled_out` vs `introduced`

## Interpretation

### Introduced vs treated

Some sentences both introduce a clinical entity and describe an intervention. Weak labels may assign treatment to multiple propositions in the same sentence, including propositions that are merely mentioned.

### Improving vs resolved

Clinical improvement and full resolution are semantically close. The difference often depends on longitudinal context beyond one sentence.

### Ruled out vs introduced

Negated findings can both introduce an entity and rule it out. This motivates a stronger state-update formulation that explicitly conditions on the old state and proposition type.

## Next audit plan

Sample 100 model errors and label each as:

- correct weak label, model error
- bad proposition
- bad weak transition
- ambiguous clinical evidence
- requires broader context

This will separate modeling limitations from weak-label noise.
