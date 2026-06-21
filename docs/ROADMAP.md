# Roadmap

## Phase 1: recruiter-facing polish

- public repo
- clean README
- result summary
- method note
- sample example
- no raw data or artifacts in git

## Phase 2: scientific defensibility

- manual audit of 100 weak labels
- manual audit of 100 model errors
- report weak-label quality
- document common noise sources

## Phase 3: modeling depth

- neural EpiDelta multitask model
- ablations:
  - evidence only
  - proposition + evidence
  - old state + evidence
  - old state + proposition + evidence
- compare against local-state and majority baselines

## Phase 4: scale

- extend to larger open-access case-report corpora
- improve proposition normalization
- add temporal context windows
- evaluate cross-corpus transfer

## Phase 5: paper package

- short paper draft
- figure: MedEST state update diagram
- table: task distributions and results
- error-analysis section
- limitations and ethics statement
