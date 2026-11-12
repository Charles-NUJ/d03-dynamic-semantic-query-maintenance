# Budgeted Repair of Versioned Semantic Joins with Input Signatures

This repository contains the code, frozen inputs, semantic outputs, and verification records for the accompanying ICDE research paper.

## Problem and method

A versioned document store maintains semantic values that feed a filter and a join. A commit changes some operator inputs, while a fixed call budget may cover only part of that changed layer before publication. Stable inputs can reuse their stored values exactly. Our method ranks changed cells by their current join degree under unit call costs. Under positive heterogeneous costs, the selection becomes a 0/1 knapsack problem solved by dynamic programming. The repository separates information visible at selection time from new semantic values used only after a call.

## Repository layout

- `code/`: evaluation, optimization, ablation, plotting, and independent verification programs.
- `inputs/`: two frozen English Wikipedia revision windows and their collection contracts.
- `results/`: committed semantic outputs, query relations, method measurements, and verification records.
- `DATA_SOURCES.md`: upstream sources, attribution, and license boundaries.
- `SHA256SUMS.txt`: hashes of all published files except the manifest itself.

Check the published file inventory with:

```bash
python code/verify_sha256_manifest.py --root . --manifest SHA256SUMS.txt
```

## Quick replay from frozen semantic outputs

Use Python 3.11 in a clean environment.

```bash
python -m pip install -r requirements-replay.txt
python code/structured_database_eval.py --output-dir reproduced_results/main --random-repetitions 1000
python code/verify_structured_database_eval.py --result-dir reproduced_results/main
```

The replay builds the local SQLite relations, evaluates all methods under the same budgets, and checks the generated tables against structural and metric invariants.

## Core optimization checks

```bash
python code/verify_minimax_theorem.py --output reproduced_results/minimax_theorem.json
python code/heterogeneous_repair/test_signature_knapsack.py
python code/heterogeneous_repair/exhaustive_heterogeneous_micro.py --output reproduced_results/exhaustive_heterogeneous_micro.json
```

The first check covers the unit-cost minimax rule. The latter two checks compare the heterogeneous-cost dynamic program with exhaustive enumeration on small instances.

## Mechanism attribution

```bash
python code/run_mechanism_attribution.py --artifact-dir . --output-dir reproduced_results/mechanism_attribution --repetitions 1000
python code/verify_mechanism_attribution.py --result-dir reproduced_results/mechanism_attribution --method-results reproduced_results/main/method_results.csv --output reproduced_results/mechanism_attribution/verification.json
```

This study holds the snapshot, budget, and publication rule fixed while changing input-signature gating and degree ordering.

## Heterogeneous token-cost study

```bash
python code/heterogeneous_repair/run_actual_heterogeneous_knapsack.py --rows results/pair_predicate_results.json --predictions results/predictions.json --reference results/reference_relation.json --output-json reproduced_results/actual_knapsack.json --output-csv reproduced_results/actual_knapsack.csv --max-tokens 768 --random-repetitions 1000
python code/heterogeneous_repair/run_cost_degree_alignment_study.py --rows results/pair_predicate_results.json --predictions results/predictions.json --reference results/reference_relation.json --output-json reproduced_results/cost_degree_alignment.json --output-csv reproduced_results/cost_degree_alignment.csv --max-tokens 768 --random-repetitions 1000
python code/heterogeneous_repair/verify_cost_degree_alignment_study.py --json reproduced_results/cost_degree_alignment.json --csv reproduced_results/cost_degree_alignment.csv --output reproduced_results/cost_degree_alignment_verification.json
```

## Optional model inference

Inference needs CUDA, `requirements-inference.txt`, and a local copy of Qwen2.5-0.5B-Instruct. Model weights are not part of this repository.

For the first window, set `D03_MODEL_PATH` to the local model directory and run:

```bash
python code/run_natural_revision_inference.py
```

The second-window runner accepts explicit `--contract`, `--pairs`, `--revisions`, `--model`, and `--output-dir` paths. Frozen outputs are already included, so the main evaluation does not require a GPU or model download.

Verify the committed second-window outputs without model inference with:

```bash
python code/verify_second_window_inference.py --contract inputs/second_window_20261110/frozen/contract.json --pairs inputs/second_window_20261110/frozen/frozen_pairs.json --old-pairs inputs/frozen_pairs.json --revisions inputs/second_window_20261110/revision_content/revisions.json --result-dir results/second_window_20261110/inference --output reproduced_results/second_window_inference_verification.json
```

## Result map

- Main fixed-budget comparison: `results/method_results.csv` and `results/summary.json`.
- Query relation and cell-level records: `results/reference_relation.json` and `results/pair_predicate_results.json`.
- Unit-cost mechanism study: `results/mechanism_attribution/`.
- Heterogeneous-cost checks: `results/heterogeneous_micro_20261112/`, `results/heterogeneous_token_table_20261112/`, and `results/cost_degree_alignment_20261112_v1/`.
- Independent second window: `results/second_window_20261110/`.
- Local execution-time endpoint: `results/local_execution_endpoint/`.

## Scope

The committed semantic values are fixed-model outputs rather than human labels. Exact reuse follows from complete fixed-operator input identity. The unit-cost ordering and the heterogeneous-cost knapsack use the current query relation and the stated budget contract. Literature-labelled methods in the result files are ability-matched adaptations under the same interface, not reproductions of external systems.

## License

The authors' code is released under the MIT License. Third-party text and model files retain their upstream terms; see `DATA_SOURCES.md`.
