# Incremental Maintenance of Dynamic Semantic Queries

This artifact provides the source code, selected inputs, and experimental records for the D03 paper.

This repository contains source snapshots, controlled input tables, historical model returns, and analysis scripts for the D03 paper. It evaluates a fixed binary predicate with filter and row-label equality-join results. Quality guarantees are conditional on the stated facts and event contract. Service, fees and branch counts are logical metrics; the results do not establish production latency or real annotation cost.

## Environment

The recorded controller replay used Windows, Python 3.11.5, NumPy 1.24.3, SQLite from Python, and one BLAS thread. Install NumPy with `python -m pip install -r requirements.txt`. The first three checks below use cached model returns and do not require a GPU. Set `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, and `MKL_NUM_THREADS=1` before starting Python. Use a fresh clone for each replay; the runners refuse to replace existing result directories.

## 1. Controlled comparison and paper tables

From the repository root:

```text
python src/run_d03_controlled_r.py prepare
python src/run_d03_controlled_r.py run
python src/verify_d03_controlled_r_replay.py
python src/summarize_paper_tables.py
```

The replay uses three fixed event sequences, two query types, two fee limits, three write limits, and 11 methods: 396 trajectories and 3,564 events. It uses stored model returns and performs no new model inference. The verification reopens the 396 SQLite databases and checks 36 WCR/WQCR pairs. The table script aggregates service, fees, writes, and branch counts. Original and replayed summaries matched on these fields in the local check.

The original pre-experiment verifier is retained as `src/verify_d03_controlled_r.py`. Its strict index comparison can fail on numerically tied scores. The separate post-experiment verifier `src/verify_d03_controlled_r_replay.py` checks score and mass values with a 1e-12 tie tolerance and verifies the selected action. The locally observed first strict mismatch had a score gap of 3.25e-19; the original failure and the tie-aware pass must be interpreted separately. This treatment does not imply that different non-tied actions are equivalent.

## 2. Finite instances and quotient stress test

```text
python src/verify_d03_j_fixtures.py
python src/run_d03_major_o.py prepare
python src/run_d03_major_o.py run
python src/verify_d03_major_o.py
```

The fixture entry point separates the finite-instance routine from the original J archive verifier. It checks 512 fixtures, 2,570 explicit branches, 4,248 full-audit paths and 1,536 full/quotient choices. The stress test uses 16,384 rows in six structured group settings. It tests branch evaluations and matched actions; it is not an end-to-end latency benchmark.

## 3. Recheck the historical 180-trajectory archive

This verification uses the saved J trajectories and SQLite databases. Obtain the exact seven Qwen files listed in `reference/model_files.json` under `models/qwen2.5-0.5b-instruct/`, then run:

```text
python src/verify_model_files.py
python src/verify_d03_materialized_j.py
```

The verifier checks the saved archive and finite fixtures. It reads the seven model files for provenance hashes; it does not run inference. The original local archive verification reopened 180 databases. The saved inputs use corpus labels as supplied facts, rather than independent human verification.

One source for these model files is the pinned Hugging Face revision below. Install the `hf` CLI separately and use a storage location with about 1 GB free:

```text
hf download Qwen/Qwen2.5-0.5B-Instruct config.json generation_config.json merges.txt model.safetensors tokenizer.json tokenizer_config.json vocab.json --revision 7ae557604adf67be50417f59c2c2f167def9a775 --local-dir models/qwen2.5-0.5b-instruct
python src/verify_model_files.py
```

All six smaller files from this revision were checked against the saved hashes. The upstream LFS SHA-256 for the weight file also matches the saved model hash. Other revisions may differ.

## 4. Financial corpus replay with saved predictions

After obtaining and checking the model files above, run in a fresh clone:

```text
python src/run_d03_major_p_fpb.py prepare
python src/restore_financial_predictions.py
python src/run_d03_major_p_fpb.py run
python src/verify_d03_major_p_fpb.py
```

The restore step checks and copies four historical prediction files. It does not generate new predictions. The recorded inference produced 192 outputs; the controller replay uses those stored outputs in 36 trajectories. The local replay and independent verifier checked 180 events and 18 matched WCR/WQCR pairs. The original fresh inference and this saved-return replay are distinct activities.

## Data and code

The D03 authors' original source code, including the D03-owned source snapshots, is released under the MIT licence in `LICENSE`. This licence does not apply to third-party data or models. See `DATA_SOURCES.md` for their source revisions and terms. Financial PhraseBank and its derived material remain subject to CC BY-NC-SA 3.0, including its attribution, noncommercial, and share-alike conditions. The model is obtained separately. The exact provenance and redistribution terms of the underlying movie inputs remain subject to the original sources' terms.
