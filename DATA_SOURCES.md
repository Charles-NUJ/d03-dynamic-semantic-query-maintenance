# Data and model sources

## English Wikipedia revision windows

The two frozen revision windows under `inputs/` were collected from the English Wikipedia MediaWiki API. Each record retains the page identifier, revision identifiers, title, and revision metadata needed to trace the source through the page history. The snapshot and freeze receipts record the collection rule and hashes.

Wikipedia text is reused under the [Creative Commons Attribution-ShareAlike 4.0 License](https://creativecommons.org/licenses/by-sa/4.0/) and the [GNU Free Documentation License](https://www.gnu.org/licenses/fdl-1.3.html), subject to Wikipedia's licensing terms. Attribution is available from each page history by using the stored page and revision identifiers. The repository's MIT license applies to the authors' code and does not replace the license of Wikipedia content.

The first window is stored in:

- `inputs/frozen_pairs.json`
- `inputs/revisions.json`
- `inputs/recentchanges_raw.json`

The page-disjoint second window is stored in `inputs/second_window_20261110/`.

## Semantic model

The frozen semantic outputs were produced with [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct), released under the Apache License 2.0. Model weights are not copied into this repository. To rerun inference, download the model from its official source and pass its local directory to the inference script. The fixed operator uses deterministic decoding, a maximum input length of 768 tokens, and a two-token output limit as recorded in the frozen contracts.

## License boundary

- Authors' source code: MIT License.
- Wikipedia revision text: upstream CC BY-SA 4.0 and GFDL terms.
- Qwen model files: upstream Apache License 2.0; weights are omitted here.
- Generated tables and measurements: supplied for research verification with source provenance retained.
