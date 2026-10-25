# Data and model provenance

## Movie review workload

The locally retained files were derived from the SemBench movie scenario at
commit c556cae469d7fe4873ab2ab337493cdb58ab5de3:

- Repository: https://github.com/SemBench/SemBench
- Source Reviews_2000.csv:
  https://raw.githubusercontent.com/SemBench/SemBench/c556cae469d7fe4873ab2ab337493cdb58ab5de3/docs/static/data/paper-2025-10-01/movie/data/Reviews_2000.csv
- Source Movies_2000.csv:
  https://raw.githubusercontent.com/SemBench/SemBench/c556cae469d7fe4873ab2ab337493cdb58ab5de3/docs/static/data/paper-2025-10-01/movie/data/Movies_2000.csv
- Maintainer clarification: https://github.com/SemBench/SemBench/issues/16

The maintainer identifies the movie scenario as open and states that datasets
retain their original licences. Verify the exact origin, applicable terms,
attribution, and distribution rights of the derived input files before release.
The repository software licence does not automatically establish rights to
every underlying dataset.

## Financial PhraseBank

The separate financial workload uses the all-agree corpus from Financial
PhraseBank at commit 8d3fe0c36d5feec6b3cc5e455b0fcb4820fb9964.
The upstream card specifies CC BY-NC-SA 3.0. Its noncommercial, attribution,
and share-alike conditions must be retained where applicable.

- https://huggingface.co/datasets/takala/financial_phrasebank/blob/8d3fe0c36d5feec6b3cc5e455b0fcb4820fb9964/README.md
- https://creativecommons.org/licenses/by-nc-sa/3.0/

Attribution: Pekka Malo, Ankur Sinha, Pekka Korhonen, Jyrki Wallenius, and
Pyry Takala. "Good debt or bad debt: Detecting semantic orientations in
economic texts." Journal of the Association for Information Science and
Technology, 2014. We selected 96 sentences and constructed dynamic update
sequences; the source licence continues to apply to the derived data.
The supplied labels are evaluation facts, not independent verification of
semantic truth. Users must follow the applicable attribution, noncommercial,
and share-alike conditions when redistributing or adapting these inputs.

## Model

The experiments used Qwen2.5-0.5B-Instruct. The model card lists Apache 2.0:
https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct

The controlled comparison replays fixed outputs and does not need model weights.
New inference requires separately obtaining the exact model files and validating
their hashes, as well as the relevant GPU environment.

## Own code

The D03 authors' original source code and D03-owned source snapshots are
licensed under the MIT licence in `LICENSE`. Third-party datasets and models
retain their own terms; the MIT licence does not cover them.
