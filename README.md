# FediAID: Deepfake-Text Detection

FediAID is a community-augmented detector for AI-generated text (AIGT). It
combines frozen DeBERTa-v3-base post embeddings with retrieved Fediverse
community representations through Community Similarity Attention (CSA) module.

## Repository structure

```text
.
├── checkpoints/
│   ├── model_k27.pt
│   └── memory_bank_complete_emb.npz
├── LICENSE
├── expected/
│   ├── opensrc_lambda_sensitivity.csv
│   ├── opensrc_metrics.csv
│   └── opensrc_metrics_per_seed.csv
├── scripts/
│   ├── run_lambda_sensitivity.py
│   ├── prepare_opensrc_datasets.py
│   └── run_opensrc_reproduction.py
├── tests/
│   └── test_reproduction_validation.py
├── src/
│   ├── dataset.py
│   ├── eval_opensrc.py
│   ├── encode_posts.py
│   ├── memory.py
│   ├── model/csa.py
│   └── train.py
├── environment.yml
├── requirements-data.txt
├── requirements-reproduce.txt
└── requirements.txt
```

The repository includes source code, a released checkpoint, and a community
memory bank. FediAID's source code is released under the MIT License (see
`LICENSE`). The checkpoint-specific settings are described under
`Evaluate with released checkpoint`.

## Getting started

### Install dependencies

For training and general use:

```bash
python -m pip install -r requirements.txt
```

For reproduction:

```bash
conda env create -f environment.yml
conda activate fediaid-reproduction
```

The validated reproduction environment is specified in `environment.yml`:
Python 3.10.21, PyTorch 2.14.1+cu130, Transformers 4.57.6, Tokenizers 0.22.2,
and Hugging Face Hub 0.36.2. It includes SentencePiece and pins protobuf below
version 5, which is required by the DeBERTa tokenizer. The equivalent pip
install is:

```bash
python -m pip install -r requirements-reproduce.txt
```

A CUDA-capable GPU is recommended for large datasets. CPU execution is
supported with a smaller batch size. As a consequence, CPU-only deepfake-text detection is supported by FediAID.

### Prepare your data.

FediAID consumes JSONL records with the following fields:

| Field | Required | Description |
|---|---:|---|
| `text` | yes | Post text. |
| `post_emb` | optional | A vector of floats representing the post text in embedding space. If omitted, the script computes embeddings on the fly. |
| `label` | yes | `0` for human-written text and `1` for AI-generated text. |
| `community_id` | required for community-aware training/evaluation | Community or instance identifier. |

For training and memory-bank construction, use records containing `text`,
`label`, and `community_id`. To create post embeddings from raw JSONL data:

```bash
python -m src.encode_posts \
  --in /path/to/posts.jsonl \
  --out /path/to/posts_emb.jsonl \
  --encoder microsoft/deberta-v3-base \
  --batch_size 256 \
  --device cuda:0
```

Use the same compatible Hugging Face backbone for `--encoder` and `--model`.
Ensure post embeddings and memory-bank vectors use compatible encoder
representations.

### Build a community memory bank

Build a normalized centroid for each community from the training data:

```bash
python -m src.memory \
  --data /path/to/train_emb.jsonl \
  --model microsoft/deberta-v3-base \
  --out outputs/memory_bank.npz
```

### Train FediAID

Train the model using the training and validation JSONL files:

```bash
python -m src.train \
  --train /path/to/train_emb.jsonl \
  --val /path/to/validation_emb.jsonl \
  --memory outputs/memory_bank.npz \
  --encoder microsoft/deberta-v3-base \
  --epochs 10 \
  --k 3 \
  --lambda_mmr 0.1 \
  --out outputs/model_k3.pt
```
*`--k` is an integer; `--lambda_mmr` is in [0, 1].*

### Evaluation

Evaluate a trained FediAID checkpoint with an explicitly supplied memory bank:

```bash
python -m src.eval_opensrc \
  --datasets /path/to/evaluation/*.jsonl \
  --memory /path/to/memory_bank.npz \
  --csacheck /path/to/model.pt \
  --encoder /path/to/encoder \
  --ks 1-30 \
  --lambda_mmr 0.1 \
  --batch_size 32 \
  --device cuda:0 \
  --out results/evaluation.xlsx
```
*`--ks` accepts a range/list; `--lambda_mmr` is in [0, 1].*

The evaluator writes performances for each input file. For
CPU execution, use `--device cpu` and reduce `--batch_size`, for example `256`.

#### Evaluate with released checkpoint

The released checkpoint expects 768-dimensional embeddings and was developed
using `k=27` and MMR `lambda=0.8`, selected on Fediverse training and validation
data. External target-platform evaluation retains `k=27` and uses inference implementation's fixed default `lambda=0.45` uniformly across all target datasets and seeds,
without target-label tuning. Under this unsupervised transductive protocol,
target labels are used only to compute evaluation metrics.

For the pinned encoder revision (specific `DeBERTa` version) used in raw-text reproduction, see the `Reproduction` section.

```bash
python -m src.eval_opensrc \
  --datasets /path/to/evaluation/*.jsonl \
  --memory checkpoints/memory_bank_complete_emb.npz \
  --csacheck checkpoints/model_k27.pt \
  --encoder microsoft/deberta-v3-base \
  --ks 27 \
  --batch_size 1024 \
  --device cuda:0 \
  --out results/evaluation.xlsx
```


#### Note: Released checkpoint training provenance

The released checkpoint was trained on the
[Fediverse Deepfake-Text Corpus](https://doi.org/10.5281/zenodo.22811976). Concatenating
`corpus_01.jsonl` through `corpus_09.jsonl` in numeric order yields 1,003,993 records.
The train split (first 80%) contains 803,195 records. The remaining 200,798 records form the validation split.
The recorded training configuration is:

| Setting | Value |
|---|---|
| Encoder | `microsoft/deberta-v3-base` |
| Retrieved communities | `k=27` |
| Training-time MMR lambda | `0.8` |
| Epochs | 10 |
| Batch size | 32 |
| Learning rate | `2e-4` |
| Weight decay | `0.01` |
| Maximum sequence length | 256 |

The following command corresponds training under the recorded settings. Prepare the
training data, validation data, and memory bank (released memory banck) locally, precomputed embeddings for both train and validation splits, and then:


```bash
python -m src.train \
  --train /path/to/train_emb.jsonl \
  --val /path/to/validation_emb.jsonl \
  --memory /path/to/memory_bank.npz \
  --encoder microsoft/deberta-v3-base \
  --epochs 10 \
  --batch_size 32 \
  --lr 0.0002 \
  --weight_decay 0.01 \
  --k 27 \
  --lambda_mmr 0.8 \
  --max_length 256 \
  --out outputs/model_k27.pt
```





## Runtime notes

GPU selection is explicit. Reserve a host GPU at the shell level so the
process sees it as `cuda:0`:

```bash
CUDA_VISIBLE_DEVICES=3 python -m src.eval_opensrc \
  --datasets /path/to/evaluation/*.jsonl \
  --memory /path/to/memory_bank.npz \
  --csacheck /path/to/model.pt \
  --ks 1-30 \
  --lambda_mmr 0.1 \
  --device cuda:0
```

If raw-text evaluation requires Hugging Face downloads, set `HF_ENDPOINT` or
`GAI_HF_MIRROR` before running.

# Reproduction

## Scope of reproduction

The manuscript-oriented reproduction target is the FediAID cross-platform
open-source evaluation:

- 12 external datasets: three AIGTBench datasets, five MultiSocial datasets,
  and four source-variation datasets (`deepfake`, `fox8`, `m4`, and
  `tweepfake`).
- Five matched seeds: `0`, `1`, `2`, `3`, and `4`.
- Released checkpoint: `checkpoints/model_k27.pt`.
- Released memory bank: `checkpoints/memory_bank_complete_emb.npz`.
- Metrics: loss, accuracy, F1, and ROC-AUC for each dataset and seed.



The manuscript's eight-platform main results are expected to be approximately:

| Metric | Mean | Population standard deviation |
|---|---:|---:|
| F1 | 0.864 | 0.051 |
| ROC-AUC | 0.856 | 0.075 |

The four source-variation datasets are included in the 12-dataset workflow and
in the per-dataset expected metrics.


## External evaluation datasets

The external datasets are intentionally not bundled with this repository.
Both preparation routes below produce the same raw directories
`opensrc_platforms_seed{0..4}/`, each containing the 12 JSONL files used by the
reproduction runner. Follow all source licenses and citation requirements.

### Directly Download from Zenodo

The authors provide a prepared five-seed bundle through Zenodo:

```text
Zenodo DOI: 10.5281/zenodo.22828007
```
The Zenodo bundle contains these raw-data directories:

```text
opensrc_platforms_seed0/
opensrc_platforms_seed1/
opensrc_platforms_seed2/
opensrc_platforms_seed3/
opensrc_platforms_seed4/
```

Each directory contains the same 12 JSONL files:

| Files | Original source |
|---|---|
| `AIGTB_medium.jsonl`, `AIGTB_quora.jsonl`, `AIGTB_reddit.jsonl` | [AIGTBench](https://huggingface.co/datasets/tarryzhang/AIGTBench) |
| `multisocial_discord.jsonl`, `multisocial_gab.jsonl`, `multisocial_telegram.jsonl`, `multisocial_twitter.jsonl`, `multisocial_whatsapp.jsonl` | [MultiSocial](https://zenodo.org/records/13846152) |
| `deepfake.jsonl`, `fox8.jsonl`, `m4.jsonl`, `tweepfake.jsonl` | Their respective original public dataset releases |

The Zenodo bundle contains raw records with `text` and `label`.

### Prepare datasets from public sources

The preparation script creates the same five raw seed directories from the
public source files. It expects the following source layout:

```text
/path/to/raw-sources/
├── multisocial_anonymized.csv
├── fox8_23_dataset.ndjson
├── tweepfake_deepfake_text_detection/data/splits/
│   ├── train.csv
│   ├── validation.csv
│   └── test.csv
├── M4/*.jsonl
└── synthetic-text-datasets/RedditBot.jsonl
```

AIGTBench is loaded directly from its Hugging Face dataset repository. Install
the data-preparation dependencies, then run:

```bash
python -m pip install -r requirements-data.txt
python -m scripts.prepare_opensrc_datasets \
  --source-root /path/to/raw-sources \
  --out-root /path/to/evaluation-data
```

This writes raw JSONL files under `opensrc_platforms_seed{0..4}/`.

## Generate embedding inputs

Generate `opensrc_platforms_seed{0..4}_emb/` from the corresponding raw
directories with `src/encode_posts.py`:

```bash
for seed in 0 1 2 3 4; do
  mkdir -p /path/to/evaluation-data/opensrc_platforms_seed${seed}_emb
  for input in /path/to/evaluation-data/opensrc_platforms_seed${seed}/*.jsonl; do
    name=$(basename "${input%.jsonl}")
    python -m src.encode_posts \
      --in "$input" \
      --out "/path/to/evaluation-data/opensrc_platforms_seed${seed}_emb/${name}_emb.jsonl" \
      --encoder microsoft/deberta-v3-base \
      --revision 8ccc9b6f36199bec6961081d44eb72fb3f7353f3 \
      --batch_size 256 \
      --device cuda:0
  done
done
```

*Precomputed embeddings are recommended for speed. If this step is skipped, evaluation will still work and compute embeddings on the fly.*



## Reproduce the evaluation

Using the locally generated embedding directories:

```bash
python scripts/run_opensrc_reproduction.py \
  --data-root /path/to/evaluation-data \
  --device cuda:0
```

To use the raw Zenodo files directly w/o precomputed embeddings, select the raw directory template explicitly:

```bash
python scripts/run_opensrc_reproduction.py \
  --data-root /path/to/evaluation-data \
  --data-template 'opensrc_platforms_seed{seed}' \
  --encoder microsoft/deberta-v3-base \
  --encoder-revision 8ccc9b6f36199bec6961081d44eb72fb3f7353f3 \
  --device cuda:0
```

This raw-text route is slower. The runner maps raw dataset filenames to the
canonical embedded-dataset names when checking the expected metrics.



For CPU execution:

```bash
python scripts/run_opensrc_reproduction.py \
  --data-root /path/to/evaluation-data \
  --device cpu \
  --batch-size 256
```

The runner evaluates all 12 datasets for seeds 0 through 4, writes one result
workbook per seed under `results/opensrc_reproduction/`, and writes
`summary.csv`. It compares per-dataset mean F1 and ROC-AUC with
`expected/opensrc_metrics.csv` and per-seed F1/AUC with
`expected/opensrc_metrics_per_seed.csv` using an absolute tolerance of 0.02.
It fails on missing references, missing/duplicate datasets, duplicate
dataset/seed pairs, or non-finite metrics.

For a partial smoke test, select one seed and skip the five-seed comparison:

```bash
python scripts/run_opensrc_reproduction.py \
  --data-root /path/to/evaluation-data \
  --seeds 0 \
  --no-check-expected \
  --device cuda:0
```

## Expected results

`expected/opensrc_metrics.csv` records the five-seed mean and standard deviation
for each of the 12 datasets. `expected/opensrc_metrics_per_seed.csv` records
F1/AUC for each dataset and seed to help diagnose platform-specific variation.
The final aggregate supports the manuscript's cross-platform claims; small
floating-point differences can occur across compatible hardware/software
builds, so both references are checked with an absolute tolerance.

### Optional MMR sensitivity

The lambda-sensitivity experiment is separate from the default reproduction
workflow. It evaluates the released checkpoint with `k=27` on the eight main
target-platform datasets for `lambda_mmr` values from `0.0` through `1.0` in
steps of `0.1`. For each lambda, F1 and ROC-AUC are averaged across those eight
platforms. Run it explicitly with:

```bash
python scripts/run_lambda_sensitivity.py \
  --data-root /path/to/evaluation-data \
  --seed 0 \
  --device cuda:0
```

The separate expected summary is stored in
`expected/opensrc_lambda_sensitivity.csv`. The script uses only the released
`checkpoints/model_k27.pt`,
`checkpoints/memory_bank_complete_emb.npz`, and `k=27`.

## Code entry points

- `src/eval_opensrc.py`: evaluates a FediAID checkpoint on explicit JSONL files.
- `src/dataset.py`: loads post embeddings and retrieves community centroids.
- `src/model/csa.py`: defines the FediAID detector.
- `src/encode_posts.py`: creates normalized post embeddings.
- `scripts/prepare_opensrc_datasets.py`: prepares the five seeded external-data
  directories from public raw sources.
- `scripts/run_opensrc_reproduction.py`: runs and summarizes the 12-dataset,
  five-seed evaluation.
- `scripts/run_lambda_sensitivity.py`: optionally verifies the eight-platform
  lambda-sensitivity summary.
