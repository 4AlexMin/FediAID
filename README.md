# FediAID Deepfake-Text Detector

This repository provides an experimental implementation of a **community‑augmented detector** for **AI‑generated text (AIGT)**, namely **FediAID**.  The goal of this project is to explore whether incorporating *community‑level information* can improve the detection of AI‑generated posts and enable cross‑platform transferability.

## Overview

The research behind this repository hypothesises that communities (such as federated instances on Mastodon) develop distinct linguistic norms and adopt AI tools at different rates.  By representing these communities as dense embeddings and *retrieving* similar communities at inference time, we can provide the detector with valuable domain context while keeping the core model portable.  The key ideas are:

1. **Dataset** – A corpus of posts labelled as human‑written (HWT) or AI‑generated (AIGT), enriched with community identifiers.  Community labels are used as metadata rather than hard‑coded features.
2. **Detector (FediAID)** – A **Community Similarity Attention (CSA)** model that fine‑tunes a pretrained encoder for text classification and augments the resulting post representations with retrieved community embeddings.  Community representations are stored in an external memory bank and are not updated during training, preserving transferability.
3. **Analysis** – After training the detector, one can quantify the prevalence of AIGT across communities and even inspect ego‑networks (social graphs) to understand how many bots or AI‑assisted users surround a typical user.  Note: ego‑network analysis is outside the scope of this codebase and should be done separately.

## Repository Structure

```
/
├── README.md           # This readme
├── requirements.txt    # Python dependencies
├── src/                # Python source code
│   ├── __init__.py     # Makes src a package
│   ├── dataset.py      # Dataset and memory bank utilities
│   ├── memory.py       # Functions to build community embeddings
│   ├── model/          # Model components
│   │   ├── __init__.py # Makes model a package
│   │   └── csa.py      # Community Similarity Attention modules
│   └── train.py        # Training script
└── .gitignore          # Standard Git ignore patterns
```

## Getting Started

1. **Install dependencies** (ideally in a virtual environment):

   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare your data.**  You need two JSONL files: a training set and a validation set.  Each line should be a JSON object with the following fields:

   - `post_id`: a unique identifier for the post
   - `text`: the raw post text (optional if you precompute embeddings)
   - `label`: `0` for human‑written or `1` for AI‑generated
   - `community_id`: identifier for the community/instance the post belongs to
   - `post_emb`: a vector of floats representing the post text in embedding space (optional if you precompute embeddings; if omitted, the script will compute embeddings on the fly)

3. **Build a community memory bank** using the training data:

   ```bash
   python -m src.memory \
      --mode build \
      --data /your_data_repo/train.jsonl \
      --model microsoft/deberta-v3-base \
      --out outputs/memory_bank.npz \
      > logs/memory.log 2>&1
   ```

   This will compute a centroid vector for each community by averaging the pretrained encoder's embeddings of the posts belonging to that community.  The centroids are L2‑normalised and stored in a `.npz` file.

4. **Train the CSA detector** on your labelled data:

   ```bash
   python -m src.train \
       --train /your_data_repo/train.jsonl \
       --val /your_data_repo/validation.jsonl \
       --memory outputs/memory_bank.npz \
       --encoder microsoft/deberta-v3-base \
       --epochs 5 \
       --k 5 \
       --out outputs/model.pt \
       > logs/train.log 2>&1
   ```

   During training the model fine‑tunes the encoder and classifier while retrieving the top‐`k` nearest community centroids for each post.  At inference time the same memory bank is used to augment post embeddings.

5. **Evaluate** the trained model on held‑out communities or new platforms by constructing an appropriate memory bank and running inference.  The `src/train.py` script includes an inference mode for this purpose.



## Runtime notes: GPU reservation and HuggingFace mirrors

For reproducible GPU usage across shared machines we prefer manual GPU reservation at the shell level. Example:

```bash
# Reserve host GPU index 3 and run the training script. Inside the process,
# that GPU will be visible as `cuda:0`.
CUDA_VISIBLE_DEVICES=3 python -m src.train --train train.jsonl --val val.jsonl --memory memory_bank.npz --encoder roberta-base
```

The repository includes a small helper `src.config.get_default_device()` which follows the simple rule used in this project:

```python
device = "cuda:0" if torch.cuda.is_available() else "cpu"
```

If you prefer to use a mirror for HuggingFace model downloads (for example if you are on a restricted network), set the `HF_ENDPOINT` environment variable before running or set `GAI_HF_MIRROR` and the scripts will apply it automatically:

```bash
export HF_ENDPOINT=https://hf-mirror.com # also the default mirror when no environment variables are set.
# or
export GAI_HF_MIRROR=https://hf-mirror.com

CUDA_VISIBLE_DEVICES=3 python -m src.train --train train.jsonl --val val.jsonl --memory memory_bank.npz --encoder roberta-base
```

The codebase prefers the explicit, manual GPU reservation workflow to avoid automatically reassigning GPUs that other users may be using.

## Ablation experiments

This repository includes utilities and example workflows to run ablation experiments that compare the full CSA model against a baseline that uses only the pretrained encoder (no community retrieval / no fine-tuning of the encoder).

Files and scripts added for ablation experiments:

- `src/train_baseline.py`: trains a baseline classifier that uses only the post embedding. The script uses `AIGTDataset` to compute or load embeddings and trains a small MLP (the encoder is not updated by the baseline training loop by default).
- `src/eval_ablation.py`: evaluates a CSA checkpoint and/or a baseline checkpoint on a validation set and prints loss, F1 and AUROC for easy comparison.
- `examples/run_ablation.sh`: example shell workflow that (1) trains the CSA model, (2) trains the baseline with a frozen encoder, and (3) evaluates both models.

Ablation workflow

1. Train the full CSA model AS ABOVE.

2. Train the baseline classifier that uses the pretrained encoder but does not fine-tune it:

   ```bash
   python -m src.train_baseline \
    --train /your_data_repo/train.jsonl \
    --val /your_data_repo/validation.jsonl \
    --memory outputs/memory_bank.npz \
    --encoder microsoft/deberta-v3-base \
    --epochs 5 \
    --out outputs/baseline.pt \
    --freeze_encoder \
    > logs/train_baseline.log 2>&1
   ```

3. Evaluate both checkpoints on the validation set:
   ```bash
   python -m src.eval_ablation \
    --val /your_data_repo/validation.jsonl \
    --memory outputs/memory_bank.npz \
    --csacheck outputs/model.pt \
    --baselinecheck outputs/baseline.pt\
    --encoder microsoft/deberta-v3-base \
    > logs/eval_ablation.log 2>&1
   ```

Notes
- The baseline classifier uses an MLP of comparable capacity to the CSA classifier (for fair comparison) but omits the community-attention branch.
- If your dataset already contains `post_emb` fields (precomputed embeddings), training and evaluation will be faster because embedding computation is skipped. If not, the scripts will compute embeddings on the fly using the provided encoder name.
- Scripts respect the manual GPU reservation approach: set `CUDA_VISIBLE_DEVICES` at the shell level to reserve a host GPU index for the run.


## Open-source evaluation


- **Evaluate open-source datasets:** discovers `data/opensrc/*.jsonl`, evaluates a baseline checkpoint and CSA checkpoints (optionally across many `k` values), and writes results.
   ```bash
   python -u -m src.eval_opensrc \
      --datasets "/your_data_repo/opensrc/*.jsonl" \
      --memory outputs/memory_bank.npz \
      --encoder microsoft/deberta-v3-base \
      --batch_size 32 \
      --csacheck "outputs/models/model_k{k}.pt" \
      --baselinecheck outputs/baseline.pt \
      --ks 1-30 \
      --out results/opensrc_eval.xlsx \
      > logs/eval_opensrc.log 2>&1
   ```

   Notes:
   - `--csacheck` may contain a pattern with `{k}` (for per-k checkpoints) or point to one file reusedd for all ks.
   - The script records `dataset, model (baseline|csa), k, loss, f1, auc, checkpoint`.


## Naming convention

To avoid accidental mismatches between checkpoints, memory banks and evaluation parameters this repository follows a simple, machine‑parsable filename convention used by the example wrappers in `shells/`:

- CSA model checkpoint:

   `model_{backbone}_{dataset_id}_k{K}.pt`

- Baseline checkpoint:

   `baseline_{backbone}_{dataset_id}.pt`

- Memory bank:

   `memory_{backbone}_{dataset_id}.npz`

Replace slashes in `backbone` (e.g. `microsoft/deberta-v3-base`) with underscores when used in filenames (scripts do this automatically). Filenames use underscores as separators . Each saved checkpoint also contains embedded metadata (encoder/backbone, dataset_id, k, best_f1, created_at).

