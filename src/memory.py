"""Utilities to build community embeddings for the DOSN-AIGT detector.

This module contains functions and a command-line interface to construct a
"memory bank" of community centroid vectors.  The memory bank is used to
retrieve the most similar communities for a given post during inference.

Each community embedding is the mean of the L2-normalized post embeddings
within that community.  If precomputed post embeddings are stored in the
JSONL input file (as the ``post_emb`` field), they are used directly.
Otherwise, text is encoded on the fly via a HuggingFace transformer
encoder.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional
import torch
import numpy as np

from src.io import load_jsonl_dataset
from src.config import get_default_device, set_hf_mirror
set_hf_mirror()

from transformers import AutoModel, AutoTokenizer


def _normalize(vec: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norm = np.linalg.norm(vec, axis=-1, keepdims=True) + eps
    return vec / norm


def build_memory(
    data_path: str,
    output_path: str,
    model_name: str = "roberta-base",
    max_length: int = 256,
) -> None:
    """Compute community centroids and save to a ``.npz`` file.

    Args:
        data_path: Path to a JSON Lines file containing training posts.  Each
            line must include ``community_id`` and either ``post_emb`` or
            ``text``.
        output_path: Where to write the memory bank (numpy ``.npz`` file).
        model_name: Name of the HuggingFace model to use for encoding if
            ``post_emb`` is not provided.  Ignored if all rows have
            precomputed ``post_emb``.
        max_length: Maximum sequence length for encoding.
    """
    # Read data
    rows = load_jsonl_dataset(data_path)
    use_precomputed = "post_emb" in rows[0]
    # Prepare encoder if needed
    if not use_precomputed:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model.eval()

        # choose device using the project's config helper; caller should manually
        # set CUDA_VISIBLE_DEVICES when running to reserve a GPU (e.g.
        # `CUDA_VISIBLE_DEVICES=3 python build_memory.py`).
        device = get_default_device()
        model.to(device)

        def encode_text(text: str) -> np.ndarray:
            """
            mean-pooled sentence embeddings
            """
            with torch.no_grad():
                tokens = tokenizer(
                    text,
                    max_length=max_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                ).to(device)
                outputs = model(**tokens)
                hidden = outputs.last_hidden_state  # [1, seq_len, dim]
                mask = tokens["attention_mask"].unsqueeze(-1).float()
                summed = (hidden * mask).sum(dim=1)
                counts = mask.sum(dim=1) + 1e-9
                mean = summed / counts
                return mean.squeeze(0).cpu().numpy()
            
    # Accumulate sums and counts per community
    sums: Dict[str, np.ndarray] = defaultdict(lambda: None)
    counts: Dict[str, int] = defaultdict(int)
    for row in rows:
        cid = row["community_id"]
        if use_precomputed:
            vec = np.asarray(row["post_emb"], dtype=np.float32)
        else:
            vec = encode_text(row["text"])
        vec = _normalize(vec)
        if sums[cid] is None:
            sums[cid] = vec.copy()
        else:
            sums[cid] += vec
        counts[cid] += 1
        
    # Compute centroids
    cids: List[str] = []
    cents: List[np.ndarray] = []
    for cid, total in sums.items():
        count = counts[cid]
        centroid = _normalize(total / max(count, 1))
        cids.append(cid)
        cents.append(centroid)
    cids_arr = np.array(cids)
    cents_arr = np.stack(cents).astype(np.float32)
    np.savez(output_path, cids=cids_arr, cents=cents_arr)
    print(f"Saved memory bank with {len(cids_arr)} communities to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build community memory bank")
    parser.add_argument("--data", required=True, help="Path to JSONL training file")
    parser.add_argument("--out", required=True, help="Output path for .npz memory bank")
    parser.add_argument(
        "--model",
        default="roberta-base",
        help="HuggingFace model name to use if post_emb is not present",
    )
    parser.add_argument(
        "--max_length", type=int, default=256, help="Maximum token length for encoding"
    )
    args = parser.parse_args()
    build_memory(args.data, args.out, args.model, args.max_length)


if __name__ == "__main__":
    main()
