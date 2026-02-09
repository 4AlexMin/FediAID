#!/usr/bin/env python
"""
High-throughput JSONL post encoder with HuggingFace transformers.

Optimised for large datasets:
- decouples IO chunking from GPU batch size
- uses torch.inference_mode + AMP
- fast tokenizer + parallelism
- streaming JSONL read/write
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List

import numpy as np
import torch
from tqdm import tqdm

from src.config import get_default_device, set_hf_mirror

# Set HF mirror before importing transformers
set_hf_mirror()
from transformers import AutoModel, AutoTokenizer


# -------------------------
# Utils
# -------------------------

def _normalize(vec: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norm = np.linalg.norm(vec, axis=-1, keepdims=True) + eps
    return vec / norm


def mean_pool(tokens, outputs) -> torch.Tensor:
    hidden = outputs.last_hidden_state          # [B, L, D]
    mask = tokens["attention_mask"].unsqueeze(-1).float()
    summed = (hidden * mask).sum(dim=1)         # [B, D]
    counts = mask.sum(dim=1) + 1e-9             # [B, 1]
    return summed / counts


# -------------------------
# IO chunking
# -------------------------

def record_chunks(path: Path, chunk_size: int) -> Iterable[List[dict]]:
    buf: List[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue
            buf.append(obj)
            if len(buf) >= chunk_size:
                yield buf
                buf = []
    if buf:
        yield buf


# -------------------------
# Embedding
# -------------------------

def embed_texts(
    model,
    tokenizer,
    texts: List[str],
    device: torch.device,
    batch_size: int,
    max_length: int,
) -> np.ndarray:
    """Embed a list of texts in fixed-size GPU batches."""
    all_embs = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]

        with torch.inference_mode(), torch.amp.autocast(
            device_type=device.type,
            enabled=(device.type == "cuda"),
        ):
            
            tokens = tokenizer(
                batch,
                max_length=max_length,
                truncation=True,
                padding=True,
                return_tensors="pt",
            ).to(device)

            outputs = model(**tokens)
            pooled = mean_pool(tokens, outputs)  # [B, D]

        embs = pooled.detach().cpu().numpy().astype(np.float32)
        embs = _normalize(embs)
        all_embs.append(embs)

    return np.vstack(all_embs)


# -------------------------
# Main processing
# -------------------------

def process_file(
    in_path: str,
    out_path: str | None,
    encoder: str,
    batch_size: int,
    max_length: int,
    skip_existing: bool,
    device_str: str | None,
) -> None:
    inp = Path(in_path)
    outp = Path(out_path) if out_path else inp.with_name(inp.stem + "_emb.jsonl")
    outp.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(device_str if device_str else get_default_device())

    tokenizer = AutoTokenizer.from_pretrained(encoder, use_fast=True)
    model = AutoModel.from_pretrained(encoder).to(device)
    model.eval()

    io_chunk_size = batch_size * 16  # 🔥 decoupled IO batching

    total_written = 0
    with outp.open("w", encoding="utf-8") as fo:
        for chunk in tqdm(
            record_chunks(inp, io_chunk_size), 
            desc=f"Encoding {inp.name}",
            dynamic_ncols=True,
            mininterval=60,    # print at most every N seconds
            leave=False,      # do not keep all lines
            ):

            to_encode_idx = []
            texts = []

            for i, rec in enumerate(chunk):
                if skip_existing and isinstance(rec.get("post_emb"), list):
                    continue
                txt = rec.get("text")
                if not txt:
                    continue
                to_encode_idx.append(i)
                texts.append(str(txt))

            embs = None
            if texts:
                embs = embed_texts(
                    model=model,
                    tokenizer=tokenizer,
                    texts=texts,
                    device=device,
                    batch_size=batch_size,
                    max_length=max_length,
                )

            emb_cursor = 0
            for i, rec in enumerate(chunk):
                if embs is not None and emb_cursor < len(to_encode_idx) and i == to_encode_idx[emb_cursor]:
                    rec["post_emb"] = embs[emb_cursor].round(6).tolist()
                    emb_cursor += 1

                fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
                total_written += 1

    print(f"Wrote {total_written} records to {outp}")


# -------------------------
# CLI
# -------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Fast JSONL post encoder")
    p.add_argument("--in", dest="inp", required=True, help="Input JSONL")
    p.add_argument("--out", default=None, help="Output JSONL")
    p.add_argument("--encoder", required=True, help="HF encoder name")
    p.add_argument("--batch_size", type=int, default=64, help="GPU batch size")
    p.add_argument("--max_length", type=int, default=256, help="Max token length")
    p.add_argument("--skip_existing", action="store_true", help="Skip records with post_emb")
    p.add_argument("--device", default=None, help="Device override")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    process_file(
        in_path=args.inp,
        out_path=args.out,
        encoder=args.encoder,
        batch_size=args.batch_size,
        max_length=args.max_length,
        skip_existing=args.skip_existing,
        device_str=args.device,
    )


if __name__ == "__main__":
    main()