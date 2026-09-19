#!/usr/bin/env python
"""Evaluate CSA and baseline models on one or more open-source JSONL datasets.

This script will:
- evaluate explicitly supplied external JSONL datasets
- for each dataset:
  - evaluate the baseline checkpoint once
  - evaluate the CSA checkpoint across a list of `k` values
- save aggregated results to an Excel file

Example:
    python -m src.eval_opensrc --datasets /path/to/eval/*.jsonl \
            --memory checkpoints/memory_bank_complete_emb.npz \
            --csacheck checkpoints/model_k27.pt \
            --ks 27 \
            --lambda_mmr 0.45

If `--csacheck` contains the string "{k}" it will be formatted with the k value
so you can point at per-k checkpoints (e.g. `outputs/model_k{k}.pt`).

Note:
Model development uses k=27 and MMR lambda=0.8, selected using Fediverse
training and validation data. External target-platform evaluation uses k=27
and a fixed inference-time lambda=0.45 uniformly across target datasets and
seeds. Under this unsupervised transductive protocol, target labels are used
only to compute evaluation metrics, not for inference or parameter selection.
"""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import pandas as pd
from typing import Sequence
import json

from src.config import get_default_device, set_hf_mirror
set_hf_mirror()
from .dataset import AIGTDataset, CommunityMemory
from .model.csa import CSADetector, BaselineDetector


def evaluate_model(model, loader, device):
    model.eval()
    import numpy as np
    from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
    import torch.nn as nn

    ce = nn.CrossEntropyLoss()
    total_loss = 0.0
    probs_list = []
    labels_list = []
    for post_emb, comm_embs, labels in loader:
        post_emb = post_emb.to(device)
        comm_embs = comm_embs.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            if isinstance(model, CSADetector):
                logits, _ = model(post_emb, comm_embs)
            else:
                logits = model(post_emb)
            loss = ce(logits, labels)
        total_loss += loss.item() * post_emb.size(0)
        probs = torch.softmax(logits, dim=-1)[:, 1]
        probs_list.append(probs.cpu())
        labels_list.append(labels.cpu())
    probs = torch.cat(probs_list).numpy()
    labels_all = torch.cat(labels_list).numpy()
    preds = (probs > 0.5).astype(int)
    acc = accuracy_score(labels_all, preds)
    f1 = f1_score(labels_all, preds)
    try:
        auc = roc_auc_score(labels_all, probs)
    except Exception:
        auc = float("nan")
    return total_loss / len(loader.dataset), acc, f1, auc


def find_datasets(pattern: str) -> list[str]:
    return sorted(glob.glob(pattern))


def load_checkpoint(path: str):
    return torch.load(path, map_location="cpu")



def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate CSA and baseline on opensrc datasets")
    parser.add_argument("--datasets", nargs="*", default=None, help="Paths to external JSONL evaluation files.")
    parser.add_argument("--memory", required=True, help="Path to community memory .npz file")
    parser.add_argument("--csacheck", default=None, help="Path to CSA checkpoint or pattern containing '{k}'")
    parser.add_argument("--baselinecheck", default=None, help="Path to baseline checkpoint")
    parser.add_argument("--encoder", default=None)
    parser.add_argument("--lambda_mmr", type=float, default=0.45, help="Inference-time MMR lambda (fixed external-evaluation default: 0.45; no target-label tuning)")
    parser.add_argument("--ks", default="27", help="Comma-separated ks or range like '1-10' or '1,2,3' (external-evaluation default: 27)")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", default="results/opensrc_eval.xlsx")
    args = parser.parse_args(argv)
    if args.datasets is None or len(args.datasets) == 0:
        datasets = []
    else:
        datasets = []
        for path in args.datasets:
            if "/*.jsonl" in path:
                datasets.extend(find_datasets(path))
            else:
                datasets.append(path)
                
    if not datasets:
        raise SystemExit("No datasets provided. Pass external JSONL files with --datasets.")
    else:
        print(f"Found {len(datasets)} datasets to evaluate.")
    
    # parse ks
    if ",-" in args.ks and "," not in args.ks:
        a, b = args.ks.split("-")
        ks = list(range(int(a), int(b) + 1))
    else:
        parts = [p.strip() for p in args.ks.split(",") if p.strip()]
        ks = []
        for p in parts:
            if "-" in p:
                a, b = p.split("-")
                ks.extend(range(int(a), int(b) + 1))
            else:
                ks.append(int(p))

    device = torch.device(args.device if args.device else get_default_device())
    memory = CommunityMemory(args.memory)

    rows = []
    for ds in datasets:
        print("===================================")
        print(f"Preparing dataset: {ds}")

        # 1. Create ONE dataset per file
        base_val_ds = AIGTDataset(
            ds, memory,
            tokenizer_name=args.encoder,
            model_name=args.encoder,
            k=None,
            lambda_mmr=args.lambda_mmr,
            device=device
        )

        # 2. Create ONE DataLoader per dataset (k updated dynamically)
        val_loader = DataLoader(
            base_val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0
        )

        # -------------------------------
        # Baseline evaluation (k=0)
        # -------------------------------
        if args.baselinecheck:
            print("----------------------------")
            print(f"Evaluating baseline on {ds}")

            print(f"Loading baseline checkpoint {args.baselinecheck}")
            ck = load_checkpoint(args.baselinecheck)
            emb_dim = ck.get("emb_dim", memory.vecs.shape[1])

            baseline_model = BaselineDetector(
                emb_dim=emb_dim,
                hidden_dim=512, # 768
                dropout=0.1
            )
            baseline_model.load_state_dict(ck["state_dict"])
            baseline_model.to(device)

            base_val_ds.set_k(0)
            loss, acc, f1, auc = evaluate_model(baseline_model, val_loader, device)

            rows.append({
                "dataset": os.path.basename(ds),
                "model": "baseline",
                "k": 0,
                "loss": float(loss),
                "acc": float(acc),
                "f1": float(f1),
                "auc": float(auc),
                "checkpoint": args.baselinecheck,
            })

            print(f"Baseline results on {os.path.basename(ds)}: loss={loss:.4f}, acc = {acc:.4f}, f1={f1:.4f}, auc={auc:.4f}")

            tmp_path = Path("results/opensrc_eval_partial_tmp.json")
            tmp_path.parent.mkdir(parents=True, exist_ok=True)
            json.dump(rows, open(tmp_path, "w"), indent=2)

        # -------------------------------
        # CSA evaluation for all ks
        # -------------------------------
        if args.csacheck:
            print("----------------------------")
            print(f"Evaluating CSA on {ds}")

            for k in ks:
                # Resolve checkpoint path
                ck_path = args.csacheck.format(k=k) if "{k}" in args.csacheck else args.csacheck
                if not Path(ck_path).exists():
                    print(f"Warning: CSA checkpoint {ck_path} not found, skipping k={k}")
                    continue

                print(f"Loading CSA checkpoint {ck_path} for k={k}")
                ck = load_checkpoint(ck_path)
                emb_dim = ck.get("emb_dim", memory.vecs.shape[1])

                model = CSADetector(
                    emb_dim=emb_dim,
                    hidden_dim=512, # 768
                    dropout=0.1
                )
                model.load_state_dict(ck["state_dict"])
                model.to(device)

                # Set k dynamically
                base_val_ds.set_k(k)

                loss, acc, f1, auc = evaluate_model(model, val_loader, device)
                rows.append({
                    "dataset": os.path.basename(ds),
                    "model": "csa",
                    "k": int(k),
                    "loss": float(loss),
                    "acc": float(acc),
                    "f1": float(f1),
                    "auc": float(auc),
                    "checkpoint": ck_path,
                })

                print(f"CSA results on {os.path.basename(ds)} (k={k}): loss={loss:.4f}, acc = {acc:.4f}, f1={f1:.4f}, auc={auc:.4f}")

                tmp_path = Path("results/opensrc_eval_partial_tmp.json")
                tmp_path.parent.mkdir(parents=True, exist_ok=True)
                json.dump(rows, open(tmp_path, "w"), indent=2)


    df = pd.DataFrame(rows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(out_path, index=False)
    print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()
