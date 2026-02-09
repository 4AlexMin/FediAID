"""Run ablation evaluations comparing CSA and baseline models.

This script loads a validation dataset and evaluates:
- A trained CSA model checkpoint (your full model)
- A baseline model checkpoint (trained using `train_baseline.py`) or a
  baseline trained in this run

It prints F1 and AUROC for easy comparison.
"""

from __future__ import annotations

import argparse
import torch
from torch.utils.data import DataLoader
from typing import Optional

from src.config import get_default_device, set_hf_mirror
set_hf_mirror()
from .dataset import AIGTDataset, CommunityMemory
from .model.csa import CSADetector, BaselineDetector


def load_checkpoint(path: str):
    data = torch.load(path, map_location="cpu")
    return data


def evaluate_model(model, loader, device):
    model.eval()
    import numpy as np
    from sklearn.metrics import f1_score, roc_auc_score
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
    f1 = f1_score(labels_all, preds)
    try:
        auc = roc_auc_score(labels_all, probs)
    except Exception:
        auc = float("nan")
    return total_loss / len(loader.dataset), f1, auc


def main():
    parser = argparse.ArgumentParser(description="Ablation evaluation: CSA vs Baseline")
    parser.add_argument("--val", required=True)
    parser.add_argument("--memory", required=True)
    parser.add_argument("--csacheck", default=None, help="Path to CSA model checkpoint (model.pt)")
    parser.add_argument("--baselinecheck", default=None, help="Path to baseline checkpoint (baseline.pt)")
    parser.add_argument("--encoder", default=None)
    parser.add_argument("--lambda_mmr", type=float, default=0.45, help="MMR lambda for community retrieval")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", default=None)
    parser.add_argument("--hf_mirror", default=None)
    args = parser.parse_args()

    if args.hf_mirror:
        set_hf_mirror(args.hf_mirror)

    device = torch.device(args.device if args.device else get_default_device())
    memory = CommunityMemory(args.memory)
    val_ds = AIGTDataset(
        args.val,
        memory,
        tokenizer_name=args.encoder,
        model_name=args.encoder,
        k=3,
        lambda_mmr=args.lambda_mmr,
        device=device,
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    if args.csacheck:
        ck = load_checkpoint(args.csacheck)
        emb_dim = ck.get("emb_dim", memory.vecs.shape[1])
        model = CSADetector(emb_dim=emb_dim, hidden_dim=512, dropout=0.1) # hidden_dim=768
        model.load_state_dict(ck["state_dict"])
        model.to(device)
        loss, f1, auc = evaluate_model(model, val_loader, device)
        print(f"CSA model: loss={loss:.4f}, f1={f1:.4f}, auc={auc:.4f}")

    if args.baselinecheck:
        ck = load_checkpoint(args.baselinecheck)
        emb_dim = ck.get("emb_dim", memory.vecs.shape[1])
        model = BaselineDetector(emb_dim=emb_dim, hidden_dim=512, dropout=0.1) # hidden_dim=768
        model.load_state_dict(ck["state_dict"])
        model.to(device)
        loss, f1, auc = evaluate_model(model, val_loader, device)
        print(f"Baseline model: loss={loss:.4f}, f1={f1:.4f}, auc={auc:.4f}")


if __name__ == "__main__":
    main()
