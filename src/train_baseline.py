"""Training script for baseline classifier (no community retrieval).

This script mirrors `src.train` but trains a classifier that uses only the
post embedding. It supports freezing the encoder when computing embeddings
on the fly to simulate the "pretrained (not fine-tuned)" ablation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime

from src.config import get_default_device, set_hf_mirror
set_hf_mirror()
from .dataset import AIGTDataset, CommunityMemory
from .model.csa import BaselineDetector


def evaluate_baseline(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float, float]:
    model.eval()
    ce = nn.CrossEntropyLoss()
    total_loss = 0.0
    probs_list = []
    labels_list = []
    for post_emb, comm_embs, labels in loader:
        post_emb = post_emb.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            logits = model(post_emb)
            loss = ce(logits, labels)
        total_loss += loss.item() * post_emb.size(0)
        probs = torch.softmax(logits, dim=-1)[:, 1]
        probs_list.append(probs.cpu())
        labels_list.append(labels.cpu())
    import numpy as np
    from sklearn.metrics import f1_score, roc_auc_score
    probs = torch.cat(probs_list).numpy()
    labels_all = torch.cat(labels_list).numpy()
    preds = (probs > 0.5).astype(int)
    f1 = f1_score(labels_all, preds)
    try:
        auc = roc_auc_score(labels_all, probs)
    except Exception:
        auc = float("nan")
    return total_loss / len(loader.dataset), f1, auc


def train_baseline(
    train_path: str,
    val_path: str,
    memory_path: str,
    encoder_name: Optional[str] = None,
    epochs: int = 10, # 5
    batch_size: int = 16, # 32
    lr: float = 5e-4, # 2e-4
    weight_decay: float = 0.005, # 0.01
    max_length: int = 256,
    out_path: str = "baseline.pt",
    device: Optional[str] = None,
    freeze_encoder: bool = True,
    dataset_id: Optional[str] = None,
) -> None:


    dev = torch.device(device if device else get_default_device())
    memory = CommunityMemory(memory_path)
    emb_dim = memory.vecs.shape[1]

    train_ds = AIGTDataset(
        train_path,
        memory,
        tokenizer_name=encoder_name,
        model_name=encoder_name,
        k=0,
        max_length=max_length,
        device=dev,
    )
    val_ds = AIGTDataset(
        val_path,
        memory,
        tokenizer_name=encoder_name,
        model_name=encoder_name,
        k=0,
        max_length=max_length,
        device=dev,
    )

    # Baseline detector
    model = BaselineDetector(emb_dim=emb_dim, hidden_dim=512, dropout=0.1).to(dev) # hidden_dim=768

    # If encoder is provided and freeze_encoder is requested, the dataset
    # will compute embeddings using the encoder but it won't be updated here.
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    ce = nn.CrossEntropyLoss()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    best_f1 = -1.0
    best_state = None
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for post_emb, comm_embs, labels in tqdm(
        train_loader,
        desc=f"Training epoch {epoch}",
        dynamic_ncols=True,
        mininterval=60,    # print at most every N seconds
        leave=False,      # do not keep all lines
        ):
            post_emb = post_emb.to(dev)
            labels = labels.to(dev)
            logits = model(post_emb)
            loss = ce(logits, labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * post_emb.size(0)
        val_loss, val_f1, val_auc = evaluate_baseline(model, val_loader, dev)
        print(
            f"Epoch {epoch}: train_loss={total_loss/len(train_ds):.4f}, val_loss={val_loss:.4f}, val_f1={val_f1:.4f}, val_auc={val_auc:.4f}",
            file=sys.stderr,
        )
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
    if best_state is not None:
        meta = {
            "state_dict": best_state,
            "emb_dim": emb_dim,
            "k": 0,
            "encoder": encoder_name,
            "dataset_id": dataset_id,
            "best_f1": float(best_f1),
            "created_at": datetime.utcnow().isoformat() + "Z",
        }
        torch.save(meta, out_path)
        print(f"Saved best baseline model (F1={best_f1:.4f}) to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train baseline classifier (ablation)")
    parser.add_argument("--train", required=True, help="Path to training JSONL file")
    parser.add_argument("--val", required=True, help="Path to validation JSONL file")
    parser.add_argument("--memory", required=True, help="Path to community memory .npz file")
    parser.add_argument("--encoder", default=None, help="HuggingFace encoder name to compute embeddings (optional)")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--out", default="baseline.pt")
    parser.add_argument("--dataset_id", default=None, help="Short identifier for training dataset (used in filenames and metadata)")
    parser.add_argument("--device", default=None)
    parser.add_argument("--freeze_encoder", action="store_true", help="If set, encoder will not be fine-tuned (default behavior). The --freeze_encoder flag is only informational")
    args = parser.parse_args()
    train_baseline(
        train_path=args.train,
        val_path=args.val,
        memory_path=args.memory,
        encoder_name=args.encoder,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_length=args.max_length,
        out_path=args.out,
        device=args.device,
        freeze_encoder=args.freeze_encoder,
        dataset_id=args.dataset_id,
    )


if __name__ == "__main__":
    main()
