"""Training script for the Community Similarity Attention (CSA) detector.

This script trains a CSA model on labelled posts with optional community
retrieval.  It expects a training JSONL file and a validation JSONL file,
both containing at minimum the fields ``label`` (0 or 1) and
``post_emb`` (precomputed embedding) or ``text`` (raw post content).
Community embeddings must be precomputed via ``src/memory.py`` and
provided as an ``.npz`` file.  The script fine‑tunes the CSA model and
outputs a checkpoint for later use.

Usage example:

```
python -m src.train \
  --train data/train.jsonl \
  --val data/validation.jsonl \
  --memory data/memory_bank.npz \
  --encoder roberta-base \
  --epochs 5 \
  --k 3 \
  --out model.pt
```

By default the script uses precomputed ``post_emb`` vectors and does not
update the underlying encoder.  If raw text and an encoder name are
provided, the dataset will compute embeddings on the fly; however this
training script does not update the encoder weights.  Updating the
encoder jointly with the classifier would require a more involved setup
and is left for future work.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os


from src.config import get_default_device, set_hf_mirror
set_hf_mirror()

from .dataset import AIGTDataset, CommunityMemory
from .model import CSADetector


def evaluate(
    model: CSADetector,
    loader: DataLoader,
    device: torch.device,
    threshold: float,
) -> tuple[float, float, float]:
    """Evaluate model on a DataLoader.

    Returns average loss, F1 score and AUROC.
    """
    model.eval()
    ce = nn.CrossEntropyLoss()
    total_loss = 0.0
    probs_list = []
    labels_list = []
    for post_emb, comm_embs, labels in loader:
        post_emb = post_emb.to(device)
        comm_embs = comm_embs.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            logits, _ = model(post_emb, comm_embs)
            loss = ce(logits, labels)
        total_loss += loss.item() * post_emb.size(0)
        probs = torch.softmax(logits, dim=-1)[:, 1]
        probs_list.append(probs.cpu())
        labels_list.append(labels.cpu())
    import numpy as np
    from sklearn.metrics import  accuracy_score, f1_score, roc_auc_score
    probs = torch.cat(probs_list).numpy()
    labels_all = torch.cat(labels_list).numpy()
    preds = (probs > threshold).astype(int)
    acc = accuracy_score(labels_all, preds)
    f1 = f1_score(labels_all, preds)
    try:
        auc = roc_auc_score(labels_all, probs)
    except Exception:
        auc = float("nan")
    return total_loss / len(loader.dataset), acc, f1, auc


def train(
    train_path: str,
    val_path: str,
    memory_path: str,
    encoder_name: Optional[str] = None,
    epochs: int = 10, # 5
    batch_size: int = 16, # 32
    lr: float = 5e-4, # 2e-4
    weight_decay: float = 0.005, # 0.01
    k: int = 3,
    max_length: int = 256,
    max_centroid_samples: int = -1,
    lambda_mmr: float = 0.45,
    threshold: float = 0.5,
    out_path: str = "model.pt",
    device: Optional[str] = None,
    dataset_id: Optional[str] = None,
) -> None:

    dev = torch.device(device if device else get_default_device())
    
    # Load memory
    memory = CommunityMemory(memory_path)
    # Determine embedding dimension from memory
    emb_dim = memory.vecs.shape[1]
    # Load datasets
    train_ds = AIGTDataset(
        train_path,
        memory,
        tokenizer_name=encoder_name,
        model_name=encoder_name,
        k=k,
        max_length=max_length,
        max_centroid_samples=max_centroid_samples,
        lambda_mmr=lambda_mmr,
        device=dev,
    )
    val_ds = AIGTDataset(
        val_path,
        memory,
        tokenizer_name=encoder_name,
        model_name=encoder_name,
        k=k,
        max_length=max_length,
        max_centroid_samples=max_centroid_samples,
        lambda_mmr=lambda_mmr,
        device=dev,
    )
    # Create model
    model = CSADetector(emb_dim=emb_dim, hidden_dim=512, dropout=0.1).to(dev) # hidden_dim=768
    # Training setup
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    ce = nn.CrossEntropyLoss()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    best_f1 = -1.0
    best_auc = -1
    best_loss = float("inf")
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
            comm_embs = comm_embs.to(dev)
            labels = labels.to(dev)
            logits, _ = model(post_emb, comm_embs)
            loss = ce(logits, labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * post_emb.size(0)
        # evaluate
        val_loss, val_acc, val_f1, val_auc = evaluate(model, val_loader, dev, threshold)
        print(
            f"Epoch {epoch}: train_loss={total_loss/len(train_ds):.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}, val_f1={val_f1:.4f}, val_auc={val_auc:.4f}",
            file=sys.stderr,
        )
        if val_auc > best_auc:
            best_acc = val_acc
            best_f1 = val_f1
            best_auc = val_auc
            best_loss = val_loss
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
    # Save best model
    if best_state is not None:
        meta = {
            "state_dict": best_state,
            "emb_dim": emb_dim,
            "k": k,
            "lambda_mmr": lambda_mmr,
            "threshold": threshold,
            "encoder": encoder_name,
            "dataset_id": dataset_id,
            "best_acc": float(best_acc),
            "best_f1": float(best_f1),
            "best_auc": float(best_auc),
            "val_loss": float(best_loss),
            "created_at": datetime.utcnow().isoformat() + "Z",
            
            "learning_rate": lr,
            "batch_size": batch_size,
            "epochs": epochs,
            "max_length": max_length,
        }
        torch.save(meta, out_path)
        print(f"Saved best model (F1_SCORE={best_f1:.4f}, AUC_SCORE={best_auc:.4f}, VAL_LOSS={best_loss:.4f}) to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the CSA detector")
    parser.add_argument("--train", required=True, help="Path to training JSONL file")
    parser.add_argument("--val", required=True, help="Path to validation JSONL file")
    parser.add_argument("--memory", required=True, help="Path to community memory .npz file")
    parser.add_argument(
        "--encoder",
        default=None,
        help=(
            "Name of HuggingFace model for embedding computation. If omitted,"
            " embeddings must be precomputed in post_emb field."
        ),
    )
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for AdamW")
    parser.add_argument("--k", type=int, default=3, help="Number of nearest communities to retrieve")
    parser.add_argument("--lambda_mmr", type=float, default=0.45, help="MMR lambda for community retrieval (relevance vs diversity)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Probability threshold for converting scores to labels")
    parser.add_argument("--dataset_id", default=None, help="Short identifier for training dataset (used in filenames and metadata)")
    parser.add_argument("--max_centroid_samples", type=int, default=-1, help="Max samples per community to use when building centroids (-1 = all)")
    parser.add_argument("--max_length", type=int, default=256, help="Max sequence length for tokenisation")
    parser.add_argument("--out", default="model.pt", help="Output path for saved model")
    parser.add_argument("--device", default=None, help="Device to use (e.g. cuda, cpu)")
    args = parser.parse_args()
    train(
        train_path=args.train,
        val_path=args.val,
        memory_path=args.memory,
        encoder_name=args.encoder,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        k=args.k,
        max_length=args.max_length,
        out_path=args.out,
        device=args.device,
        dataset_id=args.dataset_id,
        max_centroid_samples=args.max_centroid_samples,
        lambda_mmr=args.lambda_mmr,
        threshold=args.threshold,
    )


if __name__ == "__main__":
    main()
