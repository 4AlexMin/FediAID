#!/usr/bin/env python3
"""Optionally reproduce the eight-platform MMR lambda sensitivity summary."""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

TARGET_DATASETS = (
    "AIGTB_medium_emb.jsonl",
    "AIGTB_quora_emb.jsonl",
    "AIGTB_reddit_emb.jsonl",
    "multisocial_discord_emb.jsonl",
    "multisocial_gab_emb.jsonl",
    "multisocial_telegram_emb.jsonl",
    "multisocial_twitter_emb.jsonl",
    "multisocial_whatsapp_emb.jsonl",
)
DEFAULT_LAMBDAS = tuple(round(index / 10, 1) for index in range(11))


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", type=Path, default=repo_root / "checkpoints" / "model_k27.pt")
    parser.add_argument("--memory", type=Path, default=repo_root / "checkpoints" / "memory_bank_complete_emb.npz")
    parser.add_argument("--encoder", default="microsoft/deberta-v3-base")
    parser.add_argument("--k", type=int, default=27)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", type=Path, default=repo_root / "results" / "lambda_sensitivity.csv")
    parser.add_argument("--expected", type=Path, default=repo_root / "expected" / "opensrc_lambda_sensitivity.csv")
    parser.add_argument("--tolerance", type=float, default=0.02)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    data_dir = args.data_root.expanduser().resolve() / f"opensrc_platforms_seed{args.seed}_emb"
    datasets = [data_dir / name for name in TARGET_DATASETS]
    missing = [str(path) for path in datasets if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing target-platform datasets:\n" + "\n".join(missing))

    checkpoint = args.checkpoint.expanduser().resolve()
    memory = args.memory.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []

    with tempfile.TemporaryDirectory(prefix="fediaid_lambda_sensitivity_") as temporary_dir:
        temporary_root = Path(temporary_dir)
        for lambda_mmr in DEFAULT_LAMBDAS:
            result_path = temporary_root / f"lambda_{lambda_mmr:.1f}.xlsx"
            command = [
                sys.executable,
                "-m",
                "src.eval_opensrc",
                "--datasets",
                *(str(path) for path in datasets),
                "--memory",
                str(memory),
                "--csacheck",
                str(checkpoint),
                "--encoder",
                args.encoder,
                "--ks",
                str(args.k),
                "--lambda_mmr",
                str(lambda_mmr),
                "--batch_size",
                str(args.batch_size),
                "--out",
                str(result_path),
            ]
            if args.device:
                command.extend(["--device", args.device])
            print(f"Evaluating lambda_mmr={lambda_mmr:.1f}")
            subprocess.run(command, cwd=repo_root, check=True)
            result = pd.read_excel(result_path)
            result = result[result["dataset"].isin(TARGET_DATASETS)]
            if len(result) != len(TARGET_DATASETS):
                raise RuntimeError(f"Expected {len(TARGET_DATASETS)} rows, got {len(result)}")
            rows.append({
                "lambda_mmr": lambda_mmr,
                "f1_mean": result["f1"].mean(),
                "auc_mean": result["auc"].mean(),
            })

    summary = pd.DataFrame(rows)
    summary.to_csv(output, index=False, float_format="%.9f")
    print(f"Wrote {output}")

    if args.expected.exists():
        expected = pd.read_csv(args.expected)
        merged = summary.merge(expected, on="lambda_mmr", suffixes=("", "_expected"), validate="one_to_one")
        for metric in ("f1_mean", "auc_mean"):
            difference = (merged[metric] - merged[f"{metric}_expected"]).abs()
            if difference.max() > args.tolerance:
                raise RuntimeError(
                    f"{metric} differs from expected by {difference.max():.4f}; "
                    f"allowed tolerance is {args.tolerance:.4f}"
                )
        print(f"Sensitivity expected-results check: PASS (tolerance={args.tolerance:.3f})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
