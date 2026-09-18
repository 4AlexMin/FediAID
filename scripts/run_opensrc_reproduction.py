#!/usr/bin/env python3
"""Run the released FediAID checkpoint on five seeds of 12 external datasets."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


EXPECTED_DATASETS = 12
DEFAULT_DATA_TEMPLATE = "opensrc_platforms_seed{seed}_emb"
DEFAULT_ENCODER = "microsoft/deberta-v3-base"


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Parent directory containing opensrc_platforms_seed{0..4}_emb.",
    )
    parser.add_argument(
        "--data-template",
        default=DEFAULT_DATA_TEMPLATE,
        help="Directory template relative to --data-root; {seed} is replaced.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=repo_root / "checkpoints" / "model_k27.pt",
    )
    parser.add_argument(
        "--memory",
        type=Path,
        default=repo_root / "checkpoints" / "memory_bank_complete_emb.npz",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=repo_root / "results" / "opensrc_reproduction",
    )
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--device", default=None, help="For example: cuda:0 or cpu.")
    parser.add_argument("--k", type=int, default=27)
    parser.add_argument(
        "--lambda-mmr",
        type=float,
        default=0.45,
        help="MMR lambda used by the released evaluation protocol.",
    )
    parser.add_argument("--encoder", default=DEFAULT_ENCODER)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument(
        "--expected",
        type=Path,
        default=repo_root / "expected" / "opensrc_metrics.csv",
        help="Reference metrics used for a tolerance check after evaluation.",
    )
    parser.add_argument("--tolerance", type=float, default=0.02)
    parser.add_argument("--no-check-expected", action="store_true")
    return parser.parse_args()


def load_summary(result_paths: list[Path], out_dir: Path, expected_path: Path, tolerance: float) -> None:
    import pandas as pd

    out_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for seed, result_path in enumerate(result_paths):
        frame = pd.read_excel(result_path)
        if set(frame["model"]) != {"csa"} or set(frame["k"]) != {27}:
            raise RuntimeError(f"Unexpected model or k in {result_path}")
        if len(frame) != EXPECTED_DATASETS:
            raise RuntimeError(f"Expected {EXPECTED_DATASETS} rows in {result_path}, got {len(frame)}")
        frame["seed"] = seed
        frames.append(frame)

    all_results = pd.concat(frames, ignore_index=True)
    summary = (
        all_results.groupby("dataset", sort=True)
        .agg(
            f1_mean=("f1", "mean"),
            f1_std=("f1", "std"),
            auc_mean=("auc", "mean"),
            auc_std=("auc", "std"),
            runs=("seed", "count"),
        )
        .reset_index()
    )
    summary_path = out_dir / "summary.csv"
    summary.to_csv(summary_path, index=False)

    main_platforms = {
        "AIGTB_medium_emb.jsonl",
        "AIGTB_quora_emb.jsonl",
        "AIGTB_reddit_emb.jsonl",
        "multisocial_discord_emb.jsonl",
        "multisocial_gab_emb.jsonl",
        "multisocial_telegram_emb.jsonl",
        "multisocial_twitter_emb.jsonl",
        "multisocial_whatsapp_emb.jsonl",
    }
    main = all_results[all_results["dataset"].isin(main_platforms)]
    dataset_means = main.groupby("dataset")[["f1", "auc"]].mean()
    print(f"Wrote {summary_path}")
    print(
        "Eight-platform headline: "
        f"F1={dataset_means['f1'].mean():.3f} +/- {dataset_means['f1'].std(ddof=0):.3f}; "
        f"AUC={dataset_means['auc'].mean():.3f} +/- {dataset_means['auc'].std(ddof=0):.3f}"
    )

    if not expected_path.exists():
        print(f"Expected metrics not found; skipped comparison: {expected_path}")
        return
    expected = pd.read_csv(expected_path)
    merged = summary.merge(expected, on="dataset", suffixes=("", "_expected"), validate="one_to_one")
    for metric in ("f1_mean", "auc_mean"):
        difference = (merged[metric] - merged[f"{metric}_expected"]).abs()
        if difference.max() > tolerance:
            worst = merged.loc[difference.idxmax(), "dataset"]
            raise RuntimeError(
                f"{metric} differs from reference by {difference.max():.4f} on {worst}; "
                f"allowed tolerance is {tolerance:.4f}"
            )
    print(f"Reference metric check: PASS (tolerance={tolerance:.3f})")


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    data_root = args.data_root.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    memory = args.memory.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if not memory.exists():
        raise FileNotFoundError(f"Memory bank not found: {memory}")
    out_dir.mkdir(parents=True, exist_ok=True)

    result_paths = []
    for seed in args.seeds:
        seed_dir = data_root / args.data_template.format(seed=seed)
        datasets = sorted(seed_dir.glob("*.jsonl"))
        if len(datasets) != EXPECTED_DATASETS:
            raise RuntimeError(
                f"{seed_dir} must contain {EXPECTED_DATASETS} JSONL datasets; found {len(datasets)}"
            )
        output = out_dir / f"opensrc_seed{seed}.xlsx"
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
            "--ks",
            str(args.k),
            "--lambda_mmr",
            str(args.lambda_mmr),
            "--batch_size",
            str(args.batch_size),
            "--out",
            str(output),
        ]
        if args.encoder:
            command.extend(["--encoder", args.encoder])
        if args.device:
            command.extend(["--device", args.device])
        print(f"Running seed {seed}: {seed_dir}")
        subprocess.run(command, cwd=repo_root, check=True)
        result_paths.append(output)

    complete_seed_set = set(args.seeds) == {0, 1, 2, 3, 4}
    if not args.no_check_expected and complete_seed_set:
        load_summary(result_paths, out_dir, args.expected.expanduser().resolve(), args.tolerance)
    else:
        reason = "--no-check-expected" if args.no_check_expected else "partial seed set"
        print(f"Reference metric check: skipped ({reason})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
