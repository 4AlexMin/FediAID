#!/usr/bin/env python3
"""Optionally reproduce the eight-platform MMR lambda sensitivity summary."""

from __future__ import annotations

import argparse
import math
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
TARGET_DATASET_SET = frozenset(TARGET_DATASETS)
DEFAULT_LAMBDAS = tuple(round(index / 10, 1) for index in range(11))


def validate_sensitivity_table(table: pd.DataFrame, name: str) -> pd.DataFrame:
    required = {"lambda_mmr", "f1_mean", "auc_mean"}
    missing_columns = required - set(table.columns)
    if missing_columns:
        raise ValueError(f"{name} is missing columns: {sorted(missing_columns)}")
    if len(table) != len(DEFAULT_LAMBDAS):
        raise ValueError(f"{name} must contain exactly 11 lambda settings")

    validated = table.copy()
    for column in ("lambda_mmr", "f1_mean", "auc_mean"):
        try:
            validated[column] = pd.to_numeric(validated[column], errors="raise").astype(float)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{name} contains a non-numeric {column} value") from error
        if not all(math.isfinite(value) for value in validated[column]):
            raise ValueError(f"{name} contains non-finite {column} values")

    lambda_values = validated["lambda_mmr"].tolist()
    if len(set(lambda_values)) != len(DEFAULT_LAMBDAS):
        raise ValueError(f"{name} contains duplicate lambda_mmr rows")
    if set(lambda_values) != set(DEFAULT_LAMBDAS):
        raise ValueError(f"{name} must cover lambda_mmr values 0.0 through 1.0 in 0.1 increments")
    for column in ("f1_mean", "auc_mean"):
        if not validated[column].between(0.0, 1.0).all():
            raise ValueError(f"{name} {column} values must be within [0, 1]")
    return validated


def validate_dataset_results(result: pd.DataFrame, lambda_mmr: float, k: int) -> pd.DataFrame:
    prefix = f"lambda_mmr={lambda_mmr:.1f}: "
    required = {"dataset", "model", "k", "loss", "acc", "f1", "auc"}
    missing_columns = required - set(result.columns)
    if missing_columns:
        raise ValueError(f"{prefix}result is missing columns: {sorted(missing_columns)}")
    if len(result) != len(TARGET_DATASETS):
        raise ValueError(f"{prefix}expected exactly eight dataset rows, got {len(result)}")

    validated = result.copy()
    validated["dataset"] = validated["dataset"].map(lambda value: Path(str(value)).name)
    dataset_names = validated["dataset"].tolist()
    if len(set(dataset_names)) != len(TARGET_DATASETS):
        raise ValueError(f"{prefix}result contains duplicate dataset rows")
    if set(dataset_names) != TARGET_DATASET_SET:
        raise ValueError(f"{prefix}result does not contain exactly the eight target datasets")
    if set(validated["model"]) != {"csa"}:
        raise ValueError(f"{prefix}expected only CSA result rows")

    try:
        k_values = pd.to_numeric(validated["k"], errors="raise").astype(float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{prefix}result contains invalid k values") from error
    if not all(math.isfinite(value) and value.is_integer() for value in k_values):
        raise ValueError(f"{prefix}result k values must be finite integers")
    validated["k"] = k_values.astype(int)
    if set(validated["k"]) != {k}:
        raise ValueError(f"{prefix}expected only k={k} result rows")

    for column in ("loss", "acc", "f1", "auc"):
        try:
            validated[column] = pd.to_numeric(validated[column], errors="raise").astype(float)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{prefix}result contains a non-numeric {column} value") from error
        if not all(math.isfinite(value) for value in validated[column]):
            raise ValueError(f"{prefix}result contains non-finite {column} values")
    if (validated["loss"] < 0.0).any():
        raise ValueError(f"{prefix}loss values must be non-negative")
    for column in ("acc", "f1", "auc"):
        if not validated[column].between(0.0, 1.0).all():
            raise ValueError(f"{prefix}{column} values must be within [0, 1]")
    return validated


def load_expected_reference(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Expected lambda-sensitivity reference is missing: {path}")
    return validate_sensitivity_table(pd.read_csv(path), "Expected lambda-sensitivity reference")


def compare_sensitivity_results(
    summary: pd.DataFrame,
    expected: pd.DataFrame,
    tolerance: float,
) -> None:
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("Tolerance must be finite and non-negative")
    actual = validate_sensitivity_table(summary, "Lambda-sensitivity results")
    reference = validate_sensitivity_table(expected, "Expected lambda-sensitivity reference")
    merged = actual.merge(
        reference,
        on="lambda_mmr",
        how="outer",
        suffixes=("", "_expected"),
        indicator=True,
        validate="one_to_one",
    )
    if not merged["_merge"].eq("both").all():
        raise ValueError("Lambda-sensitivity result/reference coverage is incomplete")

    for metric in ("f1_mean", "auc_mean"):
        difference = (merged[metric] - merged[f"{metric}_expected"]).abs()
        if difference.isna().any() or not all(math.isfinite(value) for value in difference):
            raise ValueError(f"Lambda-sensitivity {metric} comparison contains non-finite values")
        if difference.max() > tolerance:
            index = difference.idxmax()
            raise RuntimeError(
                f"{metric} differs by {difference.loc[index]:.4f} at "
                f"lambda_mmr={merged.loc[index, 'lambda_mmr']:.1f}; "
                f"allowed tolerance is {tolerance:.4f}"
            )


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
    expected = load_expected_reference(args.expected.expanduser().resolve())
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
            result = validate_dataset_results(pd.read_excel(result_path), lambda_mmr, args.k)
            rows.append({
                "lambda_mmr": lambda_mmr,
                "f1_mean": result["f1"].mean(),
                "auc_mean": result["auc"].mean(),
            })

    summary = pd.DataFrame(rows)
    summary = validate_sensitivity_table(summary, "Lambda-sensitivity results")
    compare_sensitivity_results(summary, expected, args.tolerance)
    summary.to_csv(output, index=False, float_format="%.9f")
    print(f"Wrote {output}")
    print(f"Sensitivity expected-results check: PASS (tolerance={args.tolerance:.3f})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
