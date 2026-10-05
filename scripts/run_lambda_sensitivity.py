#!/usr/bin/env python3
"""Reproduce the eight-platform MMR lambda sensitivity across five seeds."""

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
SEEDS = (0, 1, 2, 3, 4)
EXPECTED_SEEDS = frozenset(SEEDS)
DEFAULT_LAMBDAS = tuple(round(index / 10, 1) for index in range(11))
EXPECTED_LAMBDA_SEED_DATASET_TRIPLES = frozenset(
    (lambda_mmr, seed, dataset)
    for lambda_mmr in DEFAULT_LAMBDAS
    for seed in EXPECTED_SEEDS
    for dataset in TARGET_DATASETS
)


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


def validate_sensitivity_per_seed_table(table: pd.DataFrame, name: str) -> pd.DataFrame:
    required = {"lambda_mmr", "seed", "dataset", "f1", "auc"}
    missing_columns = required - set(table.columns)
    if missing_columns:
        raise ValueError(f"{name} is missing columns: {sorted(missing_columns)}")
    if len(table) != len(EXPECTED_LAMBDA_SEED_DATASET_TRIPLES):
        raise ValueError(f"{name} must contain exactly 440 lambda/seed/dataset rows")

    validated = table.copy()
    for column in ("lambda_mmr", "seed", "f1", "auc"):
        try:
            validated[column] = pd.to_numeric(validated[column], errors="raise").astype(float)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{name} contains a non-numeric {column} value") from error
        if not all(math.isfinite(value) for value in validated[column]):
            raise ValueError(f"{name} contains non-finite {column} values")

    if not all(value.is_integer() for value in validated["seed"]):
        raise ValueError(f"{name} seed values must be finite integers")
    validated["seed"] = validated["seed"].astype(int)
    validated["dataset"] = validated["dataset"].map(lambda value: Path(str(value)).name)
    if set(validated["lambda_mmr"]) != set(DEFAULT_LAMBDAS):
        raise ValueError(f"{name} must cover lambda_mmr values 0.0 through 1.0 in 0.1 increments")
    if set(validated["seed"]) != EXPECTED_SEEDS:
        raise ValueError(f"{name} must cover seeds {sorted(EXPECTED_SEEDS)}")
    if set(validated["dataset"]) != TARGET_DATASET_SET:
        raise ValueError(f"{name} must cover exactly the eight target datasets")

    triples = list(zip(validated["lambda_mmr"], validated["seed"], validated["dataset"]))
    if len(triples) != len(set(triples)):
        raise ValueError(f"{name} contains duplicate lambda/seed/dataset rows")
    if set(triples) != EXPECTED_LAMBDA_SEED_DATASET_TRIPLES:
        raise ValueError(f"{name} lambda/seed/dataset coverage is incomplete")
    for column in ("f1", "auc"):
        if not validated[column].between(0.0, 1.0).all():
            raise ValueError(f"{name} {column} values must be within [0, 1]")
    return validated.sort_values(["lambda_mmr", "seed", "dataset"]).reset_index(drop=True)


def validate_sensitivity_summary(table: pd.DataFrame, name: str) -> pd.DataFrame:
    required = {"lambda_mmr", "datasets", "f1_mean", "f1_std", "auc_mean", "auc_std", "seeds"}
    missing_columns = required - set(table.columns)
    if missing_columns:
        raise ValueError(f"{name} is missing columns: {sorted(missing_columns)}")
    if len(table) != len(DEFAULT_LAMBDAS):
        raise ValueError(f"{name} must contain exactly 11 lambda settings")

    validated = table.copy()
    for column in ("lambda_mmr", "datasets", "f1_mean", "f1_std", "auc_mean", "auc_std", "seeds"):
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
    for column in ("f1_std", "auc_std"):
        if (validated[column] < 0.0).any():
            raise ValueError(f"{name} {column} values must be non-negative")
    for column, expected_count in (("datasets", len(TARGET_DATASETS)), ("seeds", len(SEEDS))):
        values = validated[column]
        if not all(value.is_integer() for value in values) or set(values.astype(int)) != {expected_count}:
            raise ValueError(f"{name} must record {expected_count} {column} per lambda setting")
        validated[column] = values.astype(int)
    return validated.sort_values("lambda_mmr").reset_index(drop=True)


def load_expected_reference(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Expected lambda-sensitivity summary is missing: {path}")
    return validate_sensitivity_summary(pd.read_csv(path), "Expected lambda-sensitivity summary")


def load_expected_per_seed_reference(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Expected per-seed lambda-sensitivity reference is missing: {path}")
    return validate_sensitivity_per_seed_table(
        pd.read_csv(path), "Expected per-seed lambda-sensitivity reference"
    )


def summarize_sensitivity_results(per_seed: pd.DataFrame) -> pd.DataFrame:
    validated = validate_sensitivity_per_seed_table(per_seed, "Per-seed lambda-sensitivity results")
    platform_means = (
        validated.groupby(["lambda_mmr", "dataset"], sort=True, as_index=False)
        .agg(
            f1=("f1", "mean"),
            auc=("auc", "mean"),
        )
    )
    summary = platform_means.groupby("lambda_mmr", sort=True).agg(
        datasets=("dataset", "nunique"),
        f1_mean=("f1", "mean"),
        f1_std=("f1", lambda values: values.std(ddof=0)),
        auc_mean=("auc", "mean"),
        auc_std=("auc", lambda values: values.std(ddof=0)),
    )
    summary["seeds"] = validated.groupby("lambda_mmr")["seed"].nunique()
    summary = summary.reset_index()
    return validate_sensitivity_summary(summary, "Lambda-sensitivity summary")


def compare_sensitivity_results(
    per_seed: pd.DataFrame,
    expected_per_seed: pd.DataFrame,
    expected_summary: pd.DataFrame,
    tolerance: float,
) -> None:
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("Tolerance must be finite and non-negative")
    actual_per_seed = validate_sensitivity_per_seed_table(per_seed, "Per-seed lambda-sensitivity results")
    reference_per_seed = validate_sensitivity_per_seed_table(
        expected_per_seed, "Expected per-seed lambda-sensitivity reference"
    )
    merged_per_seed = actual_per_seed.merge(
        reference_per_seed,
        on=["lambda_mmr", "seed", "dataset"],
        how="outer",
        suffixes=("", "_expected"),
        indicator=True,
        validate="one_to_one",
    )
    if not merged_per_seed["_merge"].eq("both").all():
        raise ValueError("Per-seed lambda-sensitivity result/reference coverage is incomplete")
    merged_per_seed.drop(columns="_merge", inplace=True)

    for metric in ("f1", "auc"):
        difference = (merged_per_seed[metric] - merged_per_seed[f"{metric}_expected"]).abs()
        if difference.isna().any() or not all(math.isfinite(value) for value in difference):
            raise ValueError(f"Per-seed {metric} comparison contains non-finite values")
        if difference.max() > tolerance:
            index = difference.idxmax()
            row = merged_per_seed.loc[index]
            raise RuntimeError(
                f"Per-seed {metric} differs by {difference.loc[index]:.4f} at "
                f"lambda_mmr={row['lambda_mmr']:.1f}, seed={row['seed']}; "
                f"allowed tolerance is {tolerance:.4f}"
            )

    actual_summary = summarize_sensitivity_results(actual_per_seed)
    reference_summary = validate_sensitivity_summary(
        expected_summary, "Expected lambda-sensitivity summary"
    )
    merged_summary = actual_summary.merge(
        reference_summary,
        on="lambda_mmr",
        how="outer",
        suffixes=("", "_expected"),
        indicator=True,
        validate="one_to_one",
    )
    if not merged_summary["_merge"].eq("both").all():
        raise ValueError("Aggregate lambda-sensitivity result/reference coverage is incomplete")
    merged_summary.drop(columns="_merge", inplace=True)

    for metric in ("f1_mean", "f1_std", "auc_mean", "auc_std"):
        difference = (merged_summary[metric] - merged_summary[f"{metric}_expected"]).abs()
        if difference.isna().any() or not all(math.isfinite(value) for value in difference):
            raise ValueError(f"Aggregate lambda-sensitivity {metric} comparison contains non-finite values")
        if difference.max() > tolerance:
            index = difference.idxmax()
            raise RuntimeError(
                f"Aggregate {metric} differs by {difference.loc[index]:.4f} at "
                f"lambda_mmr={merged_summary.loc[index, 'lambda_mmr']:.1f}; "
                f"allowed tolerance is {tolerance:.4f}"
            )


def parse_tolerance(value: str) -> float:
    try:
        tolerance = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("Tolerance must be finite and non-negative") from error
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise argparse.ArgumentTypeError("Tolerance must be finite and non-negative")
    return tolerance


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=repo_root / "checkpoints" / "model_k27.pt")
    parser.add_argument("--memory", type=Path, default=repo_root / "checkpoints" / "memory_bank_complete_emb.npz")
    parser.add_argument("--encoder", default="microsoft/deberta-v3-base")
    parser.add_argument("--k", type=int, default=27)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root / "results" / "lambda_sensitivity_per_seed.csv",
        help="Per-dataset, per-seed metrics for each lambda.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=repo_root / "results" / "lambda_sensitivity_summary.csv",
        help="Five-seed platform-mean and population-SD summary for each lambda.",
    )
    parser.add_argument(
        "--expected",
        type=Path,
        default=repo_root / "expected" / "opensrc_lambda_sensitivity.csv",
    )
    parser.add_argument(
        "--expected-per-seed",
        type=Path,
        default=repo_root / "expected" / "opensrc_lambda_sensitivity_per_seed.csv",
    )
    parser.add_argument("--tolerance", type=parse_tolerance, default=0.02)
    parser.add_argument("--no-check-expected", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    expected_summary = None
    expected_per_seed = None
    if not args.no_check_expected:
        expected_summary = load_expected_reference(args.expected.expanduser().resolve())
        expected_per_seed = load_expected_per_seed_reference(
            args.expected_per_seed.expanduser().resolve()
        )

    data_root = args.data_root.expanduser().resolve()
    datasets_by_seed = {}
    for seed in SEEDS:
        data_dir = data_root / f"opensrc_platforms_seed{seed}_emb"
        datasets = [data_dir / name for name in TARGET_DATASETS]
        missing = [str(path) for path in datasets if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                f"Missing target-platform datasets for seed {seed}:\n" + "\n".join(missing)
            )
        datasets_by_seed[seed] = datasets

    checkpoint = args.checkpoint.expanduser().resolve()
    memory = args.memory.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if not memory.is_file():
        raise FileNotFoundError(f"Memory bank not found: {memory}")

    output = args.output.expanduser().resolve()
    summary_output = args.summary_output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    rows = []

    with tempfile.TemporaryDirectory(prefix="fediaid_lambda_sensitivity_") as temporary_dir:
        temporary_root = Path(temporary_dir)
        for lambda_mmr in DEFAULT_LAMBDAS:
            for seed in SEEDS:
                result_path = temporary_root / f"seed_{seed}_lambda_{lambda_mmr:.1f}.xlsx"
                command = [
                    sys.executable,
                    "-m",
                    "src.eval_opensrc",
                    "--datasets",
                    *(str(path) for path in datasets_by_seed[seed]),
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
                print(f"Evaluating seed={seed}, lambda_mmr={lambda_mmr:.1f}")
                subprocess.run(command, cwd=repo_root, check=True)
                result = validate_dataset_results(pd.read_excel(result_path), lambda_mmr, args.k)
                for row in result.itertuples(index=False):
                    rows.append({
                        "lambda_mmr": lambda_mmr,
                        "seed": seed,
                        "dataset": row.dataset,
                        "f1": row.f1,
                        "auc": row.auc,
                    })

    per_seed = validate_sensitivity_per_seed_table(
        pd.DataFrame(rows), "Per-seed lambda-sensitivity results"
    )
    summary = summarize_sensitivity_results(per_seed)
    if not args.no_check_expected:
        compare_sensitivity_results(
            per_seed,
            expected_per_seed,
            expected_summary,
            args.tolerance,
        )

    per_seed.to_csv(output, index=False, float_format="%.9f")
    summary.to_csv(summary_output, index=False, float_format="%.9f")
    print(f"Wrote per-seed results to {output}")
    print(f"Wrote five-seed summary to {summary_output}")
    for row in summary.itertuples(index=False):
        print(
            f"lambda_mmr={row.lambda_mmr:.1f}: "
            f"F1={row.f1_mean:.3f} +/- {row.f1_std:.3f}; "
            f"AUC={row.auc_mean:.3f} +/- {row.auc_std:.3f}"
        )
    if args.no_check_expected:
        print("Sensitivity expected-results check: skipped (--no-check-expected)")
    else:
        print(f"Sensitivity expected-results check: PASS (tolerance={args.tolerance:.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
