#!/usr/bin/env python3
"""Run the released FediAID checkpoint on five seeds of 12 external datasets."""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
from pathlib import Path


EXPECTED_DATASETS = 12
EXPECTED_DATASET_NAMES = (
    "AIGTB_medium_emb.jsonl",
    "AIGTB_quora_emb.jsonl",
    "AIGTB_reddit_emb.jsonl",
    "deepfake_emb.jsonl",
    "fox8_emb.jsonl",
    "m4_emb.jsonl",
    "multisocial_discord_emb.jsonl",
    "multisocial_gab_emb.jsonl",
    "multisocial_telegram_emb.jsonl",
    "multisocial_twitter_emb.jsonl",
    "multisocial_whatsapp_emb.jsonl",
    "tweepfake_emb.jsonl",
)
EXPECTED_DATASET_SET = frozenset(EXPECTED_DATASET_NAMES)
EXPECTED_SEEDS = frozenset({0, 1, 2, 3, 4})
EXPECTED_K = 27
EXPECTED_LAMBDA_MMR = 0.45
DEFAULT_DATA_TEMPLATE = "opensrc_platforms_seed{seed}_emb"
DEFAULT_ENCODER = "microsoft/deberta-v3-base"
DEFAULT_ENCODER_REVISION = "8ccc9b6f36199bec6961081d44eb72fb3f7353f3"


def canonical_dataset_name(filename: str) -> str:
    """Map either raw or encoded input names to the canonical expected name."""
    name = Path(str(filename)).name
    if name in EXPECTED_DATASET_SET:
        return name
    path = Path(name)
    encoded_name = f"{path.stem}_emb{path.suffix}"
    if encoded_name in EXPECTED_DATASET_SET:
        return encoded_name
    raise ValueError(f"Unexpected evaluation dataset filename: {name}")


def validate_seed_results(frame, seed: int, k: int = EXPECTED_K):
    required_columns = {"dataset", "model", "k", "loss", "acc", "f1", "auc"}
    missing_columns = required_columns - set(frame.columns)
    if missing_columns:
        raise ValueError(f"Seed {seed} result is missing columns: {sorted(missing_columns)}")
    if len(frame) != EXPECTED_DATASETS:
        raise ValueError(f"Seed {seed}: expected {EXPECTED_DATASETS} dataset rows, got {len(frame)}")

    frame = frame.copy()
    try:
        frame["dataset"] = frame["dataset"].map(canonical_dataset_name)
    except ValueError as error:
        raise ValueError(f"Seed {seed}: {error}") from error

    duplicates = sorted(frame.loc[frame["dataset"].duplicated(keep=False), "dataset"].unique())
    if duplicates:
        raise ValueError(f"Seed {seed}: duplicate dataset rows: {duplicates}")
    actual_datasets = set(frame["dataset"])
    if actual_datasets != EXPECTED_DATASET_SET:
        missing = sorted(EXPECTED_DATASET_SET - actual_datasets)
        extra = sorted(actual_datasets - EXPECTED_DATASET_SET)
        raise ValueError(f"Seed {seed}: dataset coverage mismatch; missing={missing}, extra={extra}")
    if set(frame["model"]) != {"csa"}:
        raise ValueError(f"Seed {seed}: expected only CSA results, got {sorted(map(str, set(frame['model'])))}")
    try:
        k_values = frame["k"].astype(float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Seed {seed}: invalid k values") from error
    if not all(math.isfinite(value) and value.is_integer() for value in k_values):
        raise ValueError(f"Seed {seed}: k values must be finite integers")
    observed_k = set(k_values.astype(int))
    if observed_k != {k}:
        raise ValueError(f"Seed {seed}: expected only k={k}, got {sorted(observed_k)}")

    for column in ("loss", "acc", "f1", "auc"):
        try:
            frame[column] = frame[column].astype(float)
        except (TypeError, ValueError) as error:
            raise ValueError(f"Seed {seed}: non-numeric {column} value") from error
        if not all(math.isfinite(value) for value in frame[column]):
            raise ValueError(f"Seed {seed}: non-finite {column} value")
    for column in ("acc", "f1", "auc"):
        if not frame[column].between(0.0, 1.0).all():
            raise ValueError(f"Seed {seed}: {column} values must be within [0, 1]")
    if (frame["loss"] < 0.0).any():
        raise ValueError(f"Seed {seed}: loss values must be non-negative")
    frame["seed"] = seed
    return frame


def validate_expected_table(expected, columns: set[str], name: str) -> None:
    missing_columns = columns - set(expected.columns)
    if missing_columns:
        raise ValueError(f"{name} is missing columns: {sorted(missing_columns)}")
    if expected.empty:
        raise ValueError(f"{name} is empty")


def validate_mean_reference(expected):
    required = {"dataset", "f1_mean", "f1_std", "auc_mean", "auc_std", "runs"}
    validate_expected_table(expected, required, "Expected mean metrics")
    names = list(expected["dataset"])
    if len(names) != EXPECTED_DATASETS or len(set(names)) != EXPECTED_DATASETS:
        raise ValueError("Expected mean metrics must contain exactly one row per dataset")
    if set(names) != EXPECTED_DATASET_SET:
        raise ValueError(
            "Expected mean dataset set mismatch; "
            f"missing={sorted(EXPECTED_DATASET_SET - set(names))}, "
            f"extra={sorted(set(names) - EXPECTED_DATASET_SET)}"
        )
    for column in ("f1_mean", "f1_std", "auc_mean", "auc_std"):
        values = expected[column].astype(float)
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"Expected mean metrics contain non-finite {column} values")
    runs = expected["runs"].astype(float)
    if not all(math.isfinite(value) and value.is_integer() for value in runs) or set(runs.astype(int)) != {5}:
        raise ValueError("Expected mean metrics must record five runs per dataset")


def validate_per_seed_reference(expected):
    required = {"dataset", "seed", "f1", "auc"}
    validate_expected_table(expected, required, "Expected per-seed metrics")
    expected_pairs = {(dataset, seed) for dataset in EXPECTED_DATASET_NAMES for seed in EXPECTED_SEEDS}
    try:
        seed_values = expected["seed"].astype(float)
    except (TypeError, ValueError) as error:
        raise ValueError("Expected per-seed metrics contain invalid seed values") from error
    if not all(math.isfinite(value) and value.is_integer() for value in seed_values):
        raise ValueError("Expected per-seed values must be finite integers")
    pairs = list(zip(expected["dataset"], seed_values.astype(int)))
    if len(pairs) != len(set(pairs)):
        raise ValueError("Expected per-seed metrics contain duplicate dataset/seed rows")
    if set(pairs) != expected_pairs:
        raise ValueError(
            "Expected per-seed dataset/seed coverage mismatch; "
            f"missing={len(expected_pairs - set(pairs))}, extra={len(set(pairs) - expected_pairs)}"
        )
    for column in ("f1", "auc"):
        values = expected[column].astype(float)
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"Expected per-seed metrics contain non-finite {column} values")
        if not values.between(0.0, 1.0).all():
            raise ValueError(f"Expected per-seed {column} values must be in [0, 1]")


def validate_tolerance(tolerance: float) -> float:
    try:
        value = float(tolerance)
    except (TypeError, ValueError) as error:
        raise ValueError("Tolerance must be a finite, non-negative number") from error
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("Tolerance must be a finite, non-negative number")
    return value


def parse_tolerance(value: str) -> float:
    try:
        return validate_tolerance(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


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
    parser.add_argument(
        "--encoder-revision",
        default=DEFAULT_ENCODER_REVISION,
        help="Pinned Hugging Face snapshot used when text must be encoded on the fly.",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument(
        "--expected",
        type=Path,
        default=repo_root / "expected" / "opensrc_metrics.csv",
        help="Reference metrics used for a tolerance check after evaluation.",
    )
    parser.add_argument(
        "--expected-per-seed",
        type=Path,
        default=repo_root / "expected" / "opensrc_metrics_per_seed.csv",
        help="Per-dataset, per-seed F1/AUC references used for the tolerance check.",
    )
    parser.add_argument("--tolerance", type=parse_tolerance, default=0.02)
    parser.add_argument("--no-check-expected", action="store_true")
    return parser.parse_args()


def load_summary(
    result_paths: dict[int, Path],
    out_dir: Path,
    expected_path: Path,
    expected_per_seed_path: Path,
    tolerance: float,
) -> None:
    import pandas as pd

    tolerance = validate_tolerance(tolerance)
    if set(result_paths) != EXPECTED_SEEDS:
        raise ValueError(f"Expected results for seeds {sorted(EXPECTED_SEEDS)}, got {sorted(result_paths)}")
    if not expected_path.is_file():
        raise FileNotFoundError(f"Expected mean-metrics reference is missing: {expected_path}")
    if not expected_per_seed_path.is_file():
        raise FileNotFoundError(f"Expected per-seed reference is missing: {expected_per_seed_path}")

    out_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for seed in sorted(EXPECTED_SEEDS):
        result_path = result_paths[seed]
        if not result_path.is_file():
            raise FileNotFoundError(f"Seed {seed} result workbook is missing: {result_path}")
        frame = pd.read_excel(result_path)
        frame = validate_seed_results(frame, seed, k=EXPECTED_K)
        frames.append(frame)

    all_results = pd.concat(frames, ignore_index=True)
    actual_pairs = list(zip(all_results["dataset"], all_results["seed"]))
    required_pairs = {(dataset, seed) for dataset in EXPECTED_DATASET_NAMES for seed in EXPECTED_SEEDS}
    if len(actual_pairs) != len(set(actual_pairs)) or set(actual_pairs) != required_pairs:
        raise ValueError("Combined results do not contain each expected dataset/seed pair exactly once")

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
    validate_mean_reference(pd.read_csv(expected_path))
    per_seed_expected = pd.read_csv(expected_per_seed_path)
    validate_per_seed_reference(per_seed_expected)

    actual_per_seed = all_results[["dataset", "seed", "f1", "auc"]]
    merged_seed = actual_per_seed.merge(
        per_seed_expected,
        on=["dataset", "seed"],
        how="outer",
        suffixes=("", "_expected"),
        indicator=True,
        validate="one_to_one",
    )
    if not merged_seed["_merge"].eq("both").all():
        raise ValueError("Per-seed result/reference dataset coverage is incomplete")
    merged_seed.drop(columns="_merge", inplace=True)

    expected = pd.read_csv(expected_path)
    merged_mean = summary.merge(
        expected,
        on="dataset",
        how="outer",
        suffixes=("", "_expected"),
        indicator=True,
        validate="one_to_one",
    )
    if not merged_mean["_merge"].eq("both").all():
        raise ValueError("Aggregate result/reference dataset coverage is incomplete")
    merged_mean.drop(columns="_merge", inplace=True)

    for metric in ("f1", "auc"):
        difference = (merged_seed[metric] - merged_seed[f"{metric}_expected"]).abs()
        if difference.isna().any() or not all(math.isfinite(value) for value in difference):
            raise ValueError(f"Per-seed {metric} comparison contains non-finite values")
        if difference.max() > tolerance:
            index = difference.idxmax()
            row = merged_seed.loc[index]
            raise RuntimeError(
                f"Per-seed {metric} differs by {difference.loc[index]:.4f} for "
                f"{row['dataset']} seed {row['seed']}; tolerance is {tolerance:.4f}"
            )

    for metric in ("f1_mean", "auc_mean"):
        difference = (merged_mean[metric] - merged_mean[f"{metric}_expected"]).abs()
        if difference.isna().any() or not all(math.isfinite(value) for value in difference):
            raise ValueError(f"Aggregate {metric} comparison contains non-finite values")
        if difference.max() > tolerance:
            index = difference.idxmax()
            row = merged_mean.loc[index]
            raise RuntimeError(
                f"{metric} differs by {difference.loc[index]:.4f} on {row['dataset']}; "
                f"tolerance is {tolerance:.4f}"
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
        raise FileNotFoundError(f"Expected mean-metrics reference is missing: {expected_path}")
    print(f"Reference mean and per-seed checks: PASS (tolerance={tolerance:.3f})")


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
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Seed list contains duplicates")
    if any(seed not in EXPECTED_SEEDS for seed in args.seeds):
        raise ValueError(f"Supported reproduction seeds are {sorted(EXPECTED_SEEDS)}")
    out_dir.mkdir(parents=True, exist_ok=True)

    result_paths: dict[int, Path] = {}
    for seed in args.seeds:
        seed_dir = data_root / args.data_template.format(seed=seed)
        datasets = sorted(seed_dir.glob("*.jsonl"))
        if len(datasets) != EXPECTED_DATASETS:
            raise RuntimeError(
                f"{seed_dir} must contain {EXPECTED_DATASETS} JSONL datasets; found {len(datasets)}"
            )
        canonical_names = [canonical_dataset_name(path.name) for path in datasets]
        if len(set(canonical_names)) != EXPECTED_DATASETS or set(canonical_names) != EXPECTED_DATASET_SET:
            raise RuntimeError(
                f"{seed_dir} does not contain the exact expected datasets; "
                f"found={sorted(canonical_names)}"
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
        if args.encoder_revision:
            command.extend(["--encoder_revision", args.encoder_revision])
        if args.device:
            command.extend(["--device", args.device])
        print(f"Running seed {seed}: {seed_dir}")
        subprocess.run(command, cwd=repo_root, check=True)
        result_paths[seed] = output

    complete_seed_set = set(args.seeds) == EXPECTED_SEEDS and len(args.seeds) == len(EXPECTED_SEEDS)
    if not args.no_check_expected and complete_seed_set:
        if args.k != EXPECTED_K or not math.isclose(args.lambda_mmr, EXPECTED_LAMBDA_MMR):
            raise ValueError(
                f"Expected reference metrics apply only to k={EXPECTED_K}, "
                f"lambda_mmr={EXPECTED_LAMBDA_MMR}; pass --no-check-expected for another setting"
            )
        load_summary(
            result_paths,
            out_dir,
            args.expected.expanduser().resolve(),
            args.expected_per_seed.expanduser().resolve(),
            args.tolerance,
        )
    else:
        reason = "--no-check-expected" if args.no_check_expected else "partial seed set"
        print(f"Reference metric check: skipped ({reason})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
