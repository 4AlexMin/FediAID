import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from scripts.run_opensrc_reproduction import (
    EXPECTED_DATASET_NAMES,
    EXPECTED_SEEDS,
    canonical_dataset_name,
    load_summary,
    parse_args,
    validate_mean_reference,
    validate_per_seed_reference,
    validate_seed_results,
    validate_tolerance,
)
from scripts.run_lambda_sensitivity import (
    DEFAULT_LAMBDAS,
    EXPECTED_SEEDS as SENSITIVITY_SEEDS,
    TARGET_DATASETS,
    compare_sensitivity_results,
    load_expected_reference,
    load_expected_per_seed_reference,
    summarize_sensitivity_results,
    validate_dataset_results,
    validate_sensitivity_per_seed_table,
    validate_sensitivity_summary,
)


class ReproductionValidationTests(unittest.TestCase):
    def make_seed_frame(self):
        return pd.DataFrame(
            {
                "dataset": EXPECTED_DATASET_NAMES,
                "model": ["csa"] * len(EXPECTED_DATASET_NAMES),
                "k": [27] * len(EXPECTED_DATASET_NAMES),
                "loss": [0.5] * len(EXPECTED_DATASET_NAMES),
                "acc": [0.8] * len(EXPECTED_DATASET_NAMES),
                "f1": [0.8] * len(EXPECTED_DATASET_NAMES),
                "auc": [0.9] * len(EXPECTED_DATASET_NAMES),
            }
        )

    def test_raw_and_embedded_names_canonicalize(self):
        self.assertEqual(canonical_dataset_name("AIGTB_medium.jsonl"), "AIGTB_medium_emb.jsonl")
        self.assertEqual(canonical_dataset_name("AIGTB_medium_emb.jsonl"), "AIGTB_medium_emb.jsonl")

    def test_complete_seed_frame_passes(self):
        result = validate_seed_results(self.make_seed_frame(), seed=0)
        self.assertEqual(len(result), 12)
        self.assertTrue((result["seed"] == 0).all())

    def test_missing_dataset_fails(self):
        frame = self.make_seed_frame().iloc[:-1]
        with self.assertRaisesRegex(ValueError, "expected 12 dataset rows"):
            validate_seed_results(frame, seed=0)

    def test_duplicate_dataset_fails(self):
        frame = self.make_seed_frame()
        frame.loc[11, "dataset"] = frame.loc[0, "dataset"]
        with self.assertRaisesRegex(ValueError, "duplicate dataset"):
            validate_seed_results(frame, seed=0)

    def test_unknown_dataset_fails(self):
        frame = self.make_seed_frame()
        frame.loc[0, "dataset"] = "unlisted.jsonl"
        with self.assertRaisesRegex(ValueError, "Unexpected evaluation dataset"):
            validate_seed_results(frame, seed=0)

    def test_non_finite_metric_fails(self):
        frame = self.make_seed_frame()
        frame.loc[0, "f1"] = float("nan")
        with self.assertRaisesRegex(ValueError, "non-finite f1"):
            validate_seed_results(frame, seed=0)

    def test_fractional_k_fails(self):
        frame = self.make_seed_frame()
        frame["k"] = frame["k"].astype(float)
        frame.loc[0, "k"] = 27.5
        with self.assertRaisesRegex(ValueError, "finite integers"):
            validate_seed_results(frame, seed=0)

    def test_mean_reference_requires_all_datasets(self):
        reference = pd.DataFrame(
            {
                "dataset": EXPECTED_DATASET_NAMES[:-1],
                "f1_mean": [0.8] * 11,
                "f1_std": [0.01] * 11,
                "auc_mean": [0.9] * 11,
                "auc_std": [0.01] * 11,
                "runs": [5] * 11,
            }
        )
        with self.assertRaisesRegex(ValueError, "exactly one row per dataset"):
            validate_mean_reference(reference)

    def test_per_seed_reference_requires_all_pairs(self):
        rows = [
            {"dataset": dataset, "seed": seed, "f1": 0.8, "auc": 0.9}
            for dataset in EXPECTED_DATASET_NAMES
            for seed in EXPECTED_SEEDS
        ]
        reference = pd.DataFrame(rows[:-1])
        with self.assertRaisesRegex(ValueError, "dataset/seed coverage mismatch"):
            validate_per_seed_reference(reference)

    def test_tolerance_must_be_finite_and_non_negative(self):
        self.assertEqual(validate_tolerance("0.02"), 0.02)
        for tolerance in ("nan", "inf", "-inf", "-0.01"):
            with self.subTest(tolerance=tolerance):
                with patch("sys.argv", ["run_opensrc_reproduction.py", "--data-root", "data", "--tolerance", tolerance]):
                    with self.assertRaises(SystemExit) as error:
                        parse_args()
                self.assertEqual(error.exception.code, 2)

    def test_load_summary_rejects_non_finite_tolerance_first(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            missing_path = Path(temporary_dir) / "missing.csv"
            with self.assertRaisesRegex(ValueError, "finite, non-negative"):
                load_summary({}, Path(temporary_dir), missing_path, missing_path, float("nan"))


class LambdaSensitivityValidationTests(unittest.TestCase):
    def make_dataset_results(self):
        return pd.DataFrame(
            {
                "dataset": TARGET_DATASETS,
                "model": ["csa"] * len(TARGET_DATASETS),
                "k": [27] * len(TARGET_DATASETS),
                "loss": [0.5] * len(TARGET_DATASETS),
                "acc": [0.8] * len(TARGET_DATASETS),
                "f1": [0.8] * len(TARGET_DATASETS),
                "auc": [0.9] * len(TARGET_DATASETS),
            }
        )

    def make_per_seed_reference(self):
        return pd.DataFrame(
            [
                {
                    "lambda_mmr": lambda_mmr,
                    "seed": seed,
                    "dataset": dataset,
                    "f1": 0.6 + dataset_index * 0.01 + seed * 0.001,
                    "auc": 0.7 + dataset_index * 0.02 + seed * 0.002,
                }
                for lambda_mmr in DEFAULT_LAMBDAS
                for seed in sorted(SENSITIVITY_SEEDS)
                for dataset_index, dataset in enumerate(TARGET_DATASETS)
            ]
        )

    def make_summary_reference(self):
        return summarize_sensitivity_results(self.make_per_seed_reference())

    def test_all_eight_dataset_results_pass(self):
        validated = validate_dataset_results(self.make_dataset_results(), 0.5, 27)
        self.assertEqual(set(validated["dataset"]), set(TARGET_DATASETS))

    def test_duplicate_dataset_result_fails(self):
        result = self.make_dataset_results()
        result.loc[7, "dataset"] = result.loc[0, "dataset"]
        with self.assertRaisesRegex(ValueError, "duplicate dataset"):
            validate_dataset_results(result, 0.5, 27)

    def test_missing_dataset_result_fails(self):
        result = self.make_dataset_results().iloc[:-1]
        with self.assertRaisesRegex(ValueError, "exactly eight dataset rows"):
            validate_dataset_results(result, 0.5, 27)

    def test_non_finite_dataset_metric_fails(self):
        result = self.make_dataset_results()
        result.loc[0, "f1"] = float("nan")
        with self.assertRaisesRegex(ValueError, "non-finite f1"):
            validate_dataset_results(result, 0.5, 27)

    def test_all_five_seeds_and_eleven_lambdas_pass(self):
        validated = validate_sensitivity_per_seed_table(
            self.make_per_seed_reference(), "Test per-seed reference"
        )
        self.assertEqual(len(validated), 440)
        self.assertEqual(set(validated["lambda_mmr"]), set(DEFAULT_LAMBDAS))
        self.assertEqual(set(validated["seed"]), SENSITIVITY_SEEDS)
        self.assertEqual(set(validated["dataset"]), set(TARGET_DATASETS))

    def test_incomplete_lambda_seed_reference_fails(self):
        reference = self.make_per_seed_reference().iloc[:-1]
        with self.assertRaisesRegex(ValueError, "exactly 440 lambda/seed/dataset rows"):
            validate_sensitivity_per_seed_table(reference, "Test per-seed reference")

    def test_duplicate_lambda_seed_dataset_fails(self):
        reference = self.make_per_seed_reference()
        reference.loc[439, ["lambda_mmr", "seed", "dataset"]] = reference.loc[
            0, ["lambda_mmr", "seed", "dataset"]
        ]
        with self.assertRaisesRegex(ValueError, "duplicate lambda/seed/dataset"):
            validate_sensitivity_per_seed_table(reference, "Test per-seed reference")

    def test_unexpected_seed_fails(self):
        reference = self.make_per_seed_reference()
        reference.loc[0, "seed"] = 5
        with self.assertRaisesRegex(ValueError, "must cover seeds"):
            validate_sensitivity_per_seed_table(reference, "Test per-seed reference")

    def test_non_finite_per_seed_metric_fails(self):
        reference = self.make_per_seed_reference()
        reference.loc[0, "auc"] = float("nan")
        with self.assertRaisesRegex(ValueError, "non-finite auc"):
            validate_sensitivity_per_seed_table(reference, "Test per-seed reference")

    def test_aggregate_requires_all_lambda_settings_and_five_runs(self):
        reference = self.make_summary_reference().iloc[:-1]
        with self.assertRaisesRegex(ValueError, "exactly 11 lambda settings"):
            validate_sensitivity_summary(reference, "Test summary")

    def test_aggregate_rejects_non_finite_and_duplicate_values(self):
        reference = self.make_summary_reference()
        reference.loc[0, "f1_std"] = float("nan")
        with self.assertRaisesRegex(ValueError, "non-finite f1_std"):
            validate_sensitivity_summary(reference, "Test summary")

        reference = self.make_summary_reference()
        reference.loc[10, "lambda_mmr"] = 0.0
        with self.assertRaisesRegex(ValueError, "duplicate lambda_mmr"):
            validate_sensitivity_summary(reference, "Test summary")

    def test_aggregate_std_is_population_std_across_platform_means(self):
        summary = self.make_summary_reference()
        row = summary.loc[summary["lambda_mmr"].eq(0.0)].iloc[0]
        platform_f1_means = [0.6 + index * 0.01 + 0.002 for index in range(len(TARGET_DATASETS))]
        expected_std = pd.Series(platform_f1_means).std(ddof=0)
        self.assertAlmostEqual(row["f1_std"], expected_std)
        self.assertEqual(row["datasets"], len(TARGET_DATASETS))
        self.assertEqual(row["seeds"], len(SENSITIVITY_SEEDS))

    def test_missing_reference_file_fails(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            missing_path = Path(temporary_dir) / "missing.csv"
            with self.assertRaisesRegex(FileNotFoundError, "summary is missing"):
                load_expected_reference(missing_path)
            with self.assertRaisesRegex(FileNotFoundError, "per-seed.*reference is missing"):
                load_expected_per_seed_reference(missing_path)

    def test_sensitivity_summary_matches_reference(self):
        per_seed = self.make_per_seed_reference()
        summary = self.make_summary_reference()
        compare_sensitivity_results(per_seed, per_seed.copy(), summary, 0.02)

    def test_per_seed_mismatch_fails(self):
        actual = self.make_per_seed_reference()
        expected = actual.copy()
        expected.loc[0, "f1"] = 0.7
        with self.assertRaisesRegex(RuntimeError, "Per-seed f1 differs"):
            compare_sensitivity_results(actual, expected, self.make_summary_reference(), 0.02)

    def test_aggregate_mismatch_fails(self):
        per_seed = self.make_per_seed_reference()
        expected_summary = self.make_summary_reference()
        expected_summary.loc[0, "f1_mean"] = 0.7
        with self.assertRaisesRegex(RuntimeError, "Aggregate f1_mean differs"):
            compare_sensitivity_results(
                per_seed, per_seed.copy(), expected_summary, 0.02
            )


if __name__ == "__main__":
    unittest.main()
