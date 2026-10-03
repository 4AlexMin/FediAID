import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_opensrc_reproduction import (
    EXPECTED_DATASET_NAMES,
    EXPECTED_SEEDS,
    canonical_dataset_name,
    validate_mean_reference,
    validate_per_seed_reference,
    validate_seed_results,
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


if __name__ == "__main__":
    unittest.main()
