from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.proessenllm_dataset import prepare_metadata, split_metadata
from evaluation.leakage import audit_splits
from evaluation.metrics import distribution_metrics
from models import ProEssenLLM
from samplers import SpeciesLabelBalancedBatchSampler


def synthetic_table(species_count: int = 6, samples_per_label: int = 6) -> pd.DataFrame:
    records = []
    index = []
    for species in range(species_count):
        for label in (0, 1):
            for sample in range(samples_per_label):
                index.append(f"s{species}_l{label}_{sample}")
                records.append({"essential": label, "group": str(species)})
    return pd.DataFrame(records, index=index)


class SplitTests(unittest.TestCase):
    def setUp(self):
        table = synthetic_table()
        table["_sample_index"] = table.index.astype(str)
        table["_lmdb_key"] = table.index.astype(str)
        table["_species_id"] = table["group"].astype(str)
        table["_label"] = table["essential"].astype(int)
        codes = {value: index for index, value in enumerate(sorted(table["group"].unique()))}
        table["_species_code"] = table["group"].map(codes)
        self.table = table

    def test_within_species_is_sample_disjoint(self):
        splits = split_metadata(
            self.table,
            "within_species",
            test_ratio=0.2,
            validation_ratio=0.2,
            min_samples_per_species=1,
            random_seed=11,
        )
        audit = audit_splits(
            "within_species", splits.frames(), (), ProEssenLLM.MODEL_INPUT_KEYS
        )
        self.assertEqual(audit["status"], "passed")
        self.assertEqual(set(splits.train["_species_id"]), set(self.table["_species_id"]))

    def test_holdout_is_species_disjoint(self):
        splits = split_metadata(
            self.table,
            "species_holdout",
            validation_species=["4"],
            target_species=["5"],
            min_samples_per_species=1,
            random_seed=11,
        )
        audit = audit_splits(
            "species_holdout", splits.frames(), ("5",), ProEssenLLM.MODEL_INPUT_KEYS
        )
        self.assertTrue(audit["target_absent_from_train"])
        self.assertTrue(audit["target_absent_from_validation"])
        self.assertEqual(set(splits.test["_species_id"]), {"5"})


class SamplerTests(unittest.TestCase):
    def test_batch_contains_distinct_species_and_balanced_labels(self):
        species = []
        labels = []
        for species_id in range(4):
            species.extend([str(species_id)] * 8)
            labels.extend([0] * 4 + [1] * 4)
        sampler = SpeciesLabelBalancedBatchSampler(
            labels,
            species,
            species_per_batch=3,
            samples_per_species=4,
            positive_fraction=0.5,
            seed=7,
        )
        first_batch = next(iter(sampler))
        batch_species = np.asarray(species)[first_batch]
        batch_labels = np.asarray(labels)[first_batch]
        self.assertEqual(len(np.unique(batch_species)), 3)
        for species_id in np.unique(batch_species):
            values = batch_labels[batch_species == species_id]
            self.assertEqual(int(np.sum(values == 1)), 2)
            self.assertEqual(int(np.sum(values == 0)), 2)


class MetricTests(unittest.TestCase):
    def test_distribution_reports_true_worst_species_auc(self):
        labels = [0, 1, 0, 1, 0, 1]
        scores = [0.1, 0.9, 0.2, 0.8, 0.8, 0.2]
        species = ["a", "a", "b", "b", "c", "c"]
        summary, _ = distribution_metrics(labels, scores, species, threshold=0.5)
        self.assertAlmostEqual(summary["worst_species_auc"], 0.0)
        self.assertAlmostEqual(summary["macro_auc"], 2.0 / 3.0)


if __name__ == "__main__":
    unittest.main()
