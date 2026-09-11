from __future__ import annotations

import pickle
import sys
import tempfile
import unittest
from pathlib import Path

import lmdb
import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from build_esm_lmdb import (
    build_companion_dataframe,
    build_records,
    normalize_sequence,
    parse_args as parse_feature_builder_args,
    summarize_duplicates,
    write_lmdb,
)
from configs import parse_configuration
from datasets.proessenllm_dataset import prepare_metadata, split_metadata
from evaluation.leakage import audit_splits
from evaluation.metrics import distribution_metrics
from models import ProEssenLLM
from samplers import SpeciesLabelBalancedBatchSampler


class ConfigurationTests(unittest.TestCase):
    def test_frozen_feature_defaults_and_removed_options(self):
        args = parse_configuration(["--mode", "within_species"])
        self.assertEqual(args.max_length, 1000)
        for removed_option in (
            "encoder_mode",
            "esm_model_path",
            "esm_learning_rate",
            "head_learning_rate",
            "gradient_checkpointing",
            "lora_rank",
        ):
            self.assertFalse(hasattr(args, removed_option))


class FeatureBuilderTests(unittest.TestCase):
    def test_defaults_match_training_and_companion_keys_match_lmdb(self):
        args = parse_feature_builder_args(
            ["--data_path", "proteins.csv", "--output_lmdb", "features.lmdb"]
        )
        self.assertEqual(args.truncation_seq_length, 1000)
        table = pd.DataFrame(
            {
                "ID": ["protein_a", "protein_b"],
                "group": ["species_a", "species_b"],
                "essential": [1, 0],
                "sequence": ["ACD*", "***"],
            },
            index=[10, 11],
        )
        records, statistics, kept_indices = build_records(
            table, args, set("ACDEFGHIKLMNPQRSTVWYBXZUO")
        )
        companion = build_companion_dataframe(table, kept_indices, args)
        self.assertEqual(statistics["usable_rows"], 1)
        self.assertEqual(statistics["skipped_rows"], 1)
        self.assertEqual(companion["lmdb_key"].tolist(), ["10"])
        self.assertEqual(next(iter(records.values()))[0]["key"], "10")

    def test_writer_produces_training_compatible_metadata(self):
        class FakeAlphabet:
            @staticmethod
            def get_batch_converter(truncation_seq_length):
                def convert(batch):
                    labels, sequences = zip(*batch)
                    retained = [sequence[:truncation_seq_length] for sequence in sequences]
                    token_count = max(map(len, retained)) + 2
                    tokens = torch.zeros((len(batch), token_count), dtype=torch.long)
                    return list(labels), retained, tokens

                return convert

        class FakeModel:
            @staticmethod
            def __call__(tokens, repr_layers, return_contacts):
                del return_contacts
                shape = (tokens.size(0), tokens.size(1), 4)
                representation = torch.arange(
                    int(np.prod(shape)), dtype=torch.float32
                ).reshape(shape)
                return {"representations": {repr_layers[0]: representation}}

        args = parse_feature_builder_args(
            ["--data_path", "unused.csv", "--output_lmdb", "unused.lmdb"]
        )
        args.map_size_gb = 0.01
        args.commit_every = 1
        table = pd.DataFrame(
            {
                "ID": ["protein_a"],
                "group": ["species_a"],
                "essential": [1],
                "sequence": ["ACD"],
            },
            index=[10],
        )
        records, statistics, _ = build_records(
            table, args, set("ACDEFGHIKLMNPQRSTVWYBXZUO")
        )
        duplicate_summary = summarize_duplicates(records, statistics["usable_rows"])
        with tempfile.TemporaryDirectory(prefix="proessenllm_builder_") as temporary:
            output_path = Path(temporary) / "features.lmdb"
            count = write_lmdb(
                output_lmdb=output_path,
                sequence_to_records=records,
                stats=statistics,
                duplicate_summary=duplicate_summary,
                model=FakeModel(),
                alphabet=FakeAlphabet(),
                device=torch.device("cpu"),
                resolved_layer=1,
                feature_length=4,
                args=args,
            )
            environment = lmdb.open(
                str(output_path), subdir=False, readonly=True, lock=False
            )
            try:
                with environment.begin(write=False) as transaction:
                    metadata = pickle.loads(transaction.get(b"__metadata__"))
                    sample = pickle.loads(transaction.get(b"10"))
            finally:
                environment.close()
            self.assertEqual(count, 1)
            self.assertEqual(metadata["feature_length"], 4)
            self.assertEqual(sample["feature"].shape, (3, 4))

    def test_sequence_normalization_reports_changes(self):
        normalized, changed, invalid, stop_stats = normalize_sequence(
            " acD?* ", set("ACDX"), invalid_policy="mask", internal_stop_policy="mask"
        )
        self.assertEqual(normalized, "ACDX")
        self.assertTrue(changed)
        self.assertEqual(dict(invalid), {"?": 1})
        self.assertEqual(stop_stats["terminal_stop_removed"], 1)


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
