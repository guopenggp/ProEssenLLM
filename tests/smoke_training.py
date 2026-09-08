"""Run a one-epoch end-to-end smoke test on a synthetic LMDB."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import tempfile
from pathlib import Path

import lmdb
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs import parse_configuration
from trainers import create_trainer


def create_fixture(directory: Path) -> tuple[Path, Path]:
    records = []
    indices = []
    features = {}
    generator = np.random.default_rng(13)
    for species in range(4):
        for label in (0, 1):
            for sample in range(6):
                sample_index = f"species{species}_label{label}_{sample}"
                indices.append(sample_index)
                records.append({"essential": label, "group": str(species)})
                base = 1.0 if label else -1.0
                features[sample_index] = (
                    generator.normal(0.0, 0.2, size=(5, 4)).astype(np.float32) + base
                )
    metadata_path = directory / "metadata.pkl"
    pd.DataFrame(records, index=indices).to_pickle(metadata_path)
    lmdb_path = directory / "features.lmdb"
    environment = lmdb.open(str(lmdb_path), subdir=False, map_size=32 * 1024 * 1024)
    with environment.begin(write=True) as transaction:
        transaction.put(b"__metadata__", pickle.dumps({"feature_length": 4}))
        for sample_index, feature in features.items():
            transaction.put(
                sample_index.encode("utf-8"), pickle.dumps({"feature": feature})
            )
    environment.close()
    return metadata_path, lmdb_path


def run_smoke(mode: str) -> None:
    with tempfile.TemporaryDirectory(prefix=f"proessenllm_{mode}_") as temporary:
        root = Path(temporary)
        metadata_path, lmdb_path = create_fixture(root)
        output_path = root / "results"
        arguments = [
            "--mode",
            mode,
            "--data_path",
            str(metadata_path),
            "--feature_dir",
            str(lmdb_path),
            "--save_path",
            str(output_path),
            "--input_size",
            "4",
            "--hidden_size",
            "8",
            "--num_heads",
            "2",
            "--num_layers",
            "1",
            "--max_length",
            "6",
            "--sampling_strategy",
            "species_balanced",
            "--species_per_batch",
            "2",
            "--samples_per_species",
            "2",
            "--positive_fraction",
            "0.5",
            "--batch_size",
            "4",
            "--min_samples_per_species",
            "1",
            "--num_epochs",
            "1",
            "--early_stopping_patience",
            "2",
            "--num_workers",
            "0",
            "--device",
            "cpu",
        ]
        if mode == "species_holdout":
            arguments.extend(["--target_species", "3", "--validation_species", "2"])
        configuration = parse_configuration(arguments)
        summary = create_trainer(configuration).run()
        audit = json.loads((output_path / "leakage_audit.json").read_text(encoding="utf-8"))
        if audit["status"] != "passed":
            raise AssertionError(audit)
        if not (output_path / "best_validation_model.pt").exists():
            raise AssertionError("Checkpoint was not created")
        if not (output_path / "species_performance.csv").exists():
            raise AssertionError("Species performance CSV was not created")
        print(
            json.dumps(
                {
                    "mode": mode,
                    "selected_epoch": summary["selected_epoch"],
                    "test_auc": summary["test"]["micro_auc"],
                    "audit": audit["status"],
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["within_species", "species_holdout"], required=True)
    run_smoke(parser.parse_args().mode)
