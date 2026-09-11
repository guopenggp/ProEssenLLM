"""Metadata, leakage-safe task splits, datasets, collators, and loaders."""

from __future__ import annotations

import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import lmdb
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from samplers import SpeciesLabelBalancedBatchSampler


def normalize_species_id(value: Any) -> str:
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _species_sort_key(value: str):
    return (not value.isdigit(), int(value) if value.isdigit() else value)


def _read_table(path: str) -> pd.DataFrame:
    suffix = Path(path).suffix.lower()
    if suffix in {".pkl", ".pickle"}:
        loaded = pd.read_pickle(path)
    elif suffix == ".csv":
        loaded = pd.read_csv(path)
    elif suffix in {".tsv", ".txt"}:
        loaded = pd.read_csv(path, sep="\t")
    else:
        raise ValueError(f"Unsupported metadata table: {path}")
    if isinstance(loaded, pd.DataFrame):
        return loaded.copy()
    if isinstance(loaded, Mapping):
        return pd.DataFrame(loaded)
    raise TypeError(f"Expected a DataFrame or mapping, got {type(loaded).__name__}")


def prepare_metadata(
    data_path: str,
    label_name: str = "essential",
    group_column: str = "group",
) -> pd.DataFrame:
    table = _read_table(data_path)
    missing = [name for name in (label_name, group_column) if name not in table]
    if missing:
        raise ValueError(f"Metadata table is missing columns: {missing}")
    sample_indices = pd.Series(table.index, index=table.index).map(str)
    if sample_indices.duplicated().any():
        raise ValueError("Metadata indices must remain unique after string normalization")
    table["_sample_index"] = sample_indices
    table["_lmdb_key"] = table["lmdb_key"].astype(str) if "lmdb_key" in table else sample_indices
    if table["_lmdb_key"].duplicated().any():
        raise ValueError("LMDB keys must be unique")
    table["_species_id"] = table[group_column].map(normalize_species_id)
    try:
        table["_label"] = table[label_name].astype(int)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Column '{label_name}' cannot be converted to binary integers") from exc
    if not table["_label"].isin([0, 1]).all():
        raise ValueError(f"Column '{label_name}' must contain only 0 and 1")
    species_ids = sorted(table["_species_id"].unique().tolist(), key=_species_sort_key)
    species_codes = {species_id: code for code, species_id in enumerate(species_ids)}
    table["_species_code"] = table["_species_id"].map(species_codes).astype(int)
    return table


def _split_count(size: int, ratio: float) -> int:
    if size <= 1 or ratio <= 0:
        return 0
    return min(max(1, int(round(size * ratio))), size - 1)


def _stratified_within_species_split(
    species_table: pd.DataFrame,
    validation_ratio: float,
    test_ratio: float,
    rng: random.Random,
) -> tuple[list[str], list[str], list[str]]:
    train: list[str] = []
    validation: list[str] = []
    test: list[str] = []
    for label in (0, 1):
        indices = species_table.loc[species_table["_label"] == label, "_sample_index"].tolist()
        rng.shuffle(indices)
        test_count = _split_count(len(indices), test_ratio)
        remaining = len(indices) - test_count
        validation_count = min(
            _split_count(len(indices), validation_ratio), max(remaining - 1, 0)
        )
        test.extend(indices[:test_count])
        validation.extend(indices[test_count : test_count + validation_count])
        train.extend(indices[test_count + validation_count :])
    rng.shuffle(train)
    rng.shuffle(validation)
    rng.shuffle(test)
    return train, validation, test


def _count_summary(table: pd.DataFrame) -> dict[str, Any]:
    total = int(len(table))
    positive = int(table["_label"].sum()) if total else 0
    per_species: dict[str, dict[str, Any]] = {}
    for species_id, group in table.groupby("_species_id", sort=False):
        species_positive = int(group["_label"].sum())
        species_total = int(len(group))
        per_species[str(species_id)] = {
            "sample_number": species_total,
            "positive": species_positive,
            "negative": species_total - species_positive,
            "positive_ratio": species_positive / species_total if species_total else 0.0,
        }
    return {
        "sample_number": total,
        "positive": positive,
        "negative": total - positive,
        "positive_ratio": positive / total if total else 0.0,
        "species": sorted(table["_species_id"].unique().tolist(), key=_species_sort_key),
        "per_species": per_species,
        "sample_indices": table["_sample_index"].tolist(),
    }


@dataclass
class DataSplits:
    table: pd.DataFrame
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    excluded: pd.DataFrame
    statistics: dict[str, Any]
    mode: str
    target_species: tuple[str, ...]

    def frames(self) -> dict[str, pd.DataFrame]:
        return {
            "train": self.train,
            "validation": self.validation,
            "test": self.test,
            "excluded": self.excluded,
        }


def _validate_named_species(
    available: set[str],
    target: set[str],
    excluded: set[str],
    validation: set[str],
) -> None:
    named = {
        "target_species": target,
        "exclude_species": excluded,
        "validation_species": validation,
    }
    for name, values in named.items():
        missing = values - available
        if missing:
            raise ValueError(
                f"{name} contains species absent from data: {sorted(missing, key=_species_sort_key)}"
            )
    for left_name, left, right_name, right in (
        ("target_species", target, "exclude_species", excluded),
        ("target_species", target, "validation_species", validation),
        ("exclude_species", excluded, "validation_species", validation),
    ):
        overlap = left & right
        if overlap:
            raise ValueError(
                f"{left_name} and {right_name} overlap: {sorted(overlap, key=_species_sort_key)}"
            )


def _assert_sample_disjoint(frames: Mapping[str, pd.DataFrame]) -> None:
    names = list(frames)
    for left_index, left_name in enumerate(names):
        left_samples = set(frames[left_name]["_sample_index"])
        for right_name in names[left_index + 1 :]:
            overlap = left_samples & set(frames[right_name]["_sample_index"])
            if overlap:
                raise AssertionError(
                    f"Sample leakage between {left_name} and {right_name}: {sorted(overlap)[:5]}"
                )


def split_metadata(
    table: pd.DataFrame,
    mode: str,
    test_ratio: float = 0.1,
    validation_ratio: float = 0.1,
    validation_species_ratio: float = 0.2,
    validation_species: Sequence[Any] | None = None,
    target_species: Sequence[Any] | None = None,
    exclude_species: Sequence[Any] | None = None,
    single_class_species_policy: str = "exclude",
    min_samples_per_species: int = 1,
    min_positive_per_species: int = 1,
    min_negative_per_species: int = 1,
    random_seed: int = 42,
) -> DataSplits:
    """Create one of two intentionally distinct task splits."""
    if mode not in {"within_species", "species_holdout"}:
        raise ValueError("mode must be within_species or species_holdout")
    if single_class_species_policy not in {"exclude", "train_only", "error"}:
        raise ValueError("single_class_species_policy must be exclude, train_only, or error")
    if test_ratio < 0 or validation_ratio < 0 or test_ratio + validation_ratio >= 1:
        raise ValueError("test_ratio and validation_ratio must sum to less than 1")
    for value, name in (
        (min_samples_per_species, "min_samples_per_species"),
        (min_positive_per_species, "min_positive_per_species"),
        (min_negative_per_species, "min_negative_per_species"),
    ):
        if value < 1:
            raise ValueError(f"{name} must be at least 1")

    available = set(table["_species_id"])
    target_set = {normalize_species_id(value) for value in (target_species or [])}
    excluded_set = {normalize_species_id(value) for value in (exclude_species or [])}
    validation_set = {normalize_species_id(value) for value in (validation_species or [])}
    if mode == "within_species" and (target_set or validation_set):
        raise ValueError(
            "within_species does not accept target_species or validation_species; "
            "use species_holdout for species-level isolation"
        )
    if mode == "species_holdout" and not target_set:
        raise ValueError("species_holdout requires at least one target_species")
    _validate_named_species(available, target_set, excluded_set, validation_set)

    explicitly_excluded = table[table["_species_id"].isin(excluded_set)].copy()
    usable = table[~table["_species_id"].isin(excluded_set)].copy()
    target_test = usable[usable["_species_id"].isin(target_set)].copy()
    source = usable[~usable["_species_id"].isin(target_set)].copy()

    source_counts = source.groupby("_species_id")["_label"].agg(["size", "sum"])
    source_counts["negative"] = source_counts["size"] - source_counts["sum"]
    single_class_species = set(
        source_counts[(source_counts["sum"] == 0) | (source_counts["negative"] == 0)].index.astype(str)
    )
    low_support_species = set(
        source_counts[
            (source_counts["size"] < min_samples_per_species)
            | (source_counts["sum"] < min_positive_per_species)
            | (source_counts["negative"] < min_negative_per_species)
        ].index.astype(str)
    )
    ineligible_species = single_class_species | low_support_species
    if ineligible_species and single_class_species_policy == "error":
        raise ValueError(
            f"Ineligible source species: {sorted(ineligible_species, key=_species_sort_key)}"
        )
    training_only = source.iloc[0:0].copy()
    excluded = explicitly_excluded
    if single_class_species_policy == "exclude":
        excluded = pd.concat(
            [excluded, source[source["_species_id"].isin(ineligible_species)]], axis=0
        )
        source = source[~source["_species_id"].isin(ineligible_species)].copy()
    elif single_class_species_policy == "train_only":
        training_only = source[source["_species_id"].isin(ineligible_species)].copy()
        source = source[~source["_species_id"].isin(ineligible_species)].copy()

    rng = random.Random(int(random_seed))
    if mode == "within_species":
        train_indices: list[str] = []
        validation_indices: list[str] = []
        test_indices: list[str] = []
        for _, species_table in source.groupby("_species_id", sort=False):
            species_train, species_validation, species_test = _stratified_within_species_split(
                species_table, validation_ratio, test_ratio, rng
            )
            train_indices.extend(species_train)
            validation_indices.extend(species_validation)
            test_indices.extend(species_test)
        train = table[table["_sample_index"].isin(train_indices)].copy()
        train = pd.concat([train, training_only], axis=0)
        validation = table[table["_sample_index"].isin(validation_indices)].copy()
        test = table[table["_sample_index"].isin(test_indices)].copy()
    else:
        source_species = sorted(source["_species_id"].unique().tolist(), key=_species_sort_key)
        if validation_set:
            if not validation_set.issubset(set(source_species)):
                raise ValueError("validation_species were removed by the source-species policy")
        else:
            if not 0.0 < validation_species_ratio < 1.0:
                raise ValueError("validation_species_ratio must be in (0, 1)")
            if len(source_species) < 2:
                raise ValueError("At least two eligible source species are required")
            shuffled_species = source_species.copy()
            rng.shuffle(shuffled_species)
            validation_count = min(
                max(1, int(round(len(shuffled_species) * validation_species_ratio))),
                len(shuffled_species) - 1,
            )
            validation_set = set(shuffled_species[:validation_count])
        validation = source[source["_species_id"].isin(validation_set)].copy()
        train = source[~source["_species_id"].isin(validation_set)].copy()
        train = pd.concat([train, training_only], axis=0)
        test = target_test

    frames = {"train": train, "validation": validation, "test": test, "excluded": excluded}
    _assert_sample_disjoint(frames)
    if train.empty or validation.empty or test.empty:
        empty = [name for name in ("train", "validation", "test") if frames[name].empty]
        raise ValueError(f"Required data splits are empty: {empty}")
    if mode == "species_holdout":
        active = {name: set(frames[name]["_species_id"]) for name in ("train", "validation", "test")}
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
            overlap = active[left] & active[right]
            if overlap:
                raise AssertionError(f"Species leakage between {left} and {right}: {sorted(overlap)}")
        if set(test["_species_id"]) != target_set:
            raise AssertionError("The species_holdout test split does not exactly match target_species")

    statistics = {
        "mode": mode,
        "random_seed": int(random_seed),
        "target_species": sorted(target_set, key=_species_sort_key),
        "validation_species": sorted(set(validation["_species_id"]), key=_species_sort_key),
        "single_class_species_policy": single_class_species_policy,
        "single_class_source_species": sorted(single_class_species, key=_species_sort_key),
        "low_support_source_species": sorted(low_support_species, key=_species_sort_key),
        "minimums": {
            "samples": int(min_samples_per_species),
            "positive": int(min_positive_per_species),
            "negative": int(min_negative_per_species),
        },
        "sets": {name: _count_summary(frame) for name, frame in frames.items()},
    }
    print_split_statistics(statistics)
    return DataSplits(
        table=table,
        train=train,
        validation=validation,
        test=test,
        excluded=excluded,
        statistics=statistics,
        mode=mode,
        target_species=tuple(sorted(target_set, key=_species_sort_key)),
    )


def print_split_statistics(statistics: Mapping[str, Any]) -> None:
    print(f"\nSplit mode: {statistics['mode']}")
    for name, item in statistics["sets"].items():
        print(
            f"  {name}: n={item['sample_number']}, positive={item['positive']}, "
            f"negative={item['negative']}, species={item['species']}"
        )


def _open_lmdb(path: str):
    return lmdb.open(
        path,
        subdir=Path(path).is_dir(),
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
    )


def read_lmdb_metadata(path: str) -> dict[str, Any]:
    environment = _open_lmdb(path)
    try:
        with environment.begin(write=False) as transaction:
            payload = transaction.get(b"__metadata__")
        return pickle.loads(payload) if payload is not None else {}
    finally:
        environment.close()


def resolve_lmdb_feature_length(
    path: str, candidate_keys: Sequence[str], expected_input_size: int | None = None
) -> int:
    """Use embedded metadata, with a legacy-LMDB sample fallback."""
    metadata = read_lmdb_metadata(path)
    feature_length = metadata.get("feature_length")
    if feature_length is None:
        environment = _open_lmdb(path)
        try:
            with environment.begin(write=False) as transaction:
                for key in candidate_keys:
                    payload = transaction.get(str(key).encode("utf-8"))
                    if payload is None:
                        continue
                    sample = pickle.loads(payload)
                    feature = np.asarray(sample["feature"])
                    if feature.ndim != 2:
                        raise ValueError(f"LMDB key {key} has invalid feature shape {feature.shape}")
                    feature_length = int(feature.shape[1])
                    break
        finally:
            environment.close()
    if feature_length is None:
        raise ValueError("Could not infer the LMDB feature dimension")
    feature_length = int(feature_length)
    if expected_input_size is not None and int(expected_input_size) != feature_length:
        raise ValueError(
            f"LMDB feature dimension {feature_length} does not match input_size={expected_input_size}"
        )
    return feature_length


class FrozenLMDBDataset(Dataset):
    def __init__(
        self,
        table: pd.DataFrame,
        lmdb_path: str,
        max_length: int,
        feature_length: int,
        truncate_strategy: str = "head_tail",
    ):
        columns = ["_sample_index", "_lmdb_key", "_label", "_species_id", "_species_code"]
        self.records = table[columns].to_dict("records")
        self.lmdb_path = lmdb_path
        self.max_length = int(max_length)
        self.feature_length = int(feature_length)
        self.truncate_strategy = truncate_strategy
        self.environment = None

    def __len__(self) -> int:
        return len(self.records)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["environment"] = None
        return state

    def _get_environment(self):
        if self.environment is None:
            self.environment = _open_lmdb(self.lmdb_path)
        return self.environment

    def _truncate(self, feature: np.ndarray) -> np.ndarray:
        if feature.shape[0] <= self.max_length:
            return feature
        if self.truncate_strategy == "tail":
            return feature[-self.max_length :]
        if self.truncate_strategy == "head_tail" and self.max_length > 1:
            head = self.max_length // 2
            return np.concatenate([feature[:head], feature[-(self.max_length - head) :]], axis=0)
        return feature[: self.max_length]

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        with self._get_environment().begin(write=False) as transaction:
            payload = transaction.get(record["_lmdb_key"].encode("utf-8"))
        if payload is None:
            raise KeyError(f"LMDB key {record['_lmdb_key']} is missing")
        sample = pickle.loads(payload)
        feature = np.asarray(sample["feature"], dtype=np.float32)
        if feature.ndim != 2 or feature.shape[1] != self.feature_length:
            raise ValueError(
                f"LMDB key {record['_lmdb_key']} has feature shape {feature.shape}; "
                f"expected [length, {self.feature_length}]"
            )
        feature = np.ascontiguousarray(self._truncate(feature))
        if feature.shape[0] == 0:
            raise ValueError(f"LMDB key {record['_lmdb_key']} has an empty feature sequence")
        return {
            "residue_features": torch.from_numpy(feature),
            "label": float(record["_label"]),
            "species_id": record["_species_id"],
            "species_code": int(record["_species_code"]),
            "sample_index": record["_sample_index"],
        }


def collate_frozen(batch: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    max_length = max(item["residue_features"].shape[0] for item in batch)
    feature_length = batch[0]["residue_features"].shape[1]
    features = torch.zeros(len(batch), max_length, feature_length, dtype=torch.float32)
    valid_mask = torch.zeros(len(batch), max_length, dtype=torch.bool)
    for row, item in enumerate(batch):
        length = item["residue_features"].shape[0]
        features[row, :length] = item["residue_features"]
        valid_mask[row, :length] = True
    return {
        "residue_features": features,
        "valid_mask": valid_mask,
        "labels": torch.tensor([item["label"] for item in batch], dtype=torch.float32),
        "species_codes": torch.tensor([item["species_code"] for item in batch], dtype=torch.long),
        "species_ids": [item["species_id"] for item in batch],
        "sample_indices": [item["sample_index"] for item in batch],
    }


def _seed_worker(_worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def create_dataloaders(
    splits: DataSplits,
    batch_size: int,
    max_length: int,
    random_seed: int,
    num_workers: int,
    sampling_strategy: str,
    species_per_batch: int,
    samples_per_species: int,
    positive_fraction: float,
    feature_dir: str | None = None,
    feature_length: int | None = None,
    truncate_strategy: str = "head_tail",
) -> tuple[dict[str, DataLoader], SpeciesLabelBalancedBatchSampler | None]:
    if sampling_strategy not in {"random", "species_balanced"}:
        raise ValueError("sampling_strategy must be random or species_balanced")
    frames = {"train": splits.train, "validation": splits.validation, "test": splits.test}
    if not feature_dir or feature_length is None:
        raise ValueError("feature_dir and feature_length are required")
    datasets = {
        name: FrozenLMDBDataset(
            frame, feature_dir, max_length, feature_length, truncate_strategy
        )
        for name, frame in frames.items()
    }
    collate_function = collate_frozen

    batch_sampler = None
    if sampling_strategy == "species_balanced":
        batch_sampler = SpeciesLabelBalancedBatchSampler(
            splits.train["_label"].tolist(),
            splits.train["_species_id"].tolist(),
            species_per_batch,
            samples_per_species,
            positive_fraction,
            random_seed,
            len(splits.train),
        )
    loader_common: dict[str, Any] = {
        "num_workers": int(num_workers),
        "pin_memory": torch.cuda.is_available(),
        "collate_fn": collate_function,
        "worker_init_fn": _seed_worker,
        "persistent_workers": bool(num_workers > 0),
    }
    if num_workers > 0:
        loader_common["prefetch_factor"] = 2
    loaders: dict[str, DataLoader] = {}
    for offset, (name, dataset) in enumerate(datasets.items()):
        generator = torch.Generator().manual_seed(int(random_seed) + offset)
        if name == "train" and batch_sampler is not None:
            loaders[name] = DataLoader(
                dataset,
                batch_sampler=batch_sampler,
                generator=generator,
                **loader_common,
            )
        else:
            loaders[name] = DataLoader(
                dataset,
                batch_size=int(batch_size),
                shuffle=(name == "train"),
                generator=generator,
                **loader_common,
            )
    return loaders, batch_sampler
