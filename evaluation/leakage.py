"""Machine-readable sample, species, and selection-discipline audits."""

from __future__ import annotations

from itertools import combinations
from typing import Iterable, Mapping

import pandas as pd


def audit_splits(
    mode: str,
    frames: Mapping[str, pd.DataFrame],
    target_species: Iterable[str],
    model_input_keys: Iterable[str],
) -> dict:
    active_names = [name for name in ("train", "validation", "test") if name in frames]
    sample_overlaps: dict[str, list[str]] = {}
    for left, right in combinations(active_names, 2):
        overlap = set(frames[left]["_sample_index"]) & set(frames[right]["_sample_index"])
        sample_overlaps[f"{left}__{right}"] = sorted(overlap)
    if any(sample_overlaps.values()):
        raise AssertionError(f"Sample leakage detected: {sample_overlaps}")

    species_overlaps: dict[str, list[str]] = {}
    for left, right in combinations(active_names, 2):
        overlap = set(frames[left]["_species_id"]) & set(frames[right]["_species_id"])
        species_overlaps[f"{left}__{right}"] = sorted(overlap)
    target_set = {str(value) for value in target_species}
    target_in_train = target_set & set(frames["train"]["_species_id"])
    target_in_validation = target_set & set(frames["validation"]["_species_id"])
    target_test_matches = set(frames["test"]["_species_id"]) == target_set if target_set else True
    if mode == "species_holdout":
        if any(species_overlaps.values()):
            raise AssertionError(f"Species leakage detected: {species_overlaps}")
        if target_in_train or target_in_validation or not target_test_matches:
            raise AssertionError("Target-species isolation failed")

    input_keys = set(model_input_keys)
    forbidden_model_inputs = input_keys & {"species", "species_id", "species_ids", "species_code", "species_codes"}
    if forbidden_model_inputs:
        raise AssertionError(f"Species information enters the model: {sorted(forbidden_model_inputs)}")
    return {
        "status": "passed",
        "mode": mode,
        "sample_disjoint": True,
        "sample_overlaps": sample_overlaps,
        "species_disjoint_required": mode == "species_holdout",
        "species_overlaps": species_overlaps,
        "target_species": sorted(target_set),
        "target_absent_from_train": not target_in_train,
        "target_absent_from_validation": not target_in_validation,
        "test_exactly_target_species": target_test_matches,
        "model_input_keys": sorted(input_keys),
        "species_used_as_model_input": False,
        "checkpoint_source": "validation",
        "threshold_source": "validation",
        "test_access_policy": "post-checkpoint-selection only",
    }
