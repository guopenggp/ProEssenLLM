"""Deterministic species-and-label-balanced mini-batches."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Iterator, Sequence

import numpy as np
from torch.utils.data import Sampler


def normalize_species_id(value) -> str:
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _sort_key(value: str):
    return (not value.isdigit(), int(value) if value.isdigit() else value)


class SpeciesLabelBalancedBatchSampler(Sampler[list[int]]):
    """Select multiple species, then fixed positive/negative counts per species.

    Species are sampled uniformly without replacement inside each batch. Samples
    may be drawn with replacement when a species/class pool is smaller than its
    requested quota. This guarantees exactly ``species_per_batch`` distinct
    species in every batch.
    """

    def __init__(
        self,
        labels: Sequence[int],
        species_ids: Sequence,
        species_per_batch: int = 8,
        samples_per_species: int = 16,
        positive_fraction: float = 0.4,
        seed: int = 42,
        num_samples: int | None = None,
    ):
        labels_array = np.asarray(labels, dtype=int).reshape(-1)
        species_array = np.asarray([normalize_species_id(value) for value in species_ids])
        if labels_array.size != species_array.size:
            raise ValueError("labels and species_ids must have equal length")
        if not np.isin(labels_array, [0, 1]).all():
            raise ValueError("labels must be binary")
        if species_per_batch < 2:
            raise ValueError("species_per_batch must be at least 2")
        if samples_per_species < 2:
            raise ValueError("samples_per_species must be at least 2")
        if not 0.0 < positive_fraction < 1.0:
            raise ValueError("positive_fraction must be in (0, 1)")

        self.labels = labels_array
        self.species_ids = species_array
        self.species_per_batch = int(species_per_batch)
        self.samples_per_species = int(samples_per_species)
        self.positive_fraction = float(positive_fraction)
        self.positive_per_species = int(round(samples_per_species * positive_fraction))
        self.positive_per_species = max(1, min(samples_per_species - 1, self.positive_per_species))
        self.negative_per_species = samples_per_species - self.positive_per_species
        self.seed = int(seed)
        self.epoch = 0
        self.species_to_indices: dict[str, dict[int, list[int]]] = {}
        ineligible: list[str] = []
        for species_id in sorted(np.unique(species_array).tolist(), key=_sort_key):
            local = np.flatnonzero(species_array == species_id)
            positive = local[labels_array[local] == 1].astype(int).tolist()
            negative = local[labels_array[local] == 0].astype(int).tolist()
            if not positive or not negative:
                ineligible.append(species_id)
                continue
            self.species_to_indices[species_id] = {1: positive, 0: negative}
        self.eligible_species = sorted(self.species_to_indices, key=_sort_key)
        self.ineligible_species = sorted(ineligible, key=_sort_key)
        if self.ineligible_species:
            raise ValueError(
                "species_balanced sampling requires both labels in every training species; "
                f"ineligible species: {self.ineligible_species}"
            )
        if self.species_per_batch > len(self.eligible_species):
            raise ValueError(
                f"species_per_batch={self.species_per_batch} exceeds the "
                f"{len(self.eligible_species)} eligible training species"
            )
        self.batch_size = self.species_per_batch * self.samples_per_species
        epoch_samples = int(num_samples if num_samples is not None else labels_array.size)
        self.num_batches = max(1, int(np.ceil(epoch_samples / self.batch_size)))
        self.last_batches: list[list[int]] = []

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    @staticmethod
    def _draw(rng: np.random.Generator, pool: list[int], size: int) -> list[int]:
        return rng.choice(pool, size=size, replace=len(pool) < size).astype(int).tolist()

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        batches: list[list[int]] = []
        for _ in range(self.num_batches):
            selected_species = rng.choice(
                self.eligible_species, size=self.species_per_batch, replace=False
            ).tolist()
            batch: list[int] = []
            for species_id in selected_species:
                pools = self.species_to_indices[species_id]
                batch.extend(self._draw(rng, pools[1], self.positive_per_species))
                batch.extend(self._draw(rng, pools[0], self.negative_per_species))
            rng.shuffle(batch)
            batches.append(batch)
        self.last_batches = batches
        return iter(batches)

    def __len__(self) -> int:
        return self.num_batches


def sampler_statistics(
    labels: Sequence[int],
    species_ids: Sequence,
    sampler: SpeciesLabelBalancedBatchSampler | None,
    strategy: str,
) -> dict:
    normalized_species = [normalize_species_id(value) for value in species_ids]
    labels_array = np.asarray(labels, dtype=int)

    def summarize(indices: Sequence[int]) -> dict:
        per_species: dict[str, dict[str, int | float]] = {}
        counts: dict[str, Counter] = defaultdict(Counter)
        for index in indices:
            counts[normalized_species[index]][int(labels_array[index])] += 1
        for species_id in sorted(counts, key=_sort_key):
            positive = int(counts[species_id][1])
            negative = int(counts[species_id][0])
            total = positive + negative
            per_species[species_id] = {
                "sample_number": total,
                "positive": positive,
                "negative": negative,
                "positive_ratio": positive / total if total else 0.0,
            }
        return per_species

    original_indices = list(range(len(labels_array)))
    sampled_indices = (
        [index for batch in sampler.last_batches for index in batch]
        if sampler is not None and sampler.last_batches
        else original_indices
    )
    sampled_labels = labels_array[sampled_indices] if sampled_indices else np.asarray([], dtype=int)
    result = {
        "sampling_strategy": strategy,
        "original_per_species": summarize(original_indices),
        "sampled_per_species": summarize(sampled_indices),
        "sampled_total": len(sampled_indices),
        "sampled_positive_ratio": float(np.mean(sampled_labels == 1)) if sampled_labels.size else 0.0,
    }
    if sampler is not None:
        result.update(
            {
                "species_per_batch": sampler.species_per_batch,
                "samples_per_species": sampler.samples_per_species,
                "requested_positive_fraction": sampler.positive_fraction,
                "effective_positive_fraction": sampler.positive_per_species / sampler.samples_per_species,
                "batch_size": sampler.batch_size,
                "num_batches": len(sampler),
                "epoch": sampler.epoch,
            }
        )
    return result
