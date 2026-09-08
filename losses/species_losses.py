"""Species-aware focal and pairwise ranking objectives."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Mapping

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def normalize_species_id(value) -> str:
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def compute_species_positive_weights(
    labels: Iterable,
    species_ids: Iterable,
    power: float = 0.75,
    smoothing: float = 5.0,
    clip_min: float = 0.5,
    clip_max: float = 8.0,
) -> tuple[dict[str, float], dict[str, dict[str, float | int]]]:
    """Compute positive-label weights from training data only."""
    if smoothing < 0:
        raise ValueError("positive-weight smoothing must be non-negative")
    if clip_min <= 0 or clip_max < clip_min:
        raise ValueError("Invalid positive-weight clipping range")
    counts: dict[str, dict[str, int]] = defaultdict(lambda: {"positive": 0, "negative": 0})
    for label, species_id in zip(labels, species_ids):
        normalized = normalize_species_id(species_id)
        integer_label = int(label)
        if integer_label not in (0, 1):
            raise ValueError(f"Non-binary training label: {label}")
        key = "positive" if integer_label == 1 else "negative"
        counts[normalized][key] += 1

    sort_key = lambda value: (not value.isdigit(), int(value) if value.isdigit() else value)
    weights: dict[str, float] = {}
    summary: dict[str, dict[str, float | int]] = {}
    for species_id in sorted(counts, key=sort_key):
        positive = counts[species_id]["positive"]
        negative = counts[species_id]["negative"]
        raw_weight = ((negative + smoothing) / (positive + smoothing)) ** power
        weight = float(np.clip(raw_weight, clip_min, clip_max))
        weights[species_id] = weight
        summary[species_id] = {
            "train_positive_count": positive,
            "train_negative_count": negative,
            "positive_ratio": positive / (positive + negative) if positive + negative else 0.0,
            "positive_weight": weight,
        }
    return weights, summary


def positive_weights_for_batch(
    species_ids: Iterable, weights: Mapping[str, float], device: torch.device
) -> torch.Tensor:
    values = [weights[normalize_species_id(species_id)] for species_id in species_ids]
    return torch.tensor(values, dtype=torch.float32, device=device)


class SpeciesFocalLoss(nn.Module):
    def __init__(self, gamma: float = 1.5, label_smoothing: float = 0.0):
        super().__init__()
        if gamma < 0:
            raise ValueError("focal_gamma must be non-negative")
        if not 0.0 <= label_smoothing < 1.0:
            raise ValueError("label_smoothing must be in [0, 1)")
        self.gamma = float(gamma)
        self.label_smoothing = float(label_smoothing)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        positive_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        logits = logits.reshape(-1)
        hard_targets = targets.reshape(-1).float()
        smooth_targets = hard_targets
        if self.label_smoothing:
            smooth_targets = hard_targets * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
        log_probability = F.logsigmoid(logits)
        log_negative_probability = F.logsigmoid(-logits)
        probability = log_probability.exp()
        target_probability = (
            probability * hard_targets + (1.0 - probability) * (1.0 - hard_targets)
        )
        focal_factor = (1.0 - target_probability).pow(self.gamma)
        binary_cross_entropy = (
            -smooth_targets * log_probability
            - (1.0 - smooth_targets) * log_negative_probability
        )
        if positive_weights is None:
            sample_weights = torch.ones_like(hard_targets)
        else:
            positive_weights = positive_weights.reshape(-1).to(logits)
            if positive_weights.numel() != logits.numel():
                raise ValueError("positive_weights shape does not match logits")
            sample_weights = torch.where(
                hard_targets >= 0.5, positive_weights, torch.ones_like(positive_weights)
            )
        return (focal_factor * sample_weights * binary_cross_entropy).mean()


class PairwiseRankingLoss(nn.Module):
    def __init__(self, loss_type: str = "logistic", margin: float = 1.0, max_pairs: int = 4096):
        super().__init__()
        if loss_type not in {"logistic", "hinge"}:
            raise ValueError("auc_loss_type must be logistic or hinge")
        if max_pairs <= 0:
            raise ValueError("auc_max_pairs must be positive")
        self.loss_type = loss_type
        self.margin = float(margin)
        self.max_pairs = int(max_pairs)

    def _pair_loss(self, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        if positive.numel() == 0 or negative.numel() == 0:
            return positive.new_zeros(())
        pair_count = positive.numel() * negative.numel()
        if pair_count > self.max_pairs:
            positive_count = min(positive.numel(), max(1, int(self.max_pairs**0.5)))
            negative_count = min(negative.numel(), max(1, self.max_pairs // positive_count))
            positive = positive[
                torch.randperm(positive.numel(), device=positive.device)[:positive_count]
            ]
            negative = negative[
                torch.randperm(negative.numel(), device=negative.device)[:negative_count]
            ]
        differences = positive[:, None] - negative[None, :]
        if self.loss_type == "logistic":
            return F.softplus(-differences).mean()
        return F.relu(self.margin - differences).mean()

    def global_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = logits.reshape(-1)
        targets = targets.reshape(-1)
        return self._pair_loss(logits[targets >= 0.5], logits[targets < 0.5])

    def within_species_loss(
        self, logits: torch.Tensor, targets: torch.Tensor, species_codes: torch.Tensor
    ) -> torch.Tensor:
        logits = logits.reshape(-1)
        targets = targets.reshape(-1)
        species_codes = species_codes.reshape(-1)
        losses: list[torch.Tensor] = []
        for species_code in torch.unique(species_codes):
            mask = species_codes == species_code
            species_targets = targets[mask]
            if torch.any(species_targets >= 0.5) and torch.any(species_targets < 0.5):
                species_logits = logits[mask]
                losses.append(
                    self._pair_loss(
                        species_logits[species_targets >= 0.5],
                        species_logits[species_targets < 0.5],
                    )
                )
        return torch.stack(losses).mean() if losses else logits.new_zeros(())

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        species_codes: torch.Tensor,
        global_ratio: float,
        within_species_ratio: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        global_loss = self.global_loss(logits, targets)
        within_species_loss = self.within_species_loss(logits, targets, species_codes)
        total = global_ratio * global_loss + within_species_ratio * within_species_loss
        return total, global_loss, within_species_loss


class CombinedSpeciesLoss(nn.Module):
    """Focal classification plus global and within-species ranking objectives."""

    def __init__(
        self,
        focal_gamma: float = 1.5,
        label_smoothing: float = 0.0,
        auc_weight: float = 0.10,
        auc_loss_type: str = "logistic",
        auc_margin: float = 1.0,
        auc_max_pairs: int = 4096,
        global_auc_ratio: float = 0.30,
        within_species_auc_ratio: float = 0.70,
    ):
        super().__init__()
        self.focal = SpeciesFocalLoss(focal_gamma, label_smoothing)
        self.ranking = PairwiseRankingLoss(auc_loss_type, auc_margin, auc_max_pairs)
        self.auc_weight = float(auc_weight)
        self.global_auc_ratio = float(global_auc_ratio)
        self.within_species_auc_ratio = float(within_species_auc_ratio)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        species_codes: torch.Tensor,
        positive_weights: torch.Tensor | None = None,
        return_components: bool = False,
    ):
        focal_loss = self.focal(logits, targets, positive_weights)
        ranking_loss, global_loss, within_species_loss = self.ranking(
            logits,
            targets,
            species_codes,
            self.global_auc_ratio,
            self.within_species_auc_ratio,
        )
        total_loss = focal_loss + self.auc_weight * ranking_loss
        if not return_components:
            return total_loss
        return {
            "total_loss": total_loss,
            "focal_loss": focal_loss.detach(),
            "auc_loss": ranking_loss.detach(),
            "global_pairwise_auc_loss": global_loss.detach(),
            "within_species_pairwise_auc_loss": within_species_loss.detach(),
        }
