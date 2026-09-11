"""Composite validation selection and versioned checkpoint compatibility."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from .metrics import checkpoint_score


PROJECT_NAME = "ProEssenLLM"
CHECKPOINT_FORMAT_VERSION = 2


def architecture_fingerprint(architecture: Mapping[str, Any]) -> str:
    serialized = json.dumps(dict(architecture), sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ValidationCandidate:
    selection_score: float
    macro_balanced_accuracy: float
    validation_loss: float
    epoch: int


def candidate_is_better(
    candidate: ValidationCandidate,
    incumbent: ValidationCandidate | None,
    tolerance: float,
) -> bool:
    if incumbent is None:
        return True
    score_delta = candidate.selection_score - incumbent.selection_score
    if score_delta > tolerance:
        return True
    if abs(score_delta) <= tolerance:
        balanced_accuracy_delta = (
            candidate.macro_balanced_accuracy - incumbent.macro_balanced_accuracy
        )
        if balanced_accuracy_delta > 1e-12:
            return True
        if (
            abs(balanced_accuracy_delta) <= 1e-12
            and candidate.validation_loss < incumbent.validation_loss
        ):
            return True
    return False


class BestCheckpointManager:
    def __init__(
        self,
        checkpoint_metric: str,
        patience: int,
        min_delta: float,
        tolerance: float,
    ):
        self.checkpoint_metric = checkpoint_metric
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.tolerance = float(tolerance)
        self.best: ValidationCandidate | None = None
        self.counter = 0

    def candidate(self, summary: Mapping[str, Any], epoch: int) -> ValidationCandidate:
        return ValidationCandidate(
            selection_score=checkpoint_score(summary, self.checkpoint_metric),
            macro_balanced_accuracy=float(summary["macro_balanced_accuracy"]),
            validation_loss=float(summary["loss"]),
            epoch=int(epoch),
        )

    def update(self, candidate: ValidationCandidate) -> tuple[bool, bool]:
        save_checkpoint = candidate_is_better(candidate, self.best, self.tolerance)
        previous = self.best
        if save_checkpoint:
            primary_gain = float("inf") if previous is None else candidate.selection_score - previous.selection_score
            self.best = candidate
            if primary_gain >= self.min_delta:
                self.counter = 0
            elif previous is not None and abs(primary_gain) <= self.tolerance:
                self.counter = 0
            else:
                self.counter += 1
        else:
            self.counter += 1
        return save_checkpoint, self.counter >= self.patience

    def state_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_metric": self.checkpoint_metric,
            "patience": self.patience,
            "min_delta": self.min_delta,
            "tolerance": self.tolerance,
            "counter": self.counter,
            "best": asdict(self.best) if self.best else None,
        }


def build_checkpoint_payload(
    model,
    optimizer,
    scheduler,
    scaler,
    epoch: int,
    validation_summary: Mapping[str, Any],
    validation_threshold: float,
    config: Mapping[str, Any],
    architecture: Mapping[str, Any],
    manager: BestCheckpointManager,
) -> dict[str, Any]:
    state_kind, model_state = model.checkpoint_state()
    return {
        "project": PROJECT_NAME,
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "mode": config["mode"],
        "input_source": config["input_source"],
        "architecture": dict(architecture),
        "architecture_fingerprint": architecture_fingerprint(architecture),
        "model_state_kind": state_kind,
        "model_state": model_state,
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict(),
        "epoch": int(epoch),
        "validation_metrics": dict(validation_summary),
        "validation_threshold": float(validation_threshold),
        "threshold_source": "validation",
        "checkpoint_source": "validation",
        "target_species": list(
            config.get("resolved_target_species", config.get("target_species", []))
        ),
        "validation_species": list(config.get("resolved_validation_species", [])),
        "selection_state": manager.state_dict(),
        "config": dict(config),
    }


def save_checkpoint(payload: Mapping[str, Any], path: Path) -> None:
    torch.save(dict(payload), path)


def load_checkpoint(
    path: Path,
    model,
    device: torch.device,
    expected_mode: str,
    architecture: Mapping[str, Any],
    expected_target_species: list[str] | tuple[str, ...] | None = None,
    expected_validation_species: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    if checkpoint.get("project") != PROJECT_NAME:
        raise ValueError("Checkpoint project identifier is incompatible")
    if int(checkpoint.get("format_version", -1)) != CHECKPOINT_FORMAT_VERSION:
        raise ValueError("Checkpoint format version is incompatible")
    if checkpoint.get("mode") != expected_mode:
        raise ValueError(
            f"Checkpoint mode {checkpoint.get('mode')} cannot be loaded for {expected_mode}"
        )
    expected_fingerprint = architecture_fingerprint(architecture)
    if checkpoint.get("architecture_fingerprint") != expected_fingerprint:
        raise ValueError("Checkpoint architecture does not match the current model")
    if checkpoint.get("checkpoint_source") != "validation":
        raise ValueError("Checkpoint was not selected from validation data")
    if checkpoint.get("threshold_source") != "validation":
        raise ValueError("Checkpoint threshold was not selected from validation data")
    if expected_target_species is not None:
        expected_target = sorted(map(str, expected_target_species))
        checkpoint_target = sorted(map(str, checkpoint.get("target_species", [])))
        if checkpoint_target != expected_target:
            raise ValueError("Checkpoint target species do not match the current split")
    if expected_validation_species is not None:
        expected_validation = sorted(map(str, expected_validation_species))
        checkpoint_validation = sorted(map(str, checkpoint.get("validation_species", [])))
        if checkpoint_validation != expected_validation:
            raise ValueError("Checkpoint validation species do not match the current split")
    model.load_checkpoint_state(checkpoint["model_state_kind"], checkpoint["model_state"])
    return checkpoint
