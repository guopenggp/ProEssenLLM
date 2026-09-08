"""Thresholded and threshold-free metrics at sample and species levels."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _arrays(y_true: Iterable, y_score: Iterable) -> tuple[np.ndarray, np.ndarray]:
    true = np.asarray(y_true).reshape(-1).astype(int)
    score = np.asarray(y_score).reshape(-1).astype(float)
    if true.size != score.size:
        raise ValueError(f"y_true and y_score lengths differ: {true.size} != {score.size}")
    if not np.isin(true, [0, 1]).all():
        raise ValueError("y_true must contain only 0 and 1")
    if not np.isfinite(score).all():
        raise ValueError("y_score contains NaN or infinite values")
    return true, score


def safe_auc(y_true: Iterable, y_score: Iterable) -> tuple[float | None, bool]:
    true, score = _arrays(y_true, y_score)
    if true.size == 0 or np.unique(true).size < 2:
        return None, False
    return float(roc_auc_score(true, score)), True


def safe_aupr(y_true: Iterable, y_score: Iterable) -> tuple[float | None, bool]:
    true, score = _arrays(y_true, y_score)
    if true.size == 0 or not np.any(true == 1):
        return None, False
    return float(average_precision_score(true, score)), True


def balanced_accuracy_details(y_true: Iterable, y_pred: Iterable) -> dict[str, float]:
    true = np.asarray(y_true).reshape(-1).astype(int)
    predicted = np.asarray(y_pred).reshape(-1).astype(int)
    if true.size != predicted.size:
        raise ValueError("y_true and y_pred lengths differ")
    true_negative, false_positive, false_negative, true_positive = confusion_matrix(
        true, predicted, labels=[0, 1]
    ).ravel()
    sensitivity = (
        float(true_positive / (true_positive + false_negative))
        if true_positive + false_negative
        else 0.0
    )
    specificity = (
        float(true_negative / (true_negative + false_positive))
        if true_negative + false_positive
        else 0.0
    )
    return {
        "balanced_accuracy": (sensitivity + specificity) / 2.0,
        "sensitivity": sensitivity,
        "specificity": specificity,
    }


def binary_metrics(y_true: Iterable, y_score: Iterable, threshold: float) -> dict[str, Any]:
    true, score = _arrays(y_true, y_score)
    if true.size == 0:
        raise ValueError("Metrics cannot be computed for an empty split")
    predicted = (score >= float(threshold)).astype(int)
    true_negative, false_positive, false_negative, true_positive = confusion_matrix(
        true, predicted, labels=[0, 1]
    ).ravel()
    auc, auc_available = safe_auc(true, score)
    aupr, aupr_available = safe_aupr(true, score)
    result: dict[str, Any] = {
        "sample_number": int(true.size),
        "positive_count": int(np.sum(true == 1)),
        "negative_count": int(np.sum(true == 0)),
        "positive_ratio": float(np.mean(true == 1)),
        "auc": auc,
        "auc_available": auc_available,
        "aupr": aupr,
        "aupr_available": aupr_available,
        "threshold": float(threshold),
        "accuracy": float(np.mean(true == predicted)),
        "precision": float(precision_score(true, predicted, zero_division=0)),
        "recall": float(recall_score(true, predicted, zero_division=0)),
        "f1": float(f1_score(true, predicted, zero_division=0)),
        "mcc": float(matthews_corrcoef(true, predicted)),
        "tn": int(true_negative),
        "fp": int(false_positive),
        "fn": int(false_negative),
        "tp": int(true_positive),
        "confusion_matrix": [
            [int(true_negative), int(false_positive)],
            [int(false_negative), int(true_positive)],
        ],
    }
    result.update(balanced_accuracy_details(true, predicted))
    return result


def per_species_metrics(
    y_true: Iterable,
    y_score: Iterable,
    species_ids: Iterable,
    threshold: float,
) -> dict[str, dict[str, Any]]:
    true, score = _arrays(y_true, y_score)
    species = np.asarray(species_ids).reshape(-1).astype(str)
    if species.size != true.size:
        raise ValueError("species_ids length differs from y_true")
    sort_key = lambda value: (not value.isdigit(), int(value) if value.isdigit() else value)
    result: dict[str, dict[str, Any]] = {}
    for species_id in sorted(np.unique(species), key=sort_key):
        mask = species == species_id
        item = binary_metrics(true[mask], score[mask], threshold)
        item["species_id"] = species_id
        result[species_id] = item
    return result


def distribution_metrics(
    y_true: Iterable,
    y_score: Iterable,
    species_ids: Iterable,
    threshold: float,
    selection_weights: Mapping[str, float] | None = None,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    true, score = _arrays(y_true, y_score)
    species = per_species_metrics(true, score, species_ids, threshold)
    pooled = binary_metrics(true, score, threshold)
    valid_auc = [float(item["auc"]) for item in species.values() if item["auc_available"]]
    valid_aupr = [float(item["aupr"]) for item in species.values() if item["aupr_available"]]
    balanced_accuracies = [float(item["balanced_accuracy"]) for item in species.values()]
    macro_auc = float(np.mean(valid_auc)) if valid_auc else None
    q25_auc = float(np.quantile(valid_auc, 0.25)) if valid_auc else None
    worst_auc = float(np.min(valid_auc)) if valid_auc else None
    weights = dict(
        selection_weights
        or {
            "micro_auc": 0.55,
            "macro_auc": 0.20,
            "q25_species_auc": 0.15,
            "worst_species_auc": 0.10,
        }
    )
    components = {
        "micro_auc": pooled["auc"],
        "macro_auc": macro_auc,
        "q25_species_auc": q25_auc,
        "worst_species_auc": worst_auc,
    }
    selection_score = None
    if all(value is not None for value in components.values()):
        selection_score = float(sum(weights[key] * float(components[key]) for key in components))
    summary: dict[str, Any] = dict(pooled)
    summary.update(
        {
            "micro_auc": pooled["auc"],
            "micro_aupr": pooled["aupr"],
            "macro_auc": macro_auc,
            "macro_aupr": float(np.mean(valid_aupr)) if valid_aupr else None,
            "q25_species_auc": q25_auc,
            "worst_species_auc": worst_auc,
            "micro_balanced_accuracy": pooled["balanced_accuracy"],
            "macro_balanced_accuracy": (
                float(np.mean(balanced_accuracies)) if balanced_accuracies else None
            ),
            "auc_selection_score": selection_score,
            "valid_auc_species_count": len(valid_auc),
            "species_count": len(species),
            "selection_weights": weights,
        }
    )
    return summary, species


def checkpoint_score(summary: Mapping[str, Any], metric: str) -> float:
    key = "auc_selection_score" if metric == "composite" else metric
    if key not in summary:
        raise KeyError(f"Unknown checkpoint metric: {metric}")
    value = summary[key]
    if value is None:
        raise ValueError(f"Validation metric {key} is unavailable")
    return float(value)


def find_balanced_accuracy_threshold(
    y_true: Iterable,
    y_score: Iterable,
    start: float = 0.01,
    stop: float = 0.99,
    step: float = 0.005,
) -> tuple[float, float]:
    true, score = _arrays(y_true, y_score)
    if true.size == 0 or np.unique(true).size < 2:
        raise ValueError("Validation data needs both classes for threshold selection")
    best_threshold = 0.5
    best_balanced_accuracy = -1.0
    for threshold in np.arange(start, stop + step / 2.0, step):
        value = balanced_accuracy_details(true, score >= threshold)["balanced_accuracy"]
        if value > best_balanced_accuracy + 1e-12:
            best_threshold = float(threshold)
            best_balanced_accuracy = float(value)
    return best_threshold, best_balanced_accuracy


def species_performance_rows(
    per_species: Mapping[str, Mapping[str, Any]],
    epoch: int,
    phase: str,
    checkpoint_eligible: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for species_id, values in per_species.items():
        rows.append(
            {
                "epoch": int(epoch),
                "phase": phase,
                "checkpoint_eligible": bool(checkpoint_eligible),
                "species_id": species_id,
                "sample_number": values["sample_number"],
                "positive_ratio": values["positive_ratio"],
                "auc": values["auc"],
                "auc_available": values["auc_available"],
                "aupr": values["aupr"],
                "f1": values["f1"],
                "balanced_accuracy": values["balanced_accuracy"],
                "threshold": values["threshold"],
            }
        )
    return rows


SPECIES_PERFORMANCE_FIELDS = [
    "epoch",
    "phase",
    "checkpoint_eligible",
    "species_id",
    "sample_number",
    "positive_ratio",
    "auc",
    "auc_available",
    "aupr",
    "f1",
    "balanced_accuracy",
    "threshold",
]


def write_species_performance(rows: Iterable[Mapping[str, Any]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SPECIES_PERFORMANCE_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in SPECIES_PERFORMANCE_FIELDS})


def save_predictions(raw: Mapping[str, Any], threshold: float, path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        fields = ["sample_index", "species_id", "y_true", "y_score", "y_pred", "threshold"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for sample_index, species_id, label, score in zip(
            raw["sample_indices"], raw["species_ids"], raw["y_true"], raw["y_score"]
        ):
            writer.writerow(
                {
                    "sample_index": sample_index,
                    "species_id": species_id,
                    "y_true": int(label),
                    "y_score": float(score),
                    "y_pred": int(score >= threshold),
                    "threshold": float(threshold),
                }
            )
