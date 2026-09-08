#!/usr/bin/env python
"""Compare a completed baseline run with a ProEssenLLM run."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping


METRICS = [
    ("AUC", "auc"),
    ("AUPR", "aupr"),
    ("Balanced Accuracy", "balanced_accuracy"),
    ("F1", "f1"),
    ("Macro AUC", "macro_auc"),
    ("Worst species AUC", "worst_species_auc"),
]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def locate_results(directory: Path, mode: str, is_baseline: bool) -> Path:
    candidates = ["test_results.json"]
    if mode == "species_holdout":
        candidates = (
            ["zeroshot_results.json", "target_species_results.json", "test_results.json"]
            if is_baseline
            else ["target_species_results.json", "test_results.json"]
        )
    for name in candidates:
        path = directory / name
        if path.exists():
            return path
    raise FileNotFoundError(f"No result JSON found in {directory}; tried {candidates}")


def locate_per_species(directory: Path, mode: str, is_baseline: bool) -> Path | None:
    candidates = ["test_results_per_species.json"]
    if mode == "species_holdout":
        candidates = (
            [
                "zeroshot_results_per_species.json",
                "target_species_results_per_species.json",
                "test_results_per_species.json",
            ]
            if is_baseline
            else ["target_species_results_per_species.json", "test_results_per_species.json"]
        )
    for name in candidates:
        path = directory / name
        if path.exists():
            return path
    return None


def normalized_metrics(
    summary: Mapping[str, Any], per_species: Mapping[str, Mapping[str, Any]] | list | None
) -> dict[str, float | None]:
    normalized = {
        "auc": summary.get("auc", summary.get("micro_auc")),
        "aupr": summary.get("aupr", summary.get("micro_aupr")),
        "balanced_accuracy": summary.get(
            "balanced_accuracy", summary.get("micro_balanced_accuracy")
        ),
        "f1": summary.get("f1"),
        "macro_auc": summary.get("macro_auc"),
        "worst_species_auc": summary.get("worst_species_auc"),
    }
    if isinstance(per_species, list):
        species_values = per_species
    elif isinstance(per_species, Mapping):
        species_values = list(per_species.values())
    else:
        species_values = []
    valid_auc = [
        float(item["auc"])
        for item in species_values
        if item.get("auc") is not None and item.get("auc_available", True)
    ]
    if normalized["macro_auc"] is None and valid_auc:
        normalized["macro_auc"] = sum(valid_auc) / len(valid_auc)
    if normalized["worst_species_auc"] is None and valid_auc:
        normalized["worst_species_auc"] = min(valid_auc)
    return {key: (float(value) if value is not None else None) for key, value in normalized.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline_dir", required=True, type=Path)
    parser.add_argument("--proessenllm_dir", required=True, type=Path)
    parser.add_argument("--mode", choices=["within_species", "species_holdout"], required=True)
    parser.add_argument("--output_dir", type=Path, default=Path("comparison"))
    args = parser.parse_args()

    rows = []
    run_metrics = {}
    for run_name, directory, baseline in (
        ("Original 722", args.baseline_dir, True),
        ("ProEssenLLM", args.proessenllm_dir, False),
    ):
        summary = load_json(locate_results(directory, args.mode, baseline))
        per_species_path = locate_per_species(directory, args.mode, baseline)
        per_species = load_json(per_species_path) if per_species_path else None
        run_metrics[run_name] = normalized_metrics(summary, per_species)

    for label, key in METRICS:
        baseline_value = run_metrics["Original 722"][key]
        new_value = run_metrics["ProEssenLLM"][key]
        rows.append(
            {
                "metric": label,
                "original_722": baseline_value,
                "proessenllm": new_value,
                "absolute_change": (
                    new_value - baseline_value
                    if baseline_value is not None and new_value is not None
                    else None
                ),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / f"{args.mode}_performance_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    markdown_path = args.output_dir / f"{args.mode}_performance_comparison.md"

    def display(value):
        return "N/A" if value is None else f"{value:.6f}"

    lines = [
        f"# {args.mode} performance comparison",
        "",
        "| Metric | Original 722 | ProEssenLLM | Absolute change |",
        "|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['metric']} | {display(row['original_722'])} | "
            f"{display(row['proessenllm'])} | {display(row['absolute_change'])} |"
        )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved {csv_path}")
    print(f"Saved {markdown_path}")


if __name__ == "__main__":
    main()
