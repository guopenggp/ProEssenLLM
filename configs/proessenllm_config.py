"""Command-line and YAML/JSON configuration for the unified entry point."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence


def load_configuration_file(path: str) -> dict[str, Any]:
    configuration_path = Path(path)
    if not configuration_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    if configuration_path.suffix.lower() == ".json":
        loaded = json.loads(configuration_path.read_text(encoding="utf-8"))
    elif configuration_path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise ImportError("PyYAML is required to read YAML configuration files") from exc
        loaded = yaml.safe_load(configuration_path.read_text(encoding="utf-8"))
    else:
        raise ValueError("Configuration files must be JSON or YAML")
    if not isinstance(loaded, dict):
        raise ValueError("The configuration root must be a mapping")
    return loaded


def build_parser(defaults: dict[str, Any] | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train ProEssenLLM for known-species or unknown-species prediction"
    )
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--mode",
        choices=["within_species", "species_holdout"],
        required=not bool(defaults and defaults.get("mode")),
    )
    parser.add_argument("--data_path", default="./data/metadata.pkl")
    parser.add_argument("--feature_dir", default="./data/features.lmdb")
    parser.add_argument("--save_path", default=None)
    parser.add_argument("--label_name", default="essential")
    parser.add_argument("--group_column", default="group")

    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--validation_ratio", type=float, default=0.1)
    parser.add_argument("--validation_species_ratio", type=float, default=0.2)
    parser.add_argument("--validation_species", nargs="*", default=[])
    parser.add_argument("--target_species", nargs="*", default=[])
    parser.add_argument("--exclude_species", nargs="*", default=[])
    parser.add_argument(
        "--single_class_species_policy",
        choices=["exclude", "train_only", "error"],
        default="exclude",
    )
    parser.add_argument("--min_samples_per_species", type=int, default=30)
    parser.add_argument("--min_positive_per_species", type=int, default=1)
    parser.add_argument("--min_negative_per_species", type=int, default=1)

    parser.add_argument("--input_size", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=1000)
    parser.add_argument(
        "--truncate_strategy", choices=["head", "tail", "head_tail"], default="head_tail"
    )
    parser.add_argument("--hidden_size", type=int, default=320)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--attention_dropout", type=float, default=0.1)
    parser.add_argument("--linear_dropout", type=float, default=0.15)
    parser.add_argument("--drop_path_rate", type=float, default=0.05)

    parser.add_argument(
        "--sampling_strategy", choices=["random", "species_balanced"], default="species_balanced"
    )
    parser.add_argument("--species_per_batch", type=int, default=8)
    parser.add_argument("--samples_per_species", type=int, default=16)
    parser.add_argument("--positive_fraction", type=float, default=0.4)
    parser.add_argument("--batch_size", type=int, default=128)

    parser.add_argument("--focal_gamma", type=float, default=1.5)
    parser.add_argument("--positive_weight_power", type=float, default=0.75)
    parser.add_argument("--positive_weight_smoothing", type=float, default=5.0)
    parser.add_argument("--positive_weight_clip_min", type=float, default=0.5)
    parser.add_argument("--positive_weight_clip_max", type=float, default=8.0)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--auc_weight", type=float, default=0.10)
    parser.add_argument("--auc_loss_type", choices=["logistic", "hinge"], default="logistic")
    parser.add_argument("--auc_margin", type=float, default=1.0)
    parser.add_argument("--auc_max_pairs", type=int, default=4096)
    parser.add_argument("--global_auc_ratio", type=float, default=0.30)
    parser.add_argument("--within_species_auc_ratio", type=float, default=0.70)

    parser.add_argument(
        "--checkpoint_metric",
        choices=[
            "composite",
            "micro_auc",
            "macro_auc",
            "q25_species_auc",
            "worst_species_auc",
        ],
        default="composite",
    )
    parser.add_argument("--micro_auc_weight", type=float, default=0.55)
    parser.add_argument("--macro_auc_weight", type=float, default=0.20)
    parser.add_argument("--q25_auc_weight", type=float, default=0.15)
    parser.add_argument("--worst_auc_weight", type=float, default=0.10)
    parser.add_argument("--checkpoint_tolerance", type=float, default=0.001)
    parser.add_argument("--early_stopping_patience", type=int, default=15)
    parser.add_argument("--early_stopping_min_delta", type=float, default=0.0002)

    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--num_epochs", type=int, default=150)
    parser.add_argument("--warmup_epochs", type=int, default=1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--deterministic", action=argparse.BooleanOptionalAction, default=True
    )
    if defaults:
        unknown = set(defaults) - {action.dest for action in parser._actions}
        if unknown:
            raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
        parser.set_defaults(**defaults)
    return parser


def validate_configuration(args: argparse.Namespace) -> None:
    if args.save_path is None:
        args.save_path = str(Path("results") / args.mode)
    if args.mode == "within_species":
        if args.target_species or args.validation_species:
            raise ValueError(
                "within_species cannot use target_species or validation_species"
            )
    elif not args.target_species:
        raise ValueError("species_holdout requires --target_species")
    if set(map(str, args.target_species)) & set(map(str, args.exclude_species)):
        raise ValueError("target_species and exclude_species must not overlap")
    if not args.feature_dir:
        raise ValueError("feature_dir is required")
    if args.max_length <= 0:
        raise ValueError("max_length must be positive")
    if args.hidden_size <= 0 or args.num_heads <= 0 or args.hidden_size % args.num_heads:
        raise ValueError("hidden_size must be positive and divisible by num_heads")
    if args.num_layers <= 0 or args.batch_size <= 0 or args.num_epochs <= 0:
        raise ValueError("num_layers, batch_size, and num_epochs must be positive")
    if args.gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")
    if args.sampling_strategy == "species_balanced":
        expected_batch_size = args.species_per_batch * args.samples_per_species
        if args.batch_size != expected_batch_size:
            raise ValueError(
                "species_balanced requires batch_size == species_per_batch * samples_per_species; "
                f"expected {expected_batch_size}, got {args.batch_size}"
            )
        if args.species_per_batch < 2:
            raise ValueError("species_per_batch must be at least 2")
        if not 0.0 < args.positive_fraction < 1.0:
            raise ValueError("positive_fraction must be in (0, 1)")
    if not math.isclose(
        args.global_auc_ratio + args.within_species_auc_ratio, 1.0, abs_tol=1e-9
    ):
        raise ValueError("global_auc_ratio + within_species_auc_ratio must equal 1")
    checkpoint_weight_sum = (
        args.micro_auc_weight
        + args.macro_auc_weight
        + args.q25_auc_weight
        + args.worst_auc_weight
    )
    if not math.isclose(checkpoint_weight_sum, 1.0, abs_tol=1e-9):
        raise ValueError("Checkpoint AUC weights must sum to 1")
    if args.early_stopping_patience <= 0 or args.max_grad_norm <= 0:
        raise ValueError("Patience and max_grad_norm must be positive")


def parse_configuration(argv: Sequence[str] | None = None) -> argparse.Namespace:
    preparser = argparse.ArgumentParser(add_help=False)
    preparser.add_argument("--config")
    preliminary, _ = preparser.parse_known_args(argv)
    defaults = load_configuration_file(preliminary.config) if preliminary.config else None
    parser = build_parser(defaults)
    args = parser.parse_args(argv)
    validate_configuration(args)
    return args
