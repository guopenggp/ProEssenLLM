"""Shared training engine with validation-only selection discipline."""

from __future__ import annotations

import json
import math
import random
import warnings
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from datasets import (
    DataSplits,
    create_dataloaders,
    prepare_metadata,
    resolve_lmdb_feature_length,
    split_metadata,
)
from evaluation.checkpoint import (
    BestCheckpointManager,
    build_checkpoint_payload,
    load_checkpoint,
    save_checkpoint,
)
from evaluation.leakage import audit_splits
from evaluation.metrics import (
    distribution_metrics,
    find_balanced_accuracy_threshold,
    save_predictions,
    species_performance_rows,
    write_species_performance,
)
from losses import CombinedSpeciesLoss, compute_species_positive_weights
from losses.species_losses import positive_weights_for_batch
from models import (
    ESMProteinEncoder,
    ProEssenLLM,
    ProEssenLLMClassifier,
    ProEssenLLMFeatureEncoder,
    parameter_statistics,
)
from samplers import sampler_statistics


FRAMEWORK_VERSION = "1.0.0"
CHECKPOINT_NAME = "best_validation_model.pt"


def to_serializable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_serializable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def save_json(value: Any, path: Path) -> None:
    path.write_text(
        json.dumps(to_serializable(value), ensure_ascii=False, indent=2), encoding="utf-8"
    )


def set_randomness(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(bool(deterministic), warn_only=True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = bool(deterministic)
        torch.backends.cudnn.benchmark = not bool(deterministic)


def resolve_device(requested: str) -> torch.device:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        warnings.warn(f"CUDA is unavailable; falling back from {requested} to CPU", RuntimeWarning)
        return torch.device("cpu")
    return torch.device(requested)


def selection_weights(args) -> dict[str, float]:
    return {
        "micro_auc": float(args.micro_auc_weight),
        "macro_auc": float(args.macro_auc_weight),
        "q25_species_auc": float(args.q25_auc_weight),
        "worst_species_auc": float(args.worst_auc_weight),
    }


def build_model(args, splits: DataSplits, device: torch.device):
    esm_encoder = None
    if args.encoder_mode == "frozen_lmdb":
        feature_length = resolve_lmdb_feature_length(
            args.feature_dir,
            splits.train["_lmdb_key"].tolist(),
            args.input_size,
        )
    else:
        esm_encoder = ESMProteinEncoder(
            model_path=args.esm_model_path,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            lora_target_modules=args.lora_target_modules,
            lora_last_n_layers=args.lora_last_n_layers,
            lora_train_layernorm=args.lora_train_layernorm,
            gradient_checkpointing=args.gradient_checkpointing,
        )
        feature_length = esm_encoder.output_dim
        if args.input_size is not None and int(args.input_size) != feature_length:
            raise ValueError(
                f"ESM output dimension {feature_length} does not match input_size={args.input_size}"
            )
    feature_encoder = ProEssenLLMFeatureEncoder(
        input_size=feature_length,
        hidden_size=args.hidden_size,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        attention_dropout=args.attention_dropout,
        linear_dropout=args.linear_dropout,
        drop_path_rate=args.drop_path_rate,
    )
    classifier = ProEssenLLMClassifier(args.hidden_size, args.linear_dropout)
    model = ProEssenLLM(feature_encoder, classifier, esm_encoder).to(device)
    architecture = {
        "encoder_mode": args.encoder_mode,
        "input_size": int(feature_length),
        "hidden_size": int(args.hidden_size),
        "num_heads": int(args.num_heads),
        "num_layers": int(args.num_layers),
        "attention_dropout": float(args.attention_dropout),
        "linear_dropout": float(args.linear_dropout),
        "drop_path_rate": float(args.drop_path_rate),
        "max_length": int(args.max_length),
        "lora_rank": int(args.lora_rank) if args.encoder_mode == "esm_lora" else None,
        "lora_alpha": float(args.lora_alpha) if args.encoder_mode == "esm_lora" else None,
        "lora_target_modules": (
            list(args.lora_target_modules) if args.encoder_mode == "esm_lora" else None
        ),
        "lora_last_n_layers": (
            int(args.lora_last_n_layers) if args.encoder_mode == "esm_lora" else None
        ),
        "lora_dropout": float(args.lora_dropout) if args.encoder_mode == "esm_lora" else None,
        "lora_train_layernorm": (
            bool(args.lora_train_layernorm) if args.encoder_mode == "esm_lora" else None
        ),
        "gradient_checkpointing": (
            bool(args.gradient_checkpointing) if args.encoder_mode == "esm_lora" else None
        ),
        "esm_model_path": str(args.esm_model_path) if args.encoder_mode == "esm_lora" else None,
    }
    return model, feature_length, architecture


def build_optimizer(args, model: ProEssenLLM):
    if args.encoder_mode == "frozen_lmdb":
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        return torch.optim.AdamW(
            trainable, lr=args.learning_rate, weight_decay=args.weight_decay
        )
    esm_parameters = []
    downstream_parameters = []
    forbidden = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("esm_encoder.esm_model."):
            esm_parameters.append(parameter)
            is_adapter = ".lora_a." in name or ".lora_b." in name
            is_layernorm = args.lora_train_layernorm and "norm" in name.lower()
            if not is_adapter and not is_layernorm:
                forbidden.append(name)
        else:
            downstream_parameters.append(parameter)
    if forbidden:
        raise AssertionError(f"Frozen ESM base parameters became trainable: {forbidden[:5]}")
    if not esm_parameters or not downstream_parameters:
        raise ValueError("Both ESM adapter and downstream optimizer groups must be non-empty")
    return torch.optim.AdamW(
        [
            {"params": esm_parameters, "lr": args.esm_learning_rate, "name": "esm_adapter"},
            {"params": downstream_parameters, "lr": args.head_learning_rate, "name": "downstream"},
        ],
        weight_decay=args.weight_decay,
    )


def build_scheduler(args, optimizer):
    def factor(epoch: int) -> float:
        if args.warmup_epochs > 0 and epoch < args.warmup_epochs:
            return max((epoch + 1) / args.warmup_epochs, 1e-3)
        progress = (epoch - args.warmup_epochs) / max(
            args.num_epochs - args.warmup_epochs, 1
        )
        return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def model_forward(model: ProEssenLLM, batch: Mapping[str, Any], device: torch.device):
    """Pass sequence features only; species metadata never enters the model."""
    if "tokens" in batch:
        return model(
            tokens=batch["tokens"].to(device, non_blocking=True),
            attention_mask=batch["attention_mask"].to(device, non_blocking=True),
        )
    return model(
        residue_features=batch["residue_features"].to(device, non_blocking=True),
        valid_mask=batch["valid_mask"].to(device, non_blocking=True),
    )


def train_one_epoch(
    model: ProEssenLLM,
    loader,
    optimizer,
    loss_function: CombinedSpeciesLoss,
    species_weights: Mapping[str, float],
    device: torch.device,
    scaler: GradScaler,
    use_amp: bool,
    accumulation_steps: int,
    max_grad_norm: float,
    metric_weights: Mapping[str, float],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    scores: list[float] = []
    labels: list[int] = []
    species_ids: list[str] = []
    total_loss = 0.0
    component_totals: dict[str, float] = {}
    batches = len(loader)
    for batch_index, batch in enumerate(tqdm(loader, desc="Training", leave=False)):
        targets = batch["labels"].to(device, non_blocking=True)
        species_codes = batch["species_codes"].to(device, non_blocking=True)
        positive_weights = positive_weights_for_batch(
            batch["species_ids"], species_weights, device
        )
        accumulation_group_start = (batch_index // accumulation_steps) * accumulation_steps
        accumulation_group_size = min(
            accumulation_steps, batches - accumulation_group_start
        )
        with autocast(enabled=use_amp):
            logits = model_forward(model, batch, device).reshape(-1)
            components = loss_function(
                logits,
                targets,
                species_codes,
                positive_weights,
                return_components=True,
            )
            scaled_loss = components["total_loss"] / accumulation_group_size
        scaler.scale(scaled_loss).backward()
        update_now = (batch_index + 1) % accumulation_steps == 0 or batch_index + 1 == batches
        if update_now:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                max_grad_norm,
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        scores.extend(torch.sigmoid(logits.detach()).cpu().numpy().tolist())
        labels.extend(targets.detach().cpu().numpy().astype(int).tolist())
        species_ids.extend(batch["species_ids"])
        total_loss += float(components["total_loss"].detach())
        for name, value in components.items():
            component_totals[name] = component_totals.get(name, 0.0) + float(value)
    summary, per_species = distribution_metrics(
        labels, scores, species_ids, threshold=0.5, selection_weights=metric_weights
    )
    summary["loss"] = total_loss / max(batches, 1)
    for name, total in component_totals.items():
        summary[name] = total / max(batches, 1)
    return summary, per_species


@torch.no_grad()
def evaluate(model: ProEssenLLM, loader, device: torch.device, use_amp: bool) -> dict[str, Any]:
    model.eval()
    scores: list[float] = []
    labels: list[int] = []
    species_ids: list[str] = []
    sample_indices: list[str] = []
    losses: list[float] = []
    for batch in tqdm(loader, desc="Evaluating", leave=False):
        targets = batch["labels"].to(device, non_blocking=True)
        with autocast(enabled=use_amp):
            logits = model_forward(model, batch, device).reshape(-1)
            loss = F.binary_cross_entropy_with_logits(logits, targets)
        scores.extend(torch.sigmoid(logits).cpu().numpy().tolist())
        labels.extend(targets.cpu().numpy().astype(int).tolist())
        species_ids.extend(batch["species_ids"])
        sample_indices.extend(batch["sample_indices"])
        losses.append(float(loss))
    return {
        "y_true": np.asarray(labels, dtype=int),
        "y_score": np.asarray(scores, dtype=float),
        "species_ids": np.asarray(species_ids, dtype=str),
        "sample_indices": np.asarray(sample_indices, dtype=str),
        "loss": float(np.mean(losses)) if losses else 0.0,
    }


def summarize_evaluation(
    raw: Mapping[str, Any], threshold: float, metric_weights: Mapping[str, float]
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    summary, per_species = distribution_metrics(
        raw["y_true"],
        raw["y_score"],
        raw["species_ids"],
        threshold,
        metric_weights,
    )
    summary["loss"] = float(raw["loss"])
    return summary, per_species


def print_species_metrics(phase: str, per_species: Mapping[str, Mapping[str, Any]]) -> None:
    print(f"  {phase} species metrics")
    for species_id, values in per_species.items():
        auc_text = "NA" if values["auc"] is None else f"{values['auc']:.4f}"
        aupr_text = "NA" if values["aupr"] is None else f"{values['aupr']:.4f}"
        print(
            f"    species={species_id} n={values['sample_number']} "
            f"positive_ratio={values['positive_ratio']:.4f} AUC={auc_text} "
            f"AUPR={aupr_text} F1={values['f1']:.4f} "
            f"balanced_accuracy={values['balanced_accuracy']:.4f}"
        )


def display_metric(value: Any) -> str:
    return "NA" if value is None else f"{float(value):.5f}"


class BaseTrainer:
    expected_mode: str | None = None

    def __init__(self, args):
        if self.expected_mode is not None and args.mode != self.expected_mode:
            raise ValueError(f"{type(self).__name__} requires mode={self.expected_mode}")
        self.args = args

    def run(self) -> dict[str, Any]:
        args = self.args
        set_randomness(args.random_seed, args.deterministic)
        device = resolve_device(args.device)
        use_amp = bool(args.use_amp and device.type == "cuda")
        output_directory = Path(args.save_path).resolve()
        output_directory.mkdir(parents=True, exist_ok=True)

        table, selected_sequence_column = prepare_metadata(
            args.data_path,
            args.label_name,
            args.group_column,
            args.encoder_mode,
            args.sequence_column,
        )
        splits = split_metadata(
            table=table,
            mode=args.mode,
            test_ratio=args.test_ratio,
            validation_ratio=args.validation_ratio,
            validation_species_ratio=args.validation_species_ratio,
            validation_species=args.validation_species,
            target_species=args.target_species,
            exclude_species=args.exclude_species,
            single_class_species_policy=args.single_class_species_policy,
            min_samples_per_species=args.min_samples_per_species,
            min_positive_per_species=args.min_positive_per_species,
            min_negative_per_species=args.min_negative_per_species,
            random_seed=args.random_seed,
        )
        save_json(splits.statistics, output_directory / "split_statistics.json")

        leakage_audit = audit_splits(
            args.mode,
            splits.frames(),
            splits.target_species,
            ProEssenLLM.MODEL_INPUT_KEYS,
        )
        save_json(leakage_audit, output_directory / "leakage_audit.json")

        model, feature_length, architecture = build_model(args, splits, device)
        model_statistics = parameter_statistics(model)
        save_json(model_statistics, output_directory / "model_statistics.json")
        alphabet = model.esm_encoder.alphabet if model.esm_encoder is not None else None
        loaders, batch_sampler = create_dataloaders(
            splits=splits,
            encoder_mode=args.encoder_mode,
            batch_size=args.batch_size,
            max_length=args.max_length,
            random_seed=args.random_seed,
            num_workers=args.num_workers,
            sampling_strategy=args.sampling_strategy,
            species_per_batch=args.species_per_batch,
            samples_per_species=args.samples_per_species,
            positive_fraction=args.positive_fraction,
            feature_dir=args.feature_dir,
            feature_length=feature_length,
            truncate_strategy=args.truncate_strategy,
            alphabet=alphabet,
        )

        config = vars(args).copy()
        config.update(
            {
                "framework": "ProEssenLLM",
                "framework_version": FRAMEWORK_VERSION,
                "resolved_device": str(device),
                "resolved_input_size": int(feature_length),
                "resolved_sequence_column": selected_sequence_column,
                "resolved_validation_species": sorted(
                    splits.validation["_species_id"].unique().tolist()
                ),
                "resolved_target_species": list(splits.target_species),
                "output_directory": str(output_directory),
            }
        )
        save_json(config, output_directory / "config.json")

        species_weights, species_weight_summary = compute_species_positive_weights(
            splits.train["_label"].tolist(),
            splits.train["_species_id"].tolist(),
            args.positive_weight_power,
            args.positive_weight_smoothing,
            args.positive_weight_clip_min,
            args.positive_weight_clip_max,
        )
        save_json(species_weight_summary, output_directory / "species_positive_weights.json")

        optimizer = build_optimizer(args, model)
        scheduler = build_scheduler(args, optimizer)
        scaler = GradScaler(enabled=use_amp)
        loss_function = CombinedSpeciesLoss(
            focal_gamma=args.focal_gamma,
            label_smoothing=args.label_smoothing,
            auc_weight=args.auc_weight,
            auc_loss_type=args.auc_loss_type,
            auc_margin=args.auc_margin,
            auc_max_pairs=args.auc_max_pairs,
            global_auc_ratio=args.global_auc_ratio,
            within_species_auc_ratio=args.within_species_auc_ratio,
        )
        metric_weights = selection_weights(args)
        checkpoint_manager = BestCheckpointManager(
            args.checkpoint_metric,
            args.early_stopping_patience,
            args.early_stopping_min_delta,
            args.checkpoint_tolerance,
        )
        checkpoint_path = output_directory / CHECKPOINT_NAME
        history: list[dict[str, Any]] = []
        sampler_history: list[dict[str, Any]] = []
        performance_rows: list[dict[str, Any]] = []

        for epoch in range(1, args.num_epochs + 1):
            if batch_sampler is not None:
                batch_sampler.set_epoch(epoch - 1)
            train_summary, train_per_species = train_one_epoch(
                model,
                loaders["train"],
                optimizer,
                loss_function,
                species_weights,
                device,
                scaler,
                use_amp,
                args.gradient_accumulation_steps,
                args.max_grad_norm,
                metric_weights,
            )
            validation_raw = evaluate(model, loaders["validation"], device, use_amp)
            threshold, _ = find_balanced_accuracy_threshold(
                validation_raw["y_true"], validation_raw["y_score"]
            )
            validation_summary, validation_per_species = summarize_evaluation(
                validation_raw, threshold, metric_weights
            )
            candidate = checkpoint_manager.candidate(validation_summary, epoch)
            save_best, should_stop = checkpoint_manager.update(candidate)
            if save_best:
                payload = build_checkpoint_payload(
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    epoch,
                    validation_summary,
                    threshold,
                    config,
                    architecture,
                    checkpoint_manager,
                )
                save_checkpoint(payload, checkpoint_path)
                print(f"Saved {CHECKPOINT_NAME} at epoch {epoch}")

            performance_rows.extend(
                species_performance_rows(train_per_species, epoch, "train", False)
            )
            performance_rows.extend(
                species_performance_rows(validation_per_species, epoch, "validation", True)
            )
            write_species_performance(
                performance_rows, output_directory / "species_performance.csv"
            )

            sampler_summary = sampler_statistics(
                splits.train["_label"].tolist(),
                splits.train["_species_id"].tolist(),
                batch_sampler,
                args.sampling_strategy,
            )
            sampler_summary["epoch"] = epoch
            sampler_history.append(sampler_summary)
            learning_rates = [group["lr"] for group in optimizer.param_groups]
            history.append(
                {
                    "epoch": epoch,
                    "train": train_summary,
                    "validation": validation_summary,
                    "validation_threshold": threshold,
                    "checkpoint_metric": args.checkpoint_metric,
                    "candidate_selection_score": candidate.selection_score,
                    "saved_checkpoint": save_best,
                    "learning_rates": learning_rates,
                    "early_stopping_counter": checkpoint_manager.counter,
                }
            )

            print(f"\nEpoch {epoch}/{args.num_epochs}")
            print(
                f"  train loss={train_summary['loss']:.5f} "
                f"micro_AUC={train_summary['micro_auc']} macro_AUC={train_summary['macro_auc']}"
            )
            print(
                f"  validation loss={validation_summary['loss']:.5f} "
                f"micro_AUC={display_metric(validation_summary['micro_auc'])} "
                f"macro_AUC={display_metric(validation_summary['macro_auc'])} "
                f"q25_AUC={display_metric(validation_summary['q25_species_auc'])} "
                f"worst_AUC={display_metric(validation_summary['worst_species_auc'])} "
                f"threshold={threshold:.3f}"
            )
            print_species_metrics("train", train_per_species)
            print_species_metrics("validation", validation_per_species)
            scheduler.step()
            if should_stop:
                print(f"Early stopping at epoch {epoch}")
                break

        save_json(history, output_directory / "training_history.json")
        save_json({"epochs": sampler_history}, output_directory / "sampler_statistics.json")
        if not checkpoint_path.exists():
            raise RuntimeError("Training completed without a validation checkpoint")

        checkpoint = load_checkpoint(
            checkpoint_path,
            model,
            device,
            args.mode,
            architecture,
            expected_target_species=list(splits.target_species),
            expected_validation_species=splits.validation["_species_id"].unique().tolist(),
        )
        selected_threshold = float(checkpoint["validation_threshold"])
        selected_epoch = int(checkpoint["epoch"])

        final_validation_raw = evaluate(model, loaders["validation"], device, use_amp)
        final_validation, final_validation_species = summarize_evaluation(
            final_validation_raw, selected_threshold, metric_weights
        )
        final_test_raw = evaluate(model, loaders["test"], device, use_amp)
        final_test, final_test_species = summarize_evaluation(
            final_test_raw, selected_threshold, metric_weights
        )
        performance_rows.extend(
            species_performance_rows(
                final_validation_species, selected_epoch, "selected_validation", True
            )
        )
        performance_rows.extend(
            species_performance_rows(final_test_species, selected_epoch, "test", False)
        )
        write_species_performance(
            performance_rows, output_directory / "species_performance.csv"
        )

        save_json(final_validation, output_directory / "validation_results.json")
        save_json(
            final_validation_species,
            output_directory / "validation_results_per_species.json",
        )
        save_json(final_test, output_directory / "test_results.json")
        save_json(final_test_species, output_directory / "test_results_per_species.json")
        save_predictions(final_test_raw, selected_threshold, output_directory / "test_predictions.csv")
        if args.mode == "species_holdout":
            save_json(final_test, output_directory / "target_species_results.json")
            save_json(
                final_test_species,
                output_directory / "target_species_results_per_species.json",
            )

        leakage_audit.update(
            {
                "selected_checkpoint_epoch": selected_epoch,
                "selected_threshold": selected_threshold,
                "selected_threshold_source": checkpoint["threshold_source"],
                "selected_checkpoint_source": checkpoint["checkpoint_source"],
                "test_evaluated_after_checkpoint_load": True,
            }
        )
        save_json(leakage_audit, output_directory / "leakage_audit.json")
        run_summary = {
            "mode": args.mode,
            "selected_epoch": selected_epoch,
            "selected_threshold": selected_threshold,
            "validation": final_validation,
            "test": final_test,
            "checkpoint": str(checkpoint_path),
            "output_directory": str(output_directory),
        }
        save_json(run_summary, output_directory / "run_summary.json")
        print(f"Training complete. Results: {output_directory}")
        return run_summary
