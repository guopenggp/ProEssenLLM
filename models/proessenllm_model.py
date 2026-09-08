"""Unified ProEssenLLM model with frozen-feature and ESM-LoRA encoders."""

from __future__ import annotations

import math
import re
from typing import Sequence

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


class LoRALinear(nn.Module):
    """A frozen linear layer with a trainable low-rank residual."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float):
        super().__init__()
        if rank <= 0:
            raise ValueError("lora_rank must be positive")
        self.base = base
        self.base.requires_grad_(False)
        self.rank = int(rank)
        self.scaling = float(alpha) / float(rank)
        self.dropout = nn.Dropout(float(dropout))
        self.lora_a = nn.Linear(base.in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = self.lora_b(self.lora_a(self.dropout(inputs)))
        return self.base(inputs) + self.scaling * residual


class CheckpointedModule(nn.Module):
    """Run a wrapped transformer layer through activation checkpointing."""

    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        if self.training and torch.is_grad_enabled():
            forwarded_kwargs = dict(kwargs)

            def custom_forward(*inputs):
                return self.module(*inputs, **forwarded_kwargs)

            return checkpoint(custom_forward, *args, use_reentrant=False)
        return self.module(*args, **kwargs)


def _parent_and_child(root: nn.Module, module_name: str) -> tuple[nn.Module, str]:
    parts = module_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def _layer_index(module_name: str) -> int | None:
    match = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", module_name)
    return int(match.group(1)) if match else None


def inject_lora(
    esm_model: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.05,
    target_modules: Sequence[str] = ("q_proj", "v_proj"),
    last_n_layers: int = 12,
    train_layernorm: bool = False,
    verbose: bool = True,
) -> list[str]:
    """Freeze ESM and inject LoRA into selected linear layers."""
    esm_model.requires_grad_(False)
    target_set = set(target_modules)
    candidates: list[tuple[str, nn.Linear, int | None]] = []
    layer_indices: set[int] = set()
    for name, module in esm_model.named_modules():
        index = _layer_index(name)
        if index is not None:
            layer_indices.add(index)
        if isinstance(module, nn.Linear) and name.rsplit(".", 1)[-1] in target_set:
            candidates.append((name, module, index))
    if not layer_indices:
        raise ValueError("Could not locate ESM transformer layers named 'layers.<index>'")
    selected_layers = set(sorted(layer_indices)[-int(last_n_layers) :])
    selected = [(name, module) for name, module, index in candidates if index in selected_layers]
    if not selected:
        raise ValueError(
            f"No LoRA targets matched {sorted(target_set)} in the last {last_n_layers} ESM layers"
        )
    replaced: list[str] = []
    for name, module in selected:
        parent, child = _parent_and_child(esm_model, name)
        setattr(parent, child, LoRALinear(module, rank, alpha, dropout))
        replaced.append(name)
    if train_layernorm:
        for name, module in esm_model.named_modules():
            if isinstance(module, nn.LayerNorm) and _layer_index(name) in selected_layers:
                module.requires_grad_(True)
    if verbose:
        print(f"Injected LoRA into {len(replaced)} ESM modules")
    return replaced


def enable_esm_gradient_checkpointing(esm_model: nn.Module) -> None:
    if hasattr(esm_model, "gradient_checkpointing_enable"):
        esm_model.gradient_checkpointing_enable()
        return
    layers = getattr(esm_model, "layers", None)
    if isinstance(layers, nn.ModuleList):
        for index, layer in enumerate(list(layers)):
            if not isinstance(layer, CheckpointedModule):
                layers[index] = CheckpointedModule(layer)
        return
    raise ValueError("ESM transformer layers were not found for gradient checkpointing")


class ESMProteinEncoder(nn.Module):
    """Offline fair-esm backbone with trainable LoRA adapters."""

    def __init__(
        self,
        model_path: str,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.05,
        lora_target_modules: Sequence[str] = ("q_proj", "v_proj"),
        lora_last_n_layers: int = 12,
        lora_train_layernorm: bool = False,
        gradient_checkpointing: bool = False,
        esm_model: nn.Module | None = None,
        alphabet=None,
        verbose: bool = True,
    ):
        super().__init__()
        if esm_model is None or alphabet is None:
            try:
                from esm import pretrained
            except ImportError as exc:
                raise ImportError("fair-esm is required for encoder_mode=esm_lora") from exc
            esm_model, alphabet = pretrained.load_model_and_alphabet(model_path)
        self.esm_model = esm_model
        self.alphabet = alphabet
        self.padding_idx = int(alphabet.padding_idx)
        self.cls_idx = int(getattr(alphabet, "cls_idx", -1))
        self.eos_idx = int(getattr(alphabet, "eos_idx", -1))
        self.representation_layer = int(getattr(esm_model, "num_layers"))
        self.output_dim = int(getattr(esm_model, "embed_dim"))
        self.replaced_modules = inject_lora(
            self.esm_model,
            rank=lora_rank,
            alpha=lora_alpha,
            dropout=lora_dropout,
            target_modules=lora_target_modules,
            last_n_layers=lora_last_n_layers,
            train_layernorm=lora_train_layernorm,
            verbose=verbose,
        )
        if gradient_checkpointing:
            enable_esm_gradient_checkpointing(self.esm_model)

    def forward(
        self, tokens: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output = self.esm_model(
            tokens, repr_layers=[self.representation_layer], return_contacts=False
        )
        residue_features = output["representations"][self.representation_layer][:, 1:]
        residue_tokens = tokens[:, 1:]
        valid_mask = residue_tokens.ne(self.padding_idx)
        if self.eos_idx >= 0:
            valid_mask &= residue_tokens.ne(self.eos_idx)
        if self.cls_idx >= 0:
            valid_mask &= residue_tokens.ne(self.cls_idx)
        if attention_mask is not None:
            valid_mask &= attention_mask[:, 1:].bool()
        return residue_features, valid_mask


class PreNormTransformerBlock(nn.Module):
    def __init__(
        self,
        dimension: int,
        num_heads: int,
        feedforward_dimension: int,
        attention_dropout: float,
        linear_dropout: float,
        drop_path: float,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dimension)
        self.attention = nn.MultiheadAttention(
            dimension, num_heads, dropout=attention_dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(dimension)
        self.feedforward = nn.Sequential(
            nn.Linear(dimension, feedforward_dimension),
            nn.GELU(),
            nn.Dropout(linear_dropout),
            nn.Linear(feedforward_dimension, dimension),
            nn.Dropout(linear_dropout),
        )
        self.drop_path = float(drop_path)

    def _stochastic_depth(self, inputs: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_path <= 0:
            return inputs
        keep = torch.rand(inputs.size(0), 1, 1, device=inputs.device) >= self.drop_path
        return inputs * keep / (1.0 - self.drop_path)

    def forward(self, inputs: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        normalized = self.norm1(inputs)
        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        outputs = (inputs + self._stochastic_depth(attended)).masked_fill(
            padding_mask.unsqueeze(-1), 0.0
        )
        outputs = (outputs + self._stochastic_depth(self.feedforward(self.norm2(outputs)))).masked_fill(
            padding_mask.unsqueeze(-1), 0.0
        )
        return outputs


class AttentionPooling(nn.Module):
    def __init__(self, dimension: int, dropout: float):
        super().__init__()
        bottleneck = max(dimension // 4, 1)
        self.score = nn.Sequential(
            nn.Linear(dimension, bottleneck),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck, 1),
        )

    def forward(self, inputs: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        scores = self.score(inputs).squeeze(-1)
        scores = scores.masked_fill(~valid_mask, torch.finfo(scores.dtype).min)
        scores[~valid_mask.any(dim=1)] = 0.0
        weights = torch.softmax(scores, dim=1).masked_fill(~valid_mask, 0.0)
        return torch.sum(inputs * weights.unsqueeze(-1), dim=1)


class ProEssenLLMFeatureEncoder(nn.Module):
    """Shared residue-level encoder for both supported input modes."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 320,
        num_heads: int = 4,
        num_layers: int = 3,
        attention_dropout: float = 0.1,
        linear_dropout: float = 0.15,
        drop_path_rate: float = 0.05,
    ):
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.input_projection = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.Dropout(linear_dropout),
            nn.LayerNorm(hidden_size),
        )
        self.blocks = nn.ModuleList(
            PreNormTransformerBlock(
                hidden_size,
                num_heads,
                hidden_size * 4,
                attention_dropout,
                linear_dropout,
                drop_path_rate * index / max(num_layers - 1, 1),
            )
            for index in range(num_layers)
        )
        self.final_norm = nn.LayerNorm(hidden_size)
        self.attention_pool = AttentionPooling(hidden_size, linear_dropout)

    def forward(self, residue_features: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        expected = ("batch", "length", self.input_size)
        if residue_features.ndim != 3 or residue_features.size(-1) != self.input_size:
            raise ValueError(f"Expected residue features {expected}, got {tuple(residue_features.shape)}")
        valid_mask = valid_mask.bool()
        if valid_mask.shape != residue_features.shape[:2]:
            raise ValueError("valid_mask shape does not match residue features")
        if not valid_mask.any(dim=1).all():
            raise ValueError("Every protein must contain at least one valid residue")
        padding_mask = ~valid_mask
        encoded = self.input_projection(residue_features).masked_fill(
            padding_mask.unsqueeze(-1), 0.0
        )
        for block in self.blocks:
            encoded = block(encoded, padding_mask)
        encoded = self.final_norm(encoded).masked_fill(padding_mask.unsqueeze(-1), 0.0)
        attention_pool = self.attention_pool(encoded, valid_mask)
        denominator = valid_mask.sum(dim=1, keepdim=True).clamp_min(1).to(encoded.dtype)
        mean_pool = encoded.sum(dim=1) / denominator
        max_pool = encoded.masked_fill(
            padding_mask.unsqueeze(-1), torch.finfo(encoded.dtype).min
        ).max(dim=1).values
        return torch.cat([attention_pool, mean_pool, max_pool], dim=-1)


class ProEssenLLMClassifier(nn.Module):
    def __init__(self, hidden_size: int, dropout: float = 0.15):
        super().__init__()
        self.input = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
        )
        self.residual = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, pooled_features: torch.Tensor) -> torch.Tensor:
        hidden = self.input(pooled_features)
        return self.output(self.norm(hidden + self.residual(hidden)))


class ProEssenLLM(nn.Module):
    """Protein essentiality classifier shared by both task modes.

    Species identifiers are intentionally absent from this forward interface.
    They may be used by samplers, losses, and metrics, but never as model input.
    """

    MODEL_INPUT_KEYS = frozenset({"residue_features", "valid_mask", "tokens", "attention_mask"})

    def __init__(
        self,
        feature_encoder: ProEssenLLMFeatureEncoder,
        classifier: ProEssenLLMClassifier,
        esm_encoder: ESMProteinEncoder | None = None,
    ):
        super().__init__()
        self.esm_encoder = esm_encoder
        self.feature_encoder = feature_encoder
        self.classifier = classifier

    def forward(
        self,
        residue_features: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
        tokens: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.esm_encoder is not None:
            if tokens is None:
                raise ValueError("tokens are required in esm_lora mode")
            residue_features, valid_mask = self.esm_encoder(tokens, attention_mask)
        elif residue_features is None or valid_mask is None:
            raise ValueError("residue_features and valid_mask are required in frozen_lmdb mode")
        return self.classifier(self.feature_encoder(residue_features, valid_mask))

    def checkpoint_state(self) -> tuple[str, dict[str, torch.Tensor]]:
        """Return a compact state while keeping frozen-backbone checkpoints practical."""
        if self.esm_encoder is None:
            return "full", self.state_dict()
        trainable_names = {name for name, parameter in self.named_parameters() if parameter.requires_grad}
        state = {name: tensor for name, tensor in self.state_dict().items() if name in trainable_names}
        return "trainable", state

    def load_checkpoint_state(self, state_kind: str, state: dict[str, torch.Tensor]) -> None:
        if state_kind == "full":
            self.load_state_dict(state, strict=True)
            return
        if state_kind != "trainable":
            raise ValueError(f"Unsupported model state kind: {state_kind}")
        current = self.state_dict()
        unknown = set(state) - set(current)
        if unknown:
            raise KeyError(f"Unknown checkpoint keys: {sorted(unknown)[:10]}")
        expected = {name for name, parameter in self.named_parameters() if parameter.requires_grad}
        missing = expected - set(state)
        if missing:
            raise KeyError(f"Checkpoint is missing trainable keys: {sorted(missing)[:10]}")
        current.update(state)
        self.load_state_dict(current, strict=True)


def parameter_statistics(model: nn.Module) -> dict[str, int | float]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    lora = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if ".lora_a." in name or ".lora_b." in name
    )
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "lora_parameters": lora,
        "trainable_parameter_percentage": 100.0 * trainable / total if total else 0.0,
    }
