"""ProEssenLLM model for precomputed residue-level features."""

from __future__ import annotations

import torch
import torch.nn as nn


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
    """Encode precomputed residue-level protein features."""

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
    """Protein essentiality classifier shared by both experiment modes.

    Species identifiers are intentionally absent from this forward interface.
    They may be used by samplers, losses, and metrics, but never as model input.
    """

    MODEL_INPUT_KEYS = frozenset({"residue_features", "valid_mask"})

    def __init__(
        self,
        feature_encoder: ProEssenLLMFeatureEncoder,
        classifier: ProEssenLLMClassifier,
    ):
        super().__init__()
        self.feature_encoder = feature_encoder
        self.classifier = classifier

    def forward(
        self,
        residue_features: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.classifier(self.feature_encoder(residue_features, valid_mask))

    def checkpoint_state(self) -> tuple[str, dict[str, torch.Tensor]]:
        return "full", self.state_dict()

    def load_checkpoint_state(self, state_kind: str, state: dict[str, torch.Tensor]) -> None:
        if state_kind != "full":
            raise ValueError(f"Unsupported model state kind: {state_kind}")
        self.load_state_dict(state, strict=True)


def parameter_statistics(model: nn.Module) -> dict[str, int | float]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "trainable_parameter_percentage": 100.0 * trainable / total if total else 0.0,
    }
