from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass(frozen=True)
class DynamicSchemaConfig:
    state_dim: int
    schema_slots: int = 6
    anchor_names: Tuple[str, ...] = ("entity", "relation", "event")
    base_temperature: float = 0.75
    widen_gain: float = 1.40
    narrow_gain: float = 0.55

    def validate(self) -> None:
        if self.state_dim <= 0:
            raise ValueError("state_dim must be positive.")
        if self.schema_slots <= 0:
            raise ValueError("schema_slots must be positive.")
        if len(self.anchor_names) <= 0:
            raise ValueError("anchor_names must not be empty.")
        if self.base_temperature <= 0.0:
            raise ValueError("base_temperature must be positive.")
        if self.widen_gain < 0.0 or self.narrow_gain < 0.0:
            raise ValueError("widen_gain and narrow_gain must be non-negative.")


class DynamicSchemaOperator(nn.Module):
    def __init__(self, config: DynamicSchemaConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.anchor_names = config.anchor_names
        self.schema_keys = nn.Parameter(torch.randn(config.schema_slots, config.state_dim) * 0.02)
        self.schema_to_anchor = nn.Parameter(
            torch.randn(config.schema_slots, len(config.anchor_names)) * 0.02
        )
        self.query_proj = nn.Linear(config.state_dim * 2, config.state_dim, bias=False)
        self.anchor_router = nn.Linear(config.state_dim * 2, len(config.anchor_names))
        self.widen_gate = nn.Linear(config.state_dim * 2, 1)
        self.narrow_gate = nn.Linear(config.state_dim * 2, 1)
        self.split_gate = nn.Linear(config.state_dim * 2, 1)
        self.merge_gate = nn.Linear(config.state_dim * 2, 1)
        self.suspend_gate = nn.Linear(config.state_dim * 2, 1)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.zeros_(self.anchor_router.weight)
        nn.init.zeros_(self.anchor_router.bias)
        for gate in (
            self.widen_gate,
            self.narrow_gate,
            self.split_gate,
            self.merge_gate,
            self.suspend_gate,
        ):
            nn.init.zeros_(gate.weight)
            nn.init.zeros_(gate.bias)

    def forward(self, signal: Tensor, previous: Tensor) -> Dict[str, Tensor]:
        joined = torch.cat([signal, previous], dim=-1)
        query = F.normalize(self.query_proj(joined), dim=-1, eps=1e-6)
        norm_schema_keys = F.normalize(self.schema_keys, dim=-1, eps=1e-6)
        raw_scores = torch.einsum("bd,nd->bn", query, norm_schema_keys)

        widen = torch.sigmoid(self.widen_gate(joined))
        narrow = torch.sigmoid(self.narrow_gate(joined))
        split = torch.sigmoid(self.split_gate(joined))
        merge = torch.sigmoid(self.merge_gate(joined))
        suspend = torch.sigmoid(self.suspend_gate(joined))

        temperature = self.config.base_temperature
        temperature = temperature * (1.0 + self.config.widen_gain * widen)
        temperature = temperature / (1.0 + self.config.narrow_gain * narrow)
        base_weights = F.softmax(raw_scores / temperature.clamp_min(0.05), dim=-1)

        broadened = F.softmax(raw_scores / (temperature * 1.45).clamp_min(0.05), dim=-1)
        sharpened = F.softmax(raw_scores / (temperature * 0.65).clamp_min(0.05), dim=-1)
        schema_weights = (
            (1.0 - split - merge).clamp_min(0.0) * base_weights
            + split * broadened
            + merge * sharpened
        )

        max_weight = schema_weights.max(dim=-1, keepdim=True).values
        suspend_mass = suspend * (1.0 - max_weight)
        schema_weights = (1.0 - suspend_mass) * schema_weights + suspend_mass / float(
            self.config.schema_slots
        )
        schema_weights = schema_weights / schema_weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        schema_embedding = torch.matmul(schema_weights, self.schema_keys)
        anchor_logits = self.anchor_router(joined) + torch.matmul(schema_weights, self.schema_to_anchor)
        anchor_weights = F.softmax(anchor_logits, dim=-1)

        active_threshold = 1.0 / float(self.config.schema_slots)
        active_ratio = (schema_weights > active_threshold).float().mean(dim=-1)
        stats: Dict[str, Tensor] = {
            "schema_weights": schema_weights,
            "anchor_weights": anchor_weights,
            "schema_embedding": schema_embedding,
            "schema_entropy": -(schema_weights * torch.log(schema_weights + 1e-8)).sum(dim=-1),
            "anchor_entropy": -(anchor_weights * torch.log(anchor_weights + 1e-8)).sum(dim=-1),
            "widen": widen.squeeze(-1),
            "narrow": narrow.squeeze(-1),
            "split": split.squeeze(-1),
            "merge": merge.squeeze(-1),
            "suspend": suspend.squeeze(-1),
            "suspend_mass": suspend_mass.squeeze(-1),
            "active_ratio": active_ratio,
            "temperature": temperature.squeeze(-1),
            "schema_peak": max_weight.squeeze(-1),
        }
        for index, name in enumerate(self.anchor_names):
            stats[f"{name}_anchor_weight"] = anchor_weights[:, index]
        return stats
