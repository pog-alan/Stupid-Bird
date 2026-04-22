from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .hierarchical_context import HierarchicalContextSpec, MemoryLevel
from .signal_schema import DynamicSchemaConfig, DynamicSchemaOperator


@dataclass(frozen=True)
class SBCoreMiniTorchConfig:
    vocab_size: int
    d_model: int = 96
    state_dim: int = 128
    num_layers: int = 3
    semantic_memory_slots: int = 64
    working_memory_slots: int = 16
    episodic_memory_slots: int = 8
    episodic_key_slots: int = 4
    summary_memory_slots: int = 4
    scene_memory_slots: int = 2
    router_top_k: int = 4
    dropout: float = 0.1
    tie_weights: bool = False
    pad_token_id: int = 0
    max_seq_len: int = 256
    working_protection_decay: float = 0.96
    working_usage_decay: float = 0.92
    working_age_increment: float = 0.05
    working_memory_temperature: float = 0.35
    episodic_strength_decay: float = 0.99
    episodic_age_increment: float = 0.02
    episodic_memory_temperature: float = 0.25
    episodic_key_decay: float = 0.995
    episodic_key_age_increment: float = 0.01
    episodic_key_temperature: float = 0.18
    summary_strength_decay: float = 0.998
    summary_age_increment: float = 0.005
    summary_memory_temperature: float = 0.14
    scene_strength_decay: float = 0.999
    scene_age_increment: float = 0.002
    scene_memory_temperature: float = 0.11
    summary_buffer_decay: float = 0.92
    scene_buffer_decay: float = 0.96
    summary_boundary_threshold: float = 0.58
    scene_boundary_threshold: float = 0.72
    signal_abstraction_levels: int = 3
    signal_stop_threshold: float = 0.62
    signal_schema_slots: int = 7
    use_attention: bool = False
    use_kv_cache: bool = False

    def validate(self) -> None:
        if self.use_attention:
            raise ValueError("SB-Core-Mini 不能启用 self-attention。")
        if self.use_kv_cache:
            raise ValueError("SB-Core-Mini 不能依赖 KV cache。")
        if self.router_top_k <= 0:
            raise ValueError("router_top_k 必须大于 0。")
        if self.semantic_memory_slots <= 0 or self.working_memory_slots <= 0:
            raise ValueError("memory slots 必须大于 0。")
        if self.episodic_memory_slots <= 0:
            raise ValueError("episodic_memory_slots 必须大于 0。")
        if self.episodic_key_slots <= 0:
            raise ValueError("episodic_key_slots 必须大于 0。")
        if self.summary_memory_slots <= 0:
            raise ValueError("summary_memory_slots 必须大于 0。")
        if self.scene_memory_slots <= 0:
            raise ValueError("scene_memory_slots 必须大于 0。")
        if not 0.0 < self.working_protection_decay <= 1.0:
            raise ValueError("working_protection_decay 必须在 (0, 1] 之间。")
        if not 0.0 < self.working_usage_decay <= 1.0:
            raise ValueError("working_usage_decay 必须在 (0, 1] 之间。")
        if not 0.0 <= self.working_age_increment <= 1.0:
            raise ValueError("working_age_increment 必须在 [0, 1] 之间。")
        if self.working_memory_temperature <= 0.0:
            raise ValueError("working_memory_temperature 必须大于 0。")
        if not 0.0 < self.episodic_strength_decay <= 1.0:
            raise ValueError("episodic_strength_decay 必须在 (0, 1] 之间。")
        if not 0.0 <= self.episodic_age_increment <= 1.0:
            raise ValueError("episodic_age_increment 必须在 [0, 1] 之间。")
        if self.episodic_memory_temperature <= 0.0:
            raise ValueError("episodic_memory_temperature 必须大于 0。")
        if not 0.0 < self.episodic_key_decay <= 1.0:
            raise ValueError("episodic_key_decay 必须在 (0, 1] 之间。")
        if not 0.0 <= self.episodic_key_age_increment <= 1.0:
            raise ValueError("episodic_key_age_increment 必须在 [0, 1] 之间。")
        if self.episodic_key_temperature <= 0.0:
            raise ValueError("episodic_key_temperature 必须大于 0。")
        if not 0.0 < self.summary_strength_decay <= 1.0:
            raise ValueError("summary_strength_decay 必须在 (0, 1] 之间。")
        if not 0.0 <= self.summary_age_increment <= 1.0:
            raise ValueError("summary_age_increment 必须在 [0, 1] 之间。")
        if self.summary_memory_temperature <= 0.0:
            raise ValueError("summary_memory_temperature 必须大于 0。")
        if not 0.0 < self.scene_strength_decay <= 1.0:
            raise ValueError("scene_strength_decay 必须在 (0, 1] 之间。")
        if not 0.0 <= self.scene_age_increment <= 1.0:
            raise ValueError("scene_age_increment 必须在 [0, 1] 之间。")
        if self.scene_memory_temperature <= 0.0:
            raise ValueError("scene_memory_temperature 必须大于 0。")
        if not 0.0 < self.summary_buffer_decay <= 1.0:
            raise ValueError("summary_buffer_decay 必须在 (0, 1] 之间。")
        if not 0.0 < self.scene_buffer_decay <= 1.0:
            raise ValueError("scene_buffer_decay 必须在 (0, 1] 之间。")
        if not 0.0 < self.summary_boundary_threshold < 1.0:
            raise ValueError("summary_boundary_threshold 必须在 (0, 1) 之间。")
        if not 0.0 < self.scene_boundary_threshold < 1.0:
            raise ValueError("scene_boundary_threshold 必须在 (0, 1) 之间。")
        if self.signal_abstraction_levels <= 0:
            raise ValueError("signal_abstraction_levels 必须大于 0。")
        if not 0.0 < self.signal_stop_threshold < 1.0:
            raise ValueError("signal_stop_threshold 必须在 (0, 1) 之间。")
        if self.signal_schema_slots <= 0:
            raise ValueError("signal_schema_slots must be positive.")


@dataclass(frozen=True)
class SBRuntimeGates:
    summary_read: float = 1.0
    summary_write: float = 1.0
    scene_read: float = 1.0
    scene_write: float = 1.0
    drill: float = 1.0
    forgetting: float = 1.0

    def clamped(self) -> "SBRuntimeGates":
        return SBRuntimeGates(
            **{field.name: float(min(1.0, max(0.0, getattr(self, field.name)))) for field in fields(self)}
        )

    def as_dict(self) -> Dict[str, float]:
        return {field.name: float(getattr(self, field.name)) for field in fields(self)}


@dataclass(frozen=True)
class SBCoreMemoryState:
    hidden: List[Tensor]
    working_keys: List[Tensor]
    working_values: List[Tensor]
    working_protection: List[Tensor]
    working_usage: List[Tensor]
    working_age: List[Tensor]
    episodic_keys: List[Tensor]
    episodic_values: List[Tensor]
    episodic_strength: List[Tensor]
    episodic_age: List[Tensor]
    episodic_branch_mass: List[Tensor]
    episodic_schema_mass: List[Tensor]
    episodic_replay_hits: List[Tensor]
    episodic_cold_steps: List[Tensor]
    episodic_key_keys: List[Tensor]
    episodic_key_values: List[Tensor]
    episodic_key_strength: List[Tensor]
    episodic_key_age: List[Tensor]
    episodic_key_usage: List[Tensor]
    summary_keys: List[Tensor]
    summary_values: List[Tensor]
    summary_strength: List[Tensor]
    summary_age: List[Tensor]
    summary_branch_mass: List[Tensor]
    summary_schema_mass: List[Tensor]
    scene_keys: List[Tensor]
    scene_values: List[Tensor]
    scene_strength: List[Tensor]
    scene_age: List[Tensor]
    scene_branch_mass: List[Tensor]
    scene_schema_mass: List[Tensor]
    summary_buffer_state: List[Tensor]
    summary_buffer_branch: List[Tensor]
    summary_buffer_schema: List[Tensor]
    summary_buffer_mass: List[Tensor]
    scene_buffer_state: List[Tensor]
    scene_buffer_branch: List[Tensor]
    scene_buffer_schema: List[Tensor]
    scene_buffer_mass: List[Tensor]
    previous_branch: List[Tensor]
    previous_schema: List[Tensor]

    def detached(self) -> "SBCoreMemoryState":
        payload = {}
        for field in fields(self):
            payload[field.name] = [item.detach() for item in getattr(self, field.name)]
        return SBCoreMemoryState(**payload)

    def moved_to(self, device: torch.device | str) -> "SBCoreMemoryState":
        payload = {}
        for field in fields(self):
            payload[field.name] = [item.to(device) for item in getattr(self, field.name)]
        return SBCoreMemoryState(**payload)


def _lerp_gate(left: SBRuntimeGates, right: SBRuntimeGates, weight: float) -> SBRuntimeGates:
    alpha = float(min(1.0, max(0.0, weight)))
    blended = {
        field.name: (1.0 - alpha) * getattr(left, field.name) + alpha * getattr(right, field.name)
        for field in fields(SBRuntimeGates)
    }
    return SBRuntimeGates(**blended).clamped()


def staged_runtime_gates(step_index: int, total_steps: int) -> Tuple[str, SBRuntimeGates]:
    if total_steps <= 1:
        return "full_recall", SBRuntimeGates()

    progress = float(step_index) / float(max(total_steps - 1, 1))
    anchors = [
        (
            0.0,
            "foundation",
            SBRuntimeGates(
                summary_read=0.0,
                summary_write=0.15,
                scene_read=0.0,
                scene_write=0.0,
                drill=0.0,
                forgetting=0.20,
            ),
        ),
        (
            0.35,
            "summary_bootstrap",
            SBRuntimeGates(
                summary_read=0.55,
                summary_write=0.80,
                scene_read=0.0,
                scene_write=0.10,
                drill=0.0,
                forgetting=0.45,
            ),
        ),
        (
            0.70,
            "scene_binding",
            SBRuntimeGates(
                summary_read=1.0,
                summary_write=1.0,
                scene_read=0.60,
                scene_write=0.75,
                drill=0.35,
                forgetting=0.75,
            ),
        ),
        (
            1.0,
            "full_recall",
            SBRuntimeGates(
                summary_read=1.0,
                summary_write=1.0,
                scene_read=1.0,
                scene_write=1.0,
                drill=1.0,
                forgetting=1.0,
            ),
        ),
    ]

    for index in range(len(anchors) - 1):
        start_progress, _, start_gates = anchors[index]
        end_progress, end_name, end_gates = anchors[index + 1]
        if progress <= end_progress:
            local_weight = (progress - start_progress) / max(end_progress - start_progress, 1e-6)
            return end_name, _lerp_gate(start_gates, end_gates, local_weight)
    return anchors[-1][1], anchors[-1][2]


def runtime_device_report(device: str | torch.device) -> Dict[str, str | int | float | bool]:
    device_str = str(device)
    report: Dict[str, str | int | float | bool] = {
        "requested_device": device_str,
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu_used": device_str.startswith("cuda") and torch.cuda.is_available(),
    }
    if report["gpu_used"]:
        current_index = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(current_index)
        report.update(
            {
                "device_name": torch.cuda.get_device_name(current_index),
                "device_index": int(current_index),
                "total_memory_mb": round(properties.total_memory / (1024 * 1024), 2),
                "allocated_memory_mb": round(torch.cuda.memory_allocated(current_index) / (1024 * 1024), 2),
                "reserved_memory_mb": round(torch.cuda.memory_reserved(current_index) / (1024 * 1024), 2),
            }
        )
    else:
        report.update(
            {
                "device_name": "cpu",
                "device_index": -1,
                "total_memory_mb": 0.0,
                "allocated_memory_mb": 0.0,
                "reserved_memory_mb": 0.0,
            }
        )
    return report


class SBRecurrentCell(nn.Module):
    def __init__(self, state_dim: int) -> None:
        super().__init__()
        self.gate = nn.Linear(state_dim * 3, state_dim)
        self.candidate = nn.Linear(state_dim * 3, state_dim)
        self.output = nn.Linear(state_dim * 2, state_dim)
        self.norm = nn.LayerNorm(state_dim)

    def forward(self, current: Tensor, previous: Tensor, memory_read: Tensor) -> Tensor:
        joined = torch.cat([current, previous, memory_read], dim=-1)
        gate = torch.sigmoid(self.gate(joined))
        candidate = torch.tanh(self.candidate(joined))
        hidden = (1.0 - gate) * previous + gate * candidate
        projected = self.output(torch.cat([current, hidden], dim=-1))
        return self.norm(projected + current)


class SBSignalBranch(nn.Module):
    def __init__(self, state_dim: int, levels: int, stop_threshold: float) -> None:
        super().__init__()
        self.levels = levels
        self.stop_threshold = stop_threshold
        self.level_projections = nn.ModuleList(
            [nn.Linear(state_dim * 2, state_dim) for _ in range(levels)]
        )
        self.level_gates = nn.ModuleList(
            [nn.Linear(state_dim * 2, state_dim) for _ in range(levels)]
        )
        self.level_stop_heads = nn.ModuleList([nn.Linear(state_dim * 2, 1) for _ in range(levels)])
        self.level_norms = nn.ModuleList([nn.LayerNorm(state_dim) for _ in range(levels)])
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for gate_layer in self.level_gates:
            nn.init.zeros_(gate_layer.weight)
            nn.init.constant_(gate_layer.bias, -1.2)
        for stop_head in self.level_stop_heads:
            nn.init.zeros_(stop_head.weight)
            nn.init.zeros_(stop_head.bias)

    def forward(self, signal: Tensor, previous: Tensor) -> Tuple[Tensor, Dict[str, Tensor]]:
        current = signal
        accumulated = torch.zeros_like(signal)
        active_mass = torch.ones(signal.shape[0], 1, device=signal.device, dtype=signal.dtype)
        gate_values: List[Tensor] = []
        level_deltas: List[Tensor] = []
        adequacy_values: List[Tensor] = []
        stop_depth = torch.zeros(signal.shape[0], 1, device=signal.device, dtype=signal.dtype)
        stop_mass_total = torch.zeros(signal.shape[0], 1, device=signal.device, dtype=signal.dtype)

        for level_index, (projection, gate_layer, stop_head, norm) in enumerate(
            zip(self.level_projections, self.level_gates, self.level_stop_heads, self.level_norms),
            start=1,
        ):
            joined = torch.cat([current, previous], dim=-1)
            gate = torch.sigmoid(gate_layer(joined))
            candidate = torch.tanh(projection(joined))
            updated = norm(current + gate * (candidate - current))
            adequacy = torch.sigmoid(stop_head(torch.cat([updated, previous], dim=-1)))
            stop_prob = torch.sigmoid((adequacy - self.stop_threshold) * 10.0)
            stop_mass = active_mass * stop_prob
            accumulated = accumulated + stop_mass * updated
            active_mass = active_mass * (1.0 - stop_prob)
            stop_depth = stop_depth + stop_mass * float(level_index)
            stop_mass_total = stop_mass_total + stop_mass

            gate_values.append(gate.mean(dim=-1))
            level_deltas.append((updated - current).norm(dim=-1))
            adequacy_values.append(adequacy.squeeze(-1))
            current = updated

        accumulated = accumulated + active_mass * current
        stop_depth = stop_depth + active_mass * float(self.levels)

        stats = {
            "gate_mean": torch.stack(gate_values, dim=1).mean(dim=-1),
            "delta_mean": torch.stack(level_deltas, dim=1).mean(dim=-1),
            "stop_depth": stop_depth.squeeze(-1),
            "stopped_ratio": stop_mass_total.squeeze(-1),
            "adequacy_mean": torch.stack(adequacy_values, dim=1).mean(dim=-1),
        }
        return accumulated, stats


class SBSignalAbstraction(nn.Module):
    ANCHOR_NAMES = ("entity", "relation", "event")

    def __init__(
        self,
        state_dim: int,
        levels: int,
        stop_threshold: float,
        *,
        schema_slots: int,
    ) -> None:
        super().__init__()
        self.schema_slots = schema_slots
        self.branches = nn.ModuleList(
            [
                SBSignalBranch(
                    state_dim=state_dim,
                    levels=levels,
                    stop_threshold=stop_threshold,
                )
                for _ in range(schema_slots)
            ]
        )
        self.schema_operator = DynamicSchemaOperator(
            DynamicSchemaConfig(
                state_dim=state_dim,
                schema_slots=schema_slots,
                anchor_names=self.ANCHOR_NAMES,
            )
        )
        self.residual_scale = nn.Linear(state_dim * 2, 1)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.zeros_(self.residual_scale.weight)
        nn.init.constant_(self.residual_scale.bias, -1.5)

    def forward(self, signal: Tensor, previous: Tensor) -> Tuple[Tensor, Dict[str, Tensor]]:
        joined = torch.cat([signal, previous], dim=-1)
        schema_state = self.schema_operator(signal, previous)
        branch_weights = schema_state["schema_weights"]
        anchor_weights = schema_state["anchor_weights"]

        branch_outputs: List[Tensor] = []
        branch_gate_stats: List[Tensor] = []
        branch_delta_stats: List[Tensor] = []
        branch_depth_stats: List[Tensor] = []
        branch_stop_stats: List[Tensor] = []
        branch_adequacy_stats: List[Tensor] = []

        for branch in self.branches:
            branch_output, branch_stats = branch(signal, previous)
            branch_outputs.append(branch_output)
            branch_gate_stats.append(branch_stats["gate_mean"])
            branch_delta_stats.append(branch_stats["delta_mean"])
            branch_depth_stats.append(branch_stats["stop_depth"])
            branch_stop_stats.append(branch_stats["stopped_ratio"])
            branch_adequacy_stats.append(branch_stats["adequacy_mean"])

        stacked_outputs = torch.stack(branch_outputs, dim=1)
        residuals = stacked_outputs - signal.unsqueeze(1)
        residual_scale = torch.sigmoid(self.residual_scale(joined))
        schema_residual = schema_state["schema_embedding"] - signal
        fused_residual = torch.sum(residuals * branch_weights.unsqueeze(-1), dim=1) + 0.5 * schema_residual
        fused = signal + residual_scale * fused_residual

        branch_gate_tensor = torch.stack(branch_gate_stats, dim=1)
        branch_delta_tensor = torch.stack(branch_delta_stats, dim=1)
        branch_depth_tensor = torch.stack(branch_depth_stats, dim=1)
        branch_stop_tensor = torch.stack(branch_stop_stats, dim=1)
        branch_adequacy_tensor = torch.stack(branch_adequacy_stats, dim=1)

        stats = {
            "gate_mean": torch.sum(branch_gate_tensor * branch_weights, dim=1),
            "delta_mean": torch.sum(branch_delta_tensor * branch_weights, dim=1),
            "entropy": schema_state["schema_entropy"],
            "anchor_entropy": schema_state["anchor_entropy"],
            "schema_weights": branch_weights,
            "anchor_weights": anchor_weights,
            "stop_depth": torch.sum(branch_depth_tensor * branch_weights, dim=1),
            "stopped_ratio": torch.sum(branch_stop_tensor * branch_weights, dim=1),
            "adequacy_mean": torch.sum(branch_adequacy_tensor * branch_weights, dim=1),
            "entity_weight": anchor_weights[:, 0],
            "relation_weight": anchor_weights[:, 1],
            "event_weight": anchor_weights[:, 2],
            "schema_active_ratio": schema_state["active_ratio"],
            "schema_peak": schema_state["schema_peak"],
            "schema_widen_mean": schema_state["widen"],
            "schema_narrow_mean": schema_state["narrow"],
            "schema_split_mean": schema_state["split"],
            "schema_merge_mean": schema_state["merge"],
            "schema_suspend_mean": schema_state["suspend"],
            "schema_suspend_mass_mean": schema_state["suspend_mass"],
            "schema_temperature_mean": schema_state["temperature"],
            "residual_scale": residual_scale.squeeze(-1),
        }
        return fused, stats


class SBMemoryRouter(nn.Module):
    def __init__(self, state_dim: int, top_k: int) -> None:
        super().__init__()
        self.top_k = top_k
        self.query = nn.Linear(state_dim * 2, state_dim, bias=False)

    def forward(
        self,
        current: Tensor,
        previous: Tensor,
        working_keys: Tensor,
        working_values: Tensor,
        semantic_keys: Tensor,
        semantic_values: Tensor,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        batch_size, working_slots, state_dim = working_keys.shape
        semantic_slots = semantic_keys.shape[0]

        query = F.normalize(self.query(torch.cat([current, previous], dim=-1)), dim=-1, eps=1e-6)
        norm_working_keys = F.normalize(working_keys, dim=-1, eps=1e-6)
        norm_semantic_keys = F.normalize(semantic_keys, dim=-1, eps=1e-6)

        working_scores = torch.einsum("bd,bnd->bn", query, norm_working_keys)
        semantic_scores = torch.einsum("bd,nd->bn", query, norm_semantic_keys)
        scores = torch.cat([working_scores, semantic_scores], dim=-1)

        top_k = min(self.top_k, working_slots + semantic_slots)
        top_scores, top_indices = torch.topk(scores, k=top_k, dim=-1)
        weights = F.softmax(top_scores, dim=-1)

        semantic_values = semantic_values.unsqueeze(0).expand(batch_size, -1, -1)
        all_values = torch.cat([working_values, semantic_values], dim=1)
        gather_index = top_indices.unsqueeze(-1).expand(-1, -1, state_dim)
        selected_values = torch.gather(all_values, dim=1, index=gather_index)
        memory_read = torch.sum(selected_values * weights.unsqueeze(-1), dim=1)

        return memory_read, {
            "top_indices": top_indices,
            "top_scores": top_scores,
            "weights": weights,
            "working_ratio": (top_indices < working_slots).float().mean(),
        }


class SBMemoryWriter(nn.Module):
    def __init__(
        self,
        state_dim: int,
        working_slots: int,
        *,
        protection_decay: float,
        temperature: float,
    ) -> None:
        super().__init__()
        self.working_slots = working_slots
        self.protection_decay = protection_decay
        self.temperature = temperature
        self.key_proj = nn.Linear(state_dim, state_dim)
        self.value_proj = nn.Linear(state_dim, state_dim)
        self.write_gate = nn.Linear(state_dim, 1)
        self.merge_gate = nn.Linear(state_dim, 1)
        self.binding_gate = nn.Linear(state_dim, 1)
        self.importance_gate = nn.Linear(state_dim, 1)
        self.slot_occupancy = nn.Linear(state_dim, 1)
        self.slot_protection = nn.Linear(state_dim, 1)

    def forward(
        self,
        hidden: Tensor,
        working_keys: Tensor,
        working_values: Tensor,
        working_protection: Tensor,
        working_usage: Tensor,
        working_age: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Dict[str, Tensor]]:
        candidate_key = torch.tanh(self.key_proj(hidden))
        candidate_value = torch.tanh(self.value_proj(hidden))
        write_strength = torch.sigmoid(self.write_gate(hidden))
        importance = torch.sigmoid(self.importance_gate(hidden))

        norm_candidate = F.normalize(candidate_key, dim=-1, eps=1e-6)
        norm_keys = F.normalize(working_keys, dim=-1, eps=1e-6)
        similarity = torch.einsum("bd,bnd->bn", norm_candidate, norm_keys)

        norm_occupancy = torch.clamp(working_values.norm(dim=-1) / (working_values.shape[-1] ** 0.5), 0.0, 1.0)
        learned_occupancy = torch.sigmoid(self.slot_occupancy(working_values)).squeeze(-1)
        occupancy = torch.clamp(0.5 * learned_occupancy + 0.5 * norm_occupancy, 0.0, 1.0)
        learned_protection = torch.sigmoid(self.slot_protection(working_values)).squeeze(-1)
        effective_protection = torch.clamp(0.4 * learned_protection + 0.6 * working_protection, 0.0, 1.0)
        effective_usage = torch.clamp(0.5 * occupancy + 0.5 * working_usage, 0.0, 1.0)
        effective_age = torch.clamp(working_age, 0.0, 1.0)

        merge_weights = F.softmax(similarity / self.temperature, dim=-1)
        replace_scores = (
            1.15 * (1.0 - occupancy)
            + 0.85 * (1.0 - effective_protection)
            + 0.65 * effective_age
            + 0.45 * (1.0 - effective_usage)
            + 0.25 * (1.0 - similarity)
        )
        replace_weights = F.softmax(replace_scores / self.temperature, dim=-1)

        merge_index = merge_weights.argmax(dim=-1)
        replace_index = replace_weights.argmax(dim=-1)
        max_similarity = similarity.gather(1, merge_index.unsqueeze(-1))
        matched_occupancy = occupancy.gather(1, merge_index.unsqueeze(-1))
        matched_usage = effective_usage.gather(1, merge_index.unsqueeze(-1))
        matched_age = effective_age.gather(1, merge_index.unsqueeze(-1))
        merge_preference = torch.sigmoid(
            self.merge_gate(hidden)
            + 2.4 * max_similarity
            + 1.6 * (matched_occupancy - 0.5)
            + 1.0 * (matched_usage - 0.5)
            - 0.8 * matched_age
        )
        merge_candidate = ((max_similarity > 0.55) & (matched_occupancy > 0.35)).squeeze(-1)
        use_merge = ((merge_preference.squeeze(-1) >= 0.5) & merge_candidate).long()
        target_index = torch.where(use_merge.bool(), merge_index, replace_index)
        target_weights = F.one_hot(target_index, num_classes=working_keys.shape[1]).to(hidden.dtype)
        binding_strength = torch.sigmoid(self.binding_gate(hidden) + 2.2 * max_similarity)

        conflict = torch.clamp(1.0 - similarity, 0.0, 1.0)
        overwrite = (0.15 + 0.85 * write_strength) * target_weights * (1.0 - 0.65 * effective_protection * conflict)
        overwrite = overwrite.unsqueeze(-1)

        key_mix = torch.where(
            use_merge.unsqueeze(-1).bool(),
            0.22 + 0.38 * binding_strength,
            0.78 + 0.18 * binding_strength,
        ).unsqueeze(-1)
        value_mix = torch.where(
            use_merge.unsqueeze(-1).bool(),
            0.45 + 0.35 * importance,
            0.75 + 0.20 * importance,
        ).unsqueeze(-1)

        updated_keys = working_keys + overwrite * key_mix * (candidate_key.unsqueeze(1) - working_keys)
        updated_values = working_values + overwrite * value_mix * (candidate_value.unsqueeze(1) - working_values)

        protection_boost = overwrite.squeeze(-1) * (0.5 + 0.5 * importance)
        updated_protection = torch.clamp(
            working_protection * self.protection_decay + protection_boost,
            0.0,
            1.0,
        )

        stats = {
            "write_strength": write_strength.squeeze(-1),
            "merge_preference": merge_preference.squeeze(-1),
            "binding_strength": binding_strength.squeeze(-1),
            "overwrite_ratio": overwrite.squeeze(-1).mean(dim=-1),
            "protection_mean": updated_protection.mean(dim=-1),
            "max_similarity": max_similarity.squeeze(-1),
            "slot_write_mass": overwrite.squeeze(-1),
        }
        return updated_keys, updated_values, updated_protection, stats


class SBKeyCentricReplayQueryBuilder(nn.Module):
    def __init__(self, state_dim: int, schema_dim: int) -> None:
        super().__init__()
        self.schema_dim = schema_dim
        joined_dim = state_dim * 2 + 3 + schema_dim
        self.key_query = nn.Linear(joined_dim, state_dim, bias=False)
        self.branch_query = nn.Linear(joined_dim, 3)
        self.schema_query = nn.Linear(joined_dim, schema_dim)
        self.delay_gate = nn.Linear(joined_dim, 1)
        self.salience_gate = nn.Linear(joined_dim, 1)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.zeros_(self.key_query.weight)
        nn.init.zeros_(self.branch_query.weight)
        nn.init.zeros_(self.branch_query.bias)
        nn.init.zeros_(self.delay_gate.weight)
        nn.init.constant_(self.delay_gate.bias, -0.4)
        nn.init.zeros_(self.salience_gate.weight)
        nn.init.constant_(self.salience_gate.bias, 0.2)

    def forward(self, signal: Tensor, previous: Tensor, branch_hint: Tensor, schema_hint: Tensor) -> Dict[str, Tensor]:
        joined = torch.cat([signal, previous, branch_hint, schema_hint], dim=-1)
        projected_key = self.key_query(joined)
        residual_key = 0.65 * previous + 0.35 * signal
        mismatch = 1.0 - F.cosine_similarity(signal, previous, dim=-1).clamp(-1.0, 1.0)
        branch_logits = self.branch_query(joined) + 0.25 * branch_hint
        schema_logits = self.schema_query(joined) + 0.20 * schema_hint
        return {
            "key": F.normalize(projected_key + residual_key, dim=-1, eps=1e-6),
            "branch": F.softmax(branch_logits, dim=-1),
            "schema": F.softmax(schema_logits, dim=-1),
            "delay_gate": torch.sigmoid(self.delay_gate(joined).squeeze(-1) + 2.4 * (mismatch - 0.32)),
            "salience_gate": torch.sigmoid(self.salience_gate(joined).squeeze(-1) + 0.6 * branch_hint[:, 0]),
        }


class SBEpisodicMemory(nn.Module):
    def __init__(
        self,
        state_dim: int,
        slots: int,
        *,
        strength_decay: float,
        age_increment: float,
        temperature: float,
    ) -> None:
        super().__init__()
        self.slots = slots
        self.strength_decay = strength_decay
        self.age_increment = age_increment
        self.temperature = temperature
        self.read_query = nn.Linear(state_dim * 2, state_dim, bias=False)
        self.read_mix_gate = nn.Linear(state_dim * 2, 1)
        self.key_proj = nn.Linear(state_dim * 2, state_dim)
        self.value_proj = nn.Linear(state_dim * 2, state_dim)
        self.write_gate = nn.Linear(state_dim * 2, 1)
        self.persistence_gate = nn.Linear(state_dim * 2, 1)
        self.merge_gate = nn.Linear(state_dim * 2, 1)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.zeros_(self.read_query.weight)
        nn.init.zeros_(self.read_mix_gate.weight)
        nn.init.constant_(self.read_mix_gate.bias, -0.2)
        nn.init.zeros_(self.write_gate.weight)
        nn.init.constant_(self.write_gate.bias, -0.6)

    def read(
        self,
        signal: Tensor,
        previous: Tensor,
        episodic_keys: Tensor,
        episodic_values: Tensor,
        episodic_strength: Tensor,
        episodic_age: Tensor,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        query = F.normalize(self.read_query(torch.cat([signal, previous], dim=-1)), dim=-1, eps=1e-6)
        norm_keys = F.normalize(episodic_keys, dim=-1, eps=1e-6)
        content_scores = torch.einsum("bd,bnd->bn", query, norm_keys)
        content_weights = F.softmax(content_scores / self.temperature, dim=-1)
        content_read = torch.sum(episodic_values * content_weights.unsqueeze(-1), dim=1)

        persistent_scores = (2.4 * episodic_strength) + (0.6 * (1.0 - episodic_age))
        persistent_weights = F.softmax(persistent_scores / self.temperature, dim=-1)
        persistent_read = torch.sum(episodic_values * persistent_weights.unsqueeze(-1), dim=1)

        read_mix = torch.sigmoid(self.read_mix_gate(torch.cat([signal, previous], dim=-1)))
        memory_read = read_mix * content_read + (1.0 - read_mix) * persistent_read
        max_similarity = content_scores.max(dim=-1).values

        stats = {
            "read_mix": read_mix.squeeze(-1),
            "max_similarity": max_similarity,
            "strength_mean": episodic_strength.mean(dim=-1),
            "content_scores": content_scores,
        }
        return memory_read, stats

    def write(
        self,
        signal: Tensor,
        hidden: Tensor,
        episodic_keys: Tensor,
        episodic_values: Tensor,
        episodic_strength: Tensor,
        episodic_age: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Dict[str, Tensor]]:
        joined = torch.cat([signal, hidden], dim=-1)
        candidate_key = torch.tanh(self.key_proj(joined))
        candidate_value = torch.tanh(self.value_proj(joined))
        write_strength = torch.sigmoid(self.write_gate(joined))
        persistence = torch.sigmoid(self.persistence_gate(joined))

        norm_candidate = F.normalize(candidate_key, dim=-1, eps=1e-6)
        norm_keys = F.normalize(episodic_keys, dim=-1, eps=1e-6)
        similarity = torch.einsum("bd,bnd->bn", norm_candidate, norm_keys)
        top_merge_k = min(3, episodic_keys.shape[1])
        top_similarity, top_indices = torch.topk(similarity, k=top_merge_k, dim=-1)
        max_similarity = top_similarity[:, 0]
        merge_index = top_indices[:, 0]
        second_similarity = top_similarity[:, 1] if top_merge_k > 1 else torch.zeros_like(max_similarity)

        novelty = torch.clamp(1.0 - max_similarity, 0.0, 1.0).unsqueeze(-1)
        replace_scores = 1.2 * episodic_age + 1.0 * (1.0 - episodic_strength) + 0.5 * (1.0 - similarity)
        replace_index = replace_scores.argmax(dim=-1)
        merge_preference = torch.sigmoid(self.merge_gate(joined) + 2.6 * max_similarity.unsqueeze(-1))
        full_merge_mask = (max_similarity > 0.78) & (merge_preference.squeeze(-1) >= 0.55)
        multi_merge_mask = full_merge_mask & (second_similarity > 0.68)
        partial_merge_mask = (~multi_merge_mask) & (max_similarity > 0.64) & (second_similarity > 0.52)

        target_weights = F.one_hot(replace_index, num_classes=episodic_keys.shape[1]).to(signal.dtype)
        if top_merge_k > 1:
            partial_logits = top_similarity[:, :2] / self.temperature
            partial_weights = F.softmax(partial_logits, dim=-1)
            partial_target = torch.zeros_like(target_weights)
            partial_target.scatter_add_(1, top_indices[:, :2], partial_weights)
            multi_target = partial_target
        else:
            partial_target = target_weights
            multi_target = target_weights
        if top_merge_k > 2:
            multi_logits = top_similarity / self.temperature
            multi_weights = F.softmax(multi_logits, dim=-1)
            multi_target = torch.zeros_like(target_weights)
            multi_target.scatter_add_(1, top_indices, multi_weights)

        full_target = F.one_hot(merge_index, num_classes=episodic_keys.shape[1]).to(signal.dtype)
        target_weights = torch.where(full_merge_mask.unsqueeze(-1), full_target, target_weights)
        target_weights = torch.where(partial_merge_mask.unsqueeze(-1), partial_target, target_weights)
        target_weights = torch.where(multi_merge_mask.unsqueeze(-1), multi_target, target_weights)

        overwrite_scale = torch.where(
            multi_merge_mask.unsqueeze(-1),
            0.16 + 0.52 * write_strength,
            torch.where(
                partial_merge_mask.unsqueeze(-1),
                0.18 + 0.62 * write_strength,
                0.2 + 0.8 * write_strength,
            ),
        )
        overwrite = target_weights * overwrite_scale * (0.55 + 0.45 * novelty)
        overwrite = overwrite.unsqueeze(-1)

        merge_like_mask = full_merge_mask | partial_merge_mask | multi_merge_mask
        key_mix = torch.where(
            merge_like_mask.unsqueeze(-1).bool(),
            0.28 + 0.24 * persistence,
            0.78 + 0.16 * persistence,
        ).unsqueeze(-1)
        value_mix = torch.where(
            merge_like_mask.unsqueeze(-1).bool(),
            0.42 + 0.28 * persistence,
            0.82 + 0.12 * persistence,
        ).unsqueeze(-1)

        updated_keys = episodic_keys + overwrite * key_mix * (candidate_key.unsqueeze(1) - episodic_keys)
        updated_values = episodic_values + overwrite * value_mix * (candidate_value.unsqueeze(1) - episodic_values)

        strength_boost = overwrite.squeeze(-1) * (
            0.45 + 0.35 * persistence + 0.45 * novelty + 0.25 * write_strength
        )
        updated_strength = torch.clamp(
            episodic_strength * self.strength_decay + strength_boost,
            0.0,
            1.0,
        )
        updated_age = torch.clamp(
            (episodic_age + self.age_increment) * (1.0 - overwrite.squeeze(-1)),
            0.0,
            1.0,
        )

        stats = {
            "write_strength": write_strength.squeeze(-1),
            "persistence": persistence.squeeze(-1),
            "overwrite_ratio": overwrite.squeeze(-1).mean(dim=-1),
            "strength_mean": updated_strength.mean(dim=-1),
            "max_similarity": max_similarity,
            "multi_merge_ratio": multi_merge_mask.float(),
            "partial_merge_ratio": partial_merge_mask.float(),
            "slot_write_mass": overwrite.squeeze(-1),
        }
        return updated_keys, updated_values, updated_strength, updated_age, stats


class SBShortKeyMemory(nn.Module):
    def __init__(
        self,
        state_dim: int,
        slots: int,
        *,
        strength_decay: float,
        age_increment: float,
        temperature: float,
    ) -> None:
        super().__init__()
        self.slots = slots
        self.strength_decay = strength_decay
        self.age_increment = age_increment
        self.temperature = temperature
        self.key_proj = nn.Linear(state_dim * 2, state_dim)
        self.value_proj = nn.Linear(state_dim * 2, state_dim)
        self.focus_gate = nn.Linear(state_dim * 2 + 3, 1)
        self.persistence_gate = nn.Linear(state_dim * 2 + 3, 1)
        self.consolidation_gate = nn.Linear(state_dim * 2 + 3, 1)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.zeros_(self.focus_gate.weight)
        nn.init.constant_(self.focus_gate.bias, -0.5)
        nn.init.zeros_(self.persistence_gate.weight)
        nn.init.constant_(self.persistence_gate.bias, 0.3)
        nn.init.zeros_(self.consolidation_gate.weight)
        nn.init.constant_(self.consolidation_gate.bias, -0.2)

    def read(
        self,
        *,
        query_key: Tensor,
        delay_gate: Tensor,
        salience_gate: Tensor,
        short_keys: Tensor,
        short_values: Tensor,
        short_strength: Tensor,
        short_age: Tensor,
        short_usage: Tensor,
    ) -> Tuple[Tensor, Tensor, Dict[str, Tensor]]:
        norm_keys = F.normalize(short_keys, dim=-1, eps=1e-6)
        key_scores = torch.einsum("bd,bnd->bn", query_key, norm_keys)
        persistence = 2.8 * short_strength + 0.9 * short_usage + 0.6 * (1.0 - short_age)
        delay_match = delay_gate.unsqueeze(-1) * short_age
        salience = salience_gate.unsqueeze(-1) * short_strength
        scores = 0.65 * key_scores + 0.22 * persistence + 0.08 * delay_match + 0.05 * salience

        top_k = min(2, short_values.shape[1])
        top_scores, top_indices = torch.topk(scores, k=top_k, dim=-1)
        top_weights = F.softmax(top_scores / self.temperature, dim=-1)
        gather_index = top_indices.unsqueeze(-1).expand(-1, -1, short_values.shape[-1])
        selected_values = torch.gather(short_values, dim=1, index=gather_index)
        content_read = torch.sum(selected_values * top_weights.unsqueeze(-1), dim=1)

        persistent_scores = persistence + 0.15 * salience
        persistent_top_scores, persistent_top_indices = torch.topk(persistent_scores, k=top_k, dim=-1)
        persistent_top_weights = F.softmax(persistent_top_scores / self.temperature, dim=-1)
        persistent_gather_index = persistent_top_indices.unsqueeze(-1).expand(-1, -1, short_values.shape[-1])
        persistent_values = torch.gather(short_values, dim=1, index=persistent_gather_index)
        persistent_read = torch.sum(persistent_values * persistent_top_weights.unsqueeze(-1), dim=1)

        persistent_preference = torch.clamp(
            0.22 + 0.55 * delay_gate.unsqueeze(-1) + 0.15 * (1.0 - salience_gate.unsqueeze(-1)),
            0.0,
            1.0,
        )
        read = (1.0 - persistent_preference) * content_read + persistent_preference * persistent_read

        slot_hits = torch.zeros_like(short_strength)
        slot_hits.scatter_add_(1, top_indices, top_weights)
        slot_hits.scatter_add_(1, persistent_top_indices, 0.35 * persistent_top_weights)
        stats = {
            "score_mean": scores.mean(dim=-1),
            "active_ratio": (slot_hits > 0.0).float().mean(dim=-1),
            "strength_mean": short_strength.mean(dim=-1),
            "usage_mean": short_usage.mean(dim=-1),
            "persistent_preference": persistent_preference.squeeze(-1),
        }
        return read, slot_hits, stats

    def write(
        self,
        *,
        signal: Tensor,
        hidden: Tensor,
        branch_hint: Tensor,
        abstraction_entropy: Tensor,
        delay_gate: Tensor,
        episodic_keys: Tensor,
        episodic_values: Tensor,
        episodic_strength: Tensor,
        episodic_replay_hits: Tensor,
        episodic_age: Tensor,
        short_keys: Tensor,
        short_values: Tensor,
        short_strength: Tensor,
        short_age: Tensor,
        short_usage: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Dict[str, Tensor]]:
        joined = torch.cat([signal, hidden], dim=-1)
        routed = torch.cat([signal, hidden, branch_hint], dim=-1)
        base_key = torch.tanh(self.key_proj(joined))
        base_value = torch.tanh(self.value_proj(joined))

        episodic_priority = (
            0.45 * episodic_strength
            + 0.30 * (episodic_replay_hits / 6.0)
            + 0.15 * (1.0 - episodic_age)
            + 0.10 * torch.clamp(episodic_values.norm(dim=-1) / (episodic_values.shape[-1] ** 0.5), 0.0, 1.0)
        )
        source_index = episodic_priority.argmax(dim=-1)
        gather_index = source_index.view(-1, 1, 1).expand(-1, 1, episodic_keys.shape[-1])
        source_key = torch.gather(episodic_keys, dim=1, index=gather_index).squeeze(1)
        source_value = torch.gather(episodic_values, dim=1, index=gather_index).squeeze(1)
        source_confidence = episodic_priority.gather(1, source_index.unsqueeze(-1)).squeeze(-1)

        focus_base = torch.sigmoid(self.focus_gate(routed)).squeeze(-1)
        compactness = torch.sigmoid((0.72 - abstraction_entropy) * 5.5)
        consolidation = torch.sigmoid(
            self.consolidation_gate(routed).squeeze(-1) - 0.85 + 3.0 * (source_confidence - 0.65)
        )
        persistence = torch.sigmoid(self.persistence_gate(routed)).squeeze(-1)
        key_focus = torch.clamp(0.50 * focus_base + 0.35 * compactness + 0.15 * (1.0 - delay_gate), 0.0, 1.0)
        write_phase = torch.clamp(1.0 - 0.75 * delay_gate, 0.15, 1.0)

        mixed_key = (1.0 - consolidation.unsqueeze(-1)) * base_key + consolidation.unsqueeze(-1) * source_key
        mixed_value = (
            (1.0 - 0.35 * consolidation.unsqueeze(-1)) * base_value
            + 0.35 * consolidation.unsqueeze(-1) * source_value
        )
        candidate_key = F.normalize(mixed_key, dim=-1, eps=1e-6)
        candidate_value = torch.tanh(mixed_value)

        norm_keys = F.normalize(short_keys, dim=-1, eps=1e-6)
        similarity = torch.einsum("bd,bnd->bn", candidate_key, norm_keys)
        merge_index = similarity.argmax(dim=-1)
        max_similarity = similarity.gather(1, merge_index.unsqueeze(-1)).squeeze(-1)
        replace_scores = 1.3 * short_age + 1.0 * (1.0 - short_strength) + 0.9 * (1.0 - short_usage)
        replace_index = replace_scores.argmax(dim=-1)

        use_merge = max_similarity > 0.81
        target_index = torch.where(use_merge, merge_index, replace_index)
        target_weights = F.one_hot(target_index, num_classes=short_keys.shape[1]).to(signal.dtype)

        overwrite = (
            target_weights
            * (0.10 + 0.80 * key_focus.unsqueeze(-1))
            * (0.55 + 0.45 * compactness.unsqueeze(-1))
            * write_phase.unsqueeze(-1)
        )
        overwrite = overwrite.unsqueeze(-1)

        key_mix = torch.where(
            use_merge.unsqueeze(-1).bool(),
            0.18 + 0.24 * persistence.unsqueeze(-1),
            0.78 + 0.10 * persistence.unsqueeze(-1),
        ).unsqueeze(-1)
        value_mix = torch.where(
            use_merge.unsqueeze(-1).bool(),
            0.34 + 0.22 * persistence.unsqueeze(-1),
            0.82 + 0.10 * persistence.unsqueeze(-1),
        ).unsqueeze(-1)

        updated_keys = short_keys + overwrite * key_mix * (candidate_key.unsqueeze(1) - short_keys)
        updated_values = short_values + overwrite * value_mix * (candidate_value.unsqueeze(1) - short_values)

        strength_boost = overwrite.squeeze(-1) * (
            0.55 + 0.20 * key_focus.unsqueeze(-1) + 0.15 * persistence.unsqueeze(-1)
        )
        updated_strength = torch.clamp(short_strength * self.strength_decay + strength_boost, 0.0, 1.0)
        updated_usage = torch.clamp(
            short_usage * 0.96 + overwrite.squeeze(-1) * (0.60 + 0.40 * delay_gate.unsqueeze(-1)),
            0.0,
            1.0,
        )
        updated_age = torch.clamp((short_age + self.age_increment) * (1.0 - 0.85 * overwrite.squeeze(-1)), 0.0, 1.0)

        stats = {
            "key_focus": key_focus,
            "consolidation": consolidation,
            "overwrite_ratio": overwrite.squeeze(-1).mean(dim=-1),
            "strength_mean": updated_strength.mean(dim=-1),
            "usage_mean": updated_usage.mean(dim=-1),
        }
        return updated_keys, updated_values, updated_strength, updated_age, updated_usage, stats


class SBSummaryMemory(nn.Module):
    def __init__(
        self,
        state_dim: int,
        slots: int,
        schema_dim: int,
        *,
        strength_decay: float,
        age_increment: float,
        temperature: float,
    ) -> None:
        super().__init__()
        self.slots = slots
        self.strength_decay = strength_decay
        self.age_increment = age_increment
        self.temperature = temperature
        self.schema_dim = schema_dim
        joined_dim = state_dim * 2 + 3 + schema_dim
        self.key_proj = nn.Linear(joined_dim, state_dim)
        self.value_proj = nn.Linear(joined_dim, state_dim)
        self.merge_gate = nn.Linear(joined_dim, 1)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.zeros_(self.merge_gate.weight)
        nn.init.constant_(self.merge_gate.bias, 0.4)

    def read(
        self,
        *,
        query_key: Tensor,
        query_branch: Tensor,
        query_schema: Tensor,
        delay_gate: Tensor,
        summary_branch_mass: Tensor,
        summary_schema_mass: Tensor,
        summary_keys: Tensor,
        summary_values: Tensor,
        summary_strength: Tensor,
        summary_age: Tensor,
    ) -> Tuple[Tensor, Tensor, Dict[str, Tensor]]:
        norm_keys = F.normalize(summary_keys, dim=-1, eps=1e-6)
        key_scores = torch.einsum("bd,bnd->bn", query_key, norm_keys)
        branch_alignment = torch.sum(query_branch.unsqueeze(1) * summary_branch_mass, dim=-1)
        schema_alignment = torch.sum(query_schema.unsqueeze(1) * summary_schema_mass, dim=-1)
        persistence = 2.6 * summary_strength + 0.8 * (1.0 - summary_age)
        scores = (
            key_scores
            + 0.22 * branch_alignment
            + 0.18 * schema_alignment
            + 0.22 * persistence
            + 0.12 * delay_gate.unsqueeze(-1)
        )

        top_k = min(2, summary_values.shape[1])
        top_scores, top_indices = torch.topk(scores, k=top_k, dim=-1)
        top_weights = F.softmax(top_scores / self.temperature, dim=-1)
        gather_index = top_indices.unsqueeze(-1).expand(-1, -1, summary_values.shape[-1])
        selected_values = torch.gather(summary_values, dim=1, index=gather_index)
        read = torch.sum(selected_values * top_weights.unsqueeze(-1), dim=1)

        slot_hits = torch.zeros_like(summary_strength)
        slot_hits.scatter_add_(1, top_indices, top_weights)
        stats = {
            "score_mean": scores.mean(dim=-1),
            "active_ratio": (slot_hits > 0.0).float().mean(dim=-1),
            "strength_mean": summary_strength.mean(dim=-1),
            "branch_alignment": branch_alignment.mean(dim=-1),
            "schema_alignment": schema_alignment.mean(dim=-1),
        }
        return read, slot_hits, stats

    def write(
        self,
        *,
        signal: Tensor,
        hidden: Tensor,
        branch_hint: Tensor,
        schema_hint: Tensor,
        episodic_values: Tensor,
        episodic_strength: Tensor,
        short_values: Tensor,
        short_strength: Tensor,
        summary_keys: Tensor,
        summary_values: Tensor,
        summary_strength: Tensor,
        summary_age: Tensor,
        summary_branch_mass: Tensor,
        summary_schema_mass: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Dict[str, Tensor]]:
        joined = torch.cat([signal, hidden, branch_hint, schema_hint], dim=-1)
        epi_weights = F.softmax(episodic_strength / self.temperature, dim=-1)
        epi_value = torch.sum(episodic_values * epi_weights.unsqueeze(-1), dim=1)
        short_weights = F.softmax(short_strength / self.temperature, dim=-1)
        short_value = torch.sum(short_values * short_weights.unsqueeze(-1), dim=1)
        episodic_salience = episodic_strength.reshape(episodic_strength.shape[0], -1).mean(dim=1, keepdim=True)
        short_salience = short_strength.reshape(short_strength.shape[0], -1).mean(dim=1, keepdim=True)
        source_mix = torch.sigmoid((episodic_salience - short_salience) * 2.0)
        candidate_source = source_mix * epi_value + (1.0 - source_mix) * short_value

        candidate_key = F.normalize(
            torch.tanh(self.key_proj(joined)) + 0.45 * F.normalize(candidate_source, dim=-1, eps=1e-6),
            dim=-1,
            eps=1e-6,
        )
        candidate_value = torch.tanh(torch.tanh(self.value_proj(joined)) + 0.55 * candidate_source)

        norm_keys = F.normalize(summary_keys, dim=-1, eps=1e-6)
        similarity = torch.einsum("bd,bnd->bn", candidate_key, norm_keys)
        merge_index = similarity.argmax(dim=-1)
        max_similarity = similarity.gather(1, merge_index.unsqueeze(-1)).squeeze(-1)
        replace_scores = 1.2 * summary_age + 1.0 * (1.0 - summary_strength)
        replace_index = replace_scores.argmax(dim=-1)
        merge_preference = torch.sigmoid(self.merge_gate(joined).squeeze(-1) + 2.2 * (max_similarity - 0.55))
        use_merge = merge_preference > 0.5
        target_index = torch.where(use_merge, merge_index, replace_index)
        target_weights = F.one_hot(target_index, num_classes=summary_keys.shape[1]).to(signal.dtype)

        overwrite = target_weights * (0.08 + 0.52 * torch.sigmoid((episodic_strength.mean(dim=-1) + short_strength.mean(dim=-1)) * 1.8)).unsqueeze(-1)
        overwrite = overwrite.unsqueeze(-1)

        key_mix = torch.where(use_merge, 0.18, 0.64).view(-1, 1, 1)
        value_mix = torch.where(use_merge, 0.28, 0.72).view(-1, 1, 1)
        updated_keys = summary_keys + overwrite * key_mix * (candidate_key.unsqueeze(1) - summary_keys)
        updated_values = summary_values + overwrite * value_mix * (candidate_value.unsqueeze(1) - summary_values)
        updated_strength = torch.clamp(summary_strength * self.strength_decay + 0.35 * overwrite.squeeze(-1), 0.0, 1.0)
        updated_age = torch.clamp((summary_age + self.age_increment) * (1.0 - 0.60 * overwrite.squeeze(-1)), 0.0, 1.0)

        branch_write_mass = overwrite.squeeze(-1).unsqueeze(-1)
        updated_branch_mass = summary_branch_mass * (1.0 - branch_write_mass) + branch_write_mass * branch_hint.unsqueeze(1)
        updated_branch_mass = updated_branch_mass / updated_branch_mass.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        updated_schema_mass = (
            summary_schema_mass * (1.0 - branch_write_mass) + branch_write_mass * schema_hint.unsqueeze(1)
        )
        updated_schema_mass = updated_schema_mass / updated_schema_mass.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        stats = {
            "overwrite_ratio": overwrite.squeeze(-1).mean(dim=-1),
            "strength_mean": updated_strength.mean(dim=-1),
            "merge_preference": merge_preference,
        }
        return (
            updated_keys,
            updated_values,
            updated_strength,
            updated_age,
            updated_branch_mass,
            updated_schema_mass,
            stats,
        )


class SBMiniLayer(nn.Module):
    def __init__(
        self,
        state_dim: int,
        working_slots: int,
        top_k: int,
        episodic_slots: int,
        episodic_key_slots: int,
        summary_slots: int,
        scene_slots: int,
        *,
        protection_decay: float,
        temperature: float,
        episodic_strength_decay: float,
        episodic_age_increment: float,
        episodic_temperature: float,
        episodic_key_decay: float,
        episodic_key_age_increment: float,
        episodic_key_temperature: float,
        summary_strength_decay: float,
        summary_age_increment: float,
        summary_temperature: float,
        scene_strength_decay: float,
        scene_age_increment: float,
        scene_temperature: float,
        abstraction_levels: int,
        stop_threshold: float,
        schema_slots: int,
    ) -> None:
        super().__init__()
        self.abstraction = SBSignalAbstraction(
            state_dim=state_dim,
            levels=abstraction_levels,
            stop_threshold=stop_threshold,
            schema_slots=schema_slots,
        )
        self.router = SBMemoryRouter(state_dim=state_dim, top_k=top_k)
        self.replay_query_builder = SBKeyCentricReplayQueryBuilder(
            state_dim=state_dim,
            schema_dim=schema_slots,
        )
        self.episodic = SBEpisodicMemory(
            state_dim=state_dim,
            slots=episodic_slots,
            strength_decay=episodic_strength_decay,
            age_increment=episodic_age_increment,
            temperature=episodic_temperature,
        )
        self.short_key_memory = SBShortKeyMemory(
            state_dim=state_dim,
            slots=episodic_key_slots,
            strength_decay=episodic_key_decay,
            age_increment=episodic_key_age_increment,
            temperature=episodic_key_temperature,
        )
        self.summary_memory = SBSummaryMemory(
            state_dim=state_dim,
            slots=summary_slots,
            schema_dim=schema_slots,
            strength_decay=summary_strength_decay,
            age_increment=summary_age_increment,
            temperature=summary_temperature,
        )
        self.scene_memory = SBSummaryMemory(
            state_dim=state_dim,
            slots=scene_slots,
            schema_dim=schema_slots,
            strength_decay=scene_strength_decay,
            age_increment=scene_age_increment,
            temperature=scene_temperature,
        )
        self.memory_fusion = nn.Linear(state_dim * 2, 1)
        self.replay_fusion = nn.Linear(state_dim * 2, 1)
        self.short_key_fusion = nn.Linear(state_dim * 2, 1)
        self.summary_fusion = nn.Linear(state_dim * 2, 1)
        self.scene_fusion = nn.Linear(state_dim * 2, 1)
        self.drill_fusion = nn.Linear(state_dim * 2, 1)
        nn.init.zeros_(self.replay_fusion.weight)
        nn.init.constant_(self.replay_fusion.bias, 1.5)
        nn.init.zeros_(self.short_key_fusion.weight)
        nn.init.constant_(self.short_key_fusion.bias, 0.4)
        nn.init.zeros_(self.summary_fusion.weight)
        nn.init.constant_(self.summary_fusion.bias, -0.1)
        nn.init.zeros_(self.scene_fusion.weight)
        nn.init.constant_(self.scene_fusion.bias, -0.35)
        nn.init.zeros_(self.drill_fusion.weight)
        nn.init.constant_(self.drill_fusion.bias, -0.55)
        self.cell = SBRecurrentCell(state_dim=state_dim)
        self.writer = SBMemoryWriter(
            state_dim=state_dim,
            working_slots=working_slots,
            protection_decay=protection_decay,
            temperature=temperature,
        )


class SBCoreMiniLM(nn.Module):
    def __init__(self, config: SBCoreMiniTorchConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.context_spec = HierarchicalContextSpec()
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        self.input_proj = nn.Linear(config.d_model, config.state_dim)
        self.layers = nn.ModuleList(
            [
                SBMiniLayer(
                    state_dim=config.state_dim,
                    working_slots=config.working_memory_slots,
                    top_k=config.router_top_k,
                    episodic_slots=config.episodic_memory_slots,
                    episodic_key_slots=config.episodic_key_slots,
                    summary_slots=config.summary_memory_slots,
                    scene_slots=config.scene_memory_slots,
                    protection_decay=config.working_protection_decay,
                    temperature=config.working_memory_temperature,
                    episodic_strength_decay=config.episodic_strength_decay,
                    episodic_age_increment=config.episodic_age_increment,
                    episodic_temperature=config.episodic_memory_temperature,
                    episodic_key_decay=config.episodic_key_decay,
                    episodic_key_age_increment=config.episodic_key_age_increment,
                    episodic_key_temperature=config.episodic_key_temperature,
                    summary_strength_decay=config.summary_strength_decay,
                    summary_age_increment=config.summary_age_increment,
                    summary_temperature=config.summary_memory_temperature,
                    scene_strength_decay=config.scene_strength_decay,
                    scene_age_increment=config.scene_age_increment,
                    scene_temperature=config.scene_memory_temperature,
                    abstraction_levels=config.signal_abstraction_levels,
                    stop_threshold=config.signal_stop_threshold,
                    schema_slots=config.signal_schema_slots,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.semantic_keys = nn.Parameter(
            torch.randn(config.num_layers, config.semantic_memory_slots, config.state_dim) * 0.02
        )
        self.semantic_values = nn.Parameter(
            torch.randn(config.num_layers, config.semantic_memory_slots, config.state_dim) * 0.02
        )
        self.final_norm = nn.LayerNorm(config.state_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.lm_head = nn.Linear(config.state_dim, config.vocab_size, bias=False)
        self.runtime_gates = SBRuntimeGates()
        if config.tie_weights and config.d_model == config.state_dim:
            self.lm_head.weight = self.embedding.weight

    def set_runtime_gates(self, gates: SBRuntimeGates) -> None:
        self.runtime_gates = gates.clamped()

    def get_runtime_gates(self) -> Dict[str, float]:
        return self.runtime_gates.as_dict()

    def initial_memory_state(
        self,
        batch_size: int,
        device: torch.device | str,
    ) -> SBCoreMemoryState:
        return SBCoreMemoryState(
            hidden=[torch.zeros(batch_size, self.config.state_dim, device=device) for _ in range(self.config.num_layers)],
            working_keys=[
                torch.zeros(batch_size, self.config.working_memory_slots, self.config.state_dim, device=device)
                for _ in range(self.config.num_layers)
            ],
            working_values=[
                torch.zeros(batch_size, self.config.working_memory_slots, self.config.state_dim, device=device)
                for _ in range(self.config.num_layers)
            ],
            working_protection=[
                torch.zeros(batch_size, self.config.working_memory_slots, device=device)
                for _ in range(self.config.num_layers)
            ],
            working_usage=[
                torch.zeros(batch_size, self.config.working_memory_slots, device=device)
                for _ in range(self.config.num_layers)
            ],
            working_age=[
                torch.zeros(batch_size, self.config.working_memory_slots, device=device)
                for _ in range(self.config.num_layers)
            ],
            episodic_keys=[
                torch.zeros(batch_size, self.config.episodic_memory_slots, self.config.state_dim, device=device)
                for _ in range(self.config.num_layers)
            ],
            episodic_values=[
                torch.zeros(batch_size, self.config.episodic_memory_slots, self.config.state_dim, device=device)
                for _ in range(self.config.num_layers)
            ],
            episodic_strength=[
                torch.zeros(batch_size, self.config.episodic_memory_slots, device=device)
                for _ in range(self.config.num_layers)
            ],
            episodic_age=[
                torch.zeros(batch_size, self.config.episodic_memory_slots, device=device)
                for _ in range(self.config.num_layers)
            ],
            episodic_branch_mass=[
                torch.zeros(batch_size, self.config.episodic_memory_slots, 3, device=device)
                for _ in range(self.config.num_layers)
            ],
            episodic_schema_mass=[
                torch.zeros(
                    batch_size,
                    self.config.episodic_memory_slots,
                    self.config.signal_schema_slots,
                    device=device,
                )
                for _ in range(self.config.num_layers)
            ],
            episodic_replay_hits=[
                torch.zeros(batch_size, self.config.episodic_memory_slots, device=device)
                for _ in range(self.config.num_layers)
            ],
            episodic_cold_steps=[
                torch.zeros(batch_size, self.config.episodic_memory_slots, device=device)
                for _ in range(self.config.num_layers)
            ],
            episodic_key_keys=[
                torch.zeros(batch_size, self.config.episodic_key_slots, self.config.state_dim, device=device)
                for _ in range(self.config.num_layers)
            ],
            episodic_key_values=[
                torch.zeros(batch_size, self.config.episodic_key_slots, self.config.state_dim, device=device)
                for _ in range(self.config.num_layers)
            ],
            episodic_key_strength=[
                torch.zeros(batch_size, self.config.episodic_key_slots, device=device)
                for _ in range(self.config.num_layers)
            ],
            episodic_key_age=[
                torch.zeros(batch_size, self.config.episodic_key_slots, device=device)
                for _ in range(self.config.num_layers)
            ],
            episodic_key_usage=[
                torch.zeros(batch_size, self.config.episodic_key_slots, device=device)
                for _ in range(self.config.num_layers)
            ],
            summary_keys=[
                torch.zeros(batch_size, self.config.summary_memory_slots, self.config.state_dim, device=device)
                for _ in range(self.config.num_layers)
            ],
            summary_values=[
                torch.zeros(batch_size, self.config.summary_memory_slots, self.config.state_dim, device=device)
                for _ in range(self.config.num_layers)
            ],
            summary_strength=[
                torch.zeros(batch_size, self.config.summary_memory_slots, device=device)
                for _ in range(self.config.num_layers)
            ],
            summary_age=[
                torch.zeros(batch_size, self.config.summary_memory_slots, device=device)
                for _ in range(self.config.num_layers)
            ],
            summary_branch_mass=[
                torch.zeros(batch_size, self.config.summary_memory_slots, 3, device=device)
                for _ in range(self.config.num_layers)
            ],
            summary_schema_mass=[
                torch.zeros(
                    batch_size,
                    self.config.summary_memory_slots,
                    self.config.signal_schema_slots,
                    device=device,
                )
                for _ in range(self.config.num_layers)
            ],
            scene_keys=[
                torch.zeros(batch_size, self.config.scene_memory_slots, self.config.state_dim, device=device)
                for _ in range(self.config.num_layers)
            ],
            scene_values=[
                torch.zeros(batch_size, self.config.scene_memory_slots, self.config.state_dim, device=device)
                for _ in range(self.config.num_layers)
            ],
            scene_strength=[
                torch.zeros(batch_size, self.config.scene_memory_slots, device=device)
                for _ in range(self.config.num_layers)
            ],
            scene_age=[
                torch.zeros(batch_size, self.config.scene_memory_slots, device=device)
                for _ in range(self.config.num_layers)
            ],
            scene_branch_mass=[
                torch.zeros(batch_size, self.config.scene_memory_slots, 3, device=device)
                for _ in range(self.config.num_layers)
            ],
            scene_schema_mass=[
                torch.zeros(
                    batch_size,
                    self.config.scene_memory_slots,
                    self.config.signal_schema_slots,
                    device=device,
                )
                for _ in range(self.config.num_layers)
            ],
            summary_buffer_state=[
                torch.zeros(batch_size, self.config.state_dim, device=device) for _ in range(self.config.num_layers)
            ],
            summary_buffer_branch=[
                torch.zeros(batch_size, 3, device=device) for _ in range(self.config.num_layers)
            ],
            summary_buffer_schema=[
                torch.zeros(batch_size, self.config.signal_schema_slots, device=device)
                for _ in range(self.config.num_layers)
            ],
            summary_buffer_mass=[
                torch.zeros(batch_size, 1, device=device) for _ in range(self.config.num_layers)
            ],
            scene_buffer_state=[
                torch.zeros(batch_size, self.config.state_dim, device=device) for _ in range(self.config.num_layers)
            ],
            scene_buffer_branch=[
                torch.zeros(batch_size, 3, device=device) for _ in range(self.config.num_layers)
            ],
            scene_buffer_schema=[
                torch.zeros(batch_size, self.config.signal_schema_slots, device=device)
                for _ in range(self.config.num_layers)
            ],
            scene_buffer_mass=[
                torch.zeros(batch_size, 1, device=device) for _ in range(self.config.num_layers)
            ],
            previous_branch=[
                torch.full((batch_size, 3), 1.0 / 3.0, device=device) for _ in range(self.config.num_layers)
            ],
            previous_schema=[
                torch.full(
                    (batch_size, self.config.signal_schema_slots),
                    1.0 / float(self.config.signal_schema_slots),
                    device=device,
                )
                for _ in range(self.config.num_layers)
            ],
        )

    def _episodic_replay_read(
        self,
        *,
        query_key: Tensor,
        query_branch: Tensor,
        query_schema: Tensor,
        delay_gate: Tensor,
        salience_gate: Tensor,
        episodic_branch_mass: Tensor,
        episodic_schema_mass: Tensor,
        episodic_keys: Tensor,
        episodic_values: Tensor,
        episodic_strength: Tensor,
        episodic_age: Tensor,
    ) -> Tuple[Tensor, Tensor, Dict[str, Tensor]]:
        weights = self.context_spec.replay_component_weights()
        norm_keys = F.normalize(episodic_keys, dim=-1, eps=1e-6)
        key_scores = torch.einsum("bd,bnd->bn", query_key, norm_keys)
        entity_match = query_branch[:, :1] * episodic_branch_mass[:, :, 0]
        relation_match = query_branch[:, 1:2] * episodic_branch_mass[:, :, 1]
        event_match = query_branch[:, 2:3] * episodic_branch_mass[:, :, 2]
        relation_event_match = torch.maximum(relation_match, event_match)
        schema_match = torch.sum(query_schema.unsqueeze(1) * episodic_schema_mass, dim=-1)
        task_match = torch.sigmoid(key_scores)
        salience = salience_gate.unsqueeze(-1) * episodic_strength
        delay_match = delay_gate.unsqueeze(-1) * episodic_age
        replay_scores = (
            0.88 * weights["task"] * task_match
            + 0.62 * weights["entity"] * entity_match
            + 0.55 * weights["relation_event"] * relation_event_match
            + 0.32 * schema_match
            + weights["salience"] * salience
            + 0.12 * delay_match
            + self.context_spec.replay_level_priority(MemoryLevel.EPISODIC)
        )
        top_k = min(self.config.router_top_k, episodic_values.shape[1])
        top_scores, top_indices = torch.topk(replay_scores, k=top_k, dim=-1)
        top_weights = F.softmax(top_scores / self.config.episodic_memory_temperature, dim=-1)
        gather_index = top_indices.unsqueeze(-1).expand(-1, -1, episodic_values.shape[-1])
        selected_values = torch.gather(episodic_values, dim=1, index=gather_index)
        replay_read = torch.sum(selected_values * top_weights.unsqueeze(-1), dim=1)
        slot_replay_mass = torch.zeros_like(episodic_strength)
        slot_replay_mass.scatter_add_(1, top_indices, top_weights)
        stats = {
            "score_mean": replay_scores.mean(dim=-1),
            "active_ratio": (slot_replay_mass > 0.0).float().mean(dim=-1),
            "top_score_mean": top_scores.mean(dim=-1),
            "delay_gate": delay_gate,
            "branch_alignment": relation_event_match.mean(dim=-1),
            "schema_alignment": schema_match.mean(dim=-1),
        }
        return replay_read, slot_replay_mass, stats

    def _summary_guided_drill_read(
        self,
        *,
        query_key: Tensor,
        query_branch: Tensor,
        query_schema: Tensor,
        delay_gate: Tensor,
        salience_gate: Tensor,
        summary_read: Tensor,
        scene_read: Tensor,
        summary_buffer_state: Tensor,
        scene_buffer_state: Tensor,
        summary_buffer_mass: Tensor,
        scene_buffer_mass: Tensor,
        summary_hits: Tensor,
        scene_hits: Tensor,
        summary_strength: Tensor,
        scene_strength: Tensor,
        episodic_branch_mass: Tensor,
        episodic_schema_mass: Tensor,
        episodic_keys: Tensor,
        episodic_values: Tensor,
        episodic_strength: Tensor,
        episodic_age: Tensor,
    ) -> Tuple[Tensor, Tensor, Dict[str, Tensor]]:
        norm_keys = F.normalize(episodic_keys, dim=-1, eps=1e-6)
        norm_values = F.normalize(episodic_values, dim=-1, eps=1e-6)

        summary_buffer_weight = torch.clamp(summary_buffer_mass / 3.0, 0.0, 1.0)
        scene_buffer_weight = torch.clamp(scene_buffer_mass / 2.0, 0.0, 1.0)
        blended_summary = (
            (1.0 - summary_buffer_weight) * summary_read + summary_buffer_weight * summary_buffer_state
        )
        blended_scene = (1.0 - scene_buffer_weight) * scene_read + scene_buffer_weight * scene_buffer_state
        summary_proto = F.normalize(blended_summary, dim=-1, eps=1e-6)
        scene_proto = F.normalize(blended_scene, dim=-1, eps=1e-6)
        summary_alignment = torch.einsum("bd,bnd->bn", summary_proto, norm_values)
        scene_alignment = torch.einsum("bd,bnd->bn", scene_proto, norm_values)

        branch_alignment = torch.sum(query_branch.unsqueeze(1) * episodic_branch_mass, dim=-1)
        schema_alignment = torch.sum(query_schema.unsqueeze(1) * episodic_schema_mass, dim=-1)
        persistence = torch.clamp(0.65 * episodic_strength + 0.35 * (1.0 - episodic_age), 0.0, 1.0)
        summary_support = torch.clamp(
            (summary_hits * summary_strength).sum(dim=-1, keepdim=True) + 0.35 * summary_buffer_weight,
            0.0,
            1.0,
        )
        scene_support = torch.clamp(
            (scene_hits * scene_strength).sum(dim=-1, keepdim=True) + 0.40 * scene_buffer_weight,
            0.0,
            1.0,
        )
        refined_query = F.normalize(
            (1.0 - 0.40 * summary_support - 0.28 * scene_support) * query_key
            + 0.40 * summary_support * summary_proto
            + 0.28 * scene_support * scene_proto,
            dim=-1,
            eps=1e-6,
        )
        query_scores = torch.einsum("bd,bnd->bn", refined_query, norm_keys)
        drill_trigger = torch.clamp(
            0.46 * delay_gate.unsqueeze(-1)
            + 0.16 * salience_gate.unsqueeze(-1)
            + 0.18 * summary_support
            + 0.14 * scene_support
            + 0.12 * query_branch[:, 2:3],
            0.0,
            1.0,
        )

        drill_scores = (
            0.32 * torch.sigmoid(2.2 * query_scores)
            + 0.14 * branch_alignment
            + 0.12 * schema_alignment
            + 0.18 * torch.sigmoid(2.0 * summary_alignment)
            + 0.14 * torch.sigmoid(2.0 * scene_alignment)
            + 0.16 * persistence
            + 0.08 * drill_trigger
            + 0.06 * self.context_spec.replay_level_priority(MemoryLevel.SUMMARY_EPISODE)
        )
        top_k = min(max(2, self.config.router_top_k + 1), episodic_values.shape[1])
        top_scores, top_indices = torch.topk(drill_scores, k=top_k, dim=-1)
        top_weights = F.softmax(top_scores / self.config.episodic_memory_temperature, dim=-1)
        gather_index = top_indices.unsqueeze(-1).expand(-1, -1, episodic_values.shape[-1])
        selected_values = torch.gather(episodic_values, dim=1, index=gather_index)
        drill_read = torch.sum(selected_values * top_weights.unsqueeze(-1), dim=1)

        slot_reopen_mass = torch.zeros_like(episodic_strength)
        slot_reopen_mass.scatter_add_(1, top_indices, top_weights * drill_trigger)
        stats = {
            "score_mean": drill_scores.mean(dim=-1),
            "active_ratio": (slot_reopen_mass > 0.0).float().mean(dim=-1),
            "trigger": drill_trigger.squeeze(-1),
            "alignment_mean": (
                0.46 * summary_alignment + 0.28 * scene_alignment + 0.26 * schema_alignment
            ).mean(dim=-1),
        }
        return drill_read, slot_reopen_mass, stats

    def _apply_episodic_forgetting(
        self,
        *,
        episodic_values: Tensor,
        episodic_strength: Tensor,
        episodic_age: Tensor,
        episodic_replay_hits: Tensor,
        episodic_cold_steps: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Dict[str, Tensor]]:
        policy = self.context_spec.config.forgetting_policy
        stability = torch.clamp(0.6 * episodic_strength + 0.4 * (1.0 - episodic_age), 0.0, 1.0)
        replay_boost = torch.clamp(episodic_replay_hits, 0.0, 6.0) * policy.replay_protect_gain
        retention = torch.clamp(
            policy.base_decay
            + 0.45 * episodic_strength
            + 0.30 * stability
            + replay_boost
            - policy.age_penalty * episodic_age,
            0.0,
            1.0,
        )

        keep_mask = retention > policy.cold_threshold
        cool_mask = (~keep_mask) & (retention > policy.fade_threshold)
        fade_mask = (~keep_mask) & (~cool_mask) & (retention > policy.archive_threshold)
        archive_mask = (~keep_mask) & (~cool_mask) & (~fade_mask)
        prune_mask = archive_mask & ((episodic_cold_steps + 1.0) >= float(policy.prune_after_steps))
        archive_only_mask = archive_mask & (~prune_mask)

        strength_scale = torch.ones_like(retention)
        value_scale = torch.ones_like(retention)
        age_boost = torch.zeros_like(retention)

        strength_scale = torch.where(cool_mask, torch.full_like(strength_scale, policy.cool_strength_scale), strength_scale)
        strength_scale = torch.where(fade_mask, torch.full_like(strength_scale, policy.fade_strength_scale), strength_scale)
        strength_scale = torch.where(
            archive_only_mask,
            torch.full_like(strength_scale, policy.archive_strength_scale),
            strength_scale,
        )
        strength_scale = torch.where(prune_mask, torch.zeros_like(strength_scale), strength_scale)

        value_scale = torch.where(cool_mask, torch.full_like(value_scale, policy.cool_value_scale), value_scale)
        value_scale = torch.where(fade_mask, torch.full_like(value_scale, policy.fade_value_scale), value_scale)
        value_scale = torch.where(
            archive_only_mask,
            torch.full_like(value_scale, policy.archive_value_scale),
            value_scale,
        )
        value_scale = torch.where(prune_mask, torch.zeros_like(value_scale), value_scale)

        age_boost = torch.where(cool_mask, torch.full_like(age_boost, policy.cool_age_boost), age_boost)
        age_boost = torch.where(fade_mask, torch.full_like(age_boost, policy.fade_age_boost), age_boost)
        age_boost = torch.where(
            archive_only_mask,
            torch.full_like(age_boost, policy.archive_age_boost),
            age_boost,
        )

        updated_strength = torch.clamp(episodic_strength * strength_scale, 0.0, 1.0)
        updated_values = episodic_values * value_scale.unsqueeze(-1)
        updated_age = torch.clamp(episodic_age + age_boost, 0.0, 1.0)
        next_cold_steps = torch.where(
            keep_mask,
            torch.clamp(episodic_cold_steps - 1.0, min=0.0),
            episodic_cold_steps + 1.0,
        )
        next_cold_steps = torch.where(prune_mask, torch.zeros_like(next_cold_steps), next_cold_steps)

        stats = {
            "retention_mean": retention.mean(dim=-1),
            "cool_ratio": cool_mask.float().mean(dim=-1),
            "fade_ratio": fade_mask.float().mean(dim=-1),
            "archive_ratio": archive_only_mask.float().mean(dim=-1),
            "prune_ratio": prune_mask.float().mean(dim=-1),
        }
        return updated_values, updated_strength, updated_age, next_cold_steps, stats

    def forward(
        self,
        input_ids: Tensor,
        return_aux: bool = True,
        memory_state: SBCoreMemoryState | None = None,
        return_state: bool = False,
    ) -> Dict[str, Tensor | Dict[str, float] | SBCoreMemoryState]:
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        gates = self.runtime_gates
        if seq_len > self.config.max_seq_len:
            raise ValueError(f"seq_len={seq_len} 超过 max_seq_len={self.config.max_seq_len}")
        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        x = self.embedding(input_ids) + self.position_embedding(positions)
        x = self.input_proj(x)

        state = memory_state if memory_state is not None else self.initial_memory_state(batch_size, device)
        hidden = state.hidden
        working_keys = state.working_keys
        working_values = state.working_values
        working_protection = state.working_protection
        working_usage = state.working_usage
        working_age = state.working_age
        episodic_keys = state.episodic_keys
        episodic_values = state.episodic_values
        episodic_strength = state.episodic_strength
        episodic_age = state.episodic_age
        episodic_branch_mass = state.episodic_branch_mass
        episodic_schema_mass = state.episodic_schema_mass
        episodic_replay_hits = state.episodic_replay_hits
        episodic_cold_steps = state.episodic_cold_steps
        episodic_key_keys = state.episodic_key_keys
        episodic_key_values = state.episodic_key_values
        episodic_key_strength = state.episodic_key_strength
        episodic_key_age = state.episodic_key_age
        episodic_key_usage = state.episodic_key_usage
        summary_keys = state.summary_keys
        summary_values = state.summary_values
        summary_strength = state.summary_strength
        summary_age = state.summary_age
        summary_branch_mass = state.summary_branch_mass
        summary_schema_mass = state.summary_schema_mass
        scene_keys = state.scene_keys
        scene_values = state.scene_values
        scene_strength = state.scene_strength
        scene_age = state.scene_age
        scene_branch_mass = state.scene_branch_mass
        scene_schema_mass = state.scene_schema_mass
        summary_buffer_state = state.summary_buffer_state
        summary_buffer_branch = state.summary_buffer_branch
        summary_buffer_schema = state.summary_buffer_schema
        summary_buffer_mass = state.summary_buffer_mass
        scene_buffer_state = state.scene_buffer_state
        scene_buffer_branch = state.scene_buffer_branch
        scene_buffer_schema = state.scene_buffer_schema
        scene_buffer_mass = state.scene_buffer_mass
        previous_branch = state.previous_branch
        previous_schema = state.previous_schema

        outputs: List[Tensor] = []
        route_entropy = x.new_tensor(0.0)
        working_ratio = x.new_tensor(0.0)
        write_gate_mean = x.new_tensor(0.0)
        merge_preference_mean = x.new_tensor(0.0)
        binding_strength_mean = x.new_tensor(0.0)
        episodic_read_mix_mean = x.new_tensor(0.0)
        episodic_write_gate_mean = x.new_tensor(0.0)
        episodic_persistence_mean = x.new_tensor(0.0)
        episodic_overwrite_ratio_mean = x.new_tensor(0.0)
        episodic_strength_mean = x.new_tensor(0.0)
        episodic_similarity_mean = x.new_tensor(0.0)
        episodic_replay_score_mean = x.new_tensor(0.0)
        episodic_replay_active_ratio_mean = x.new_tensor(0.0)
        episodic_replay_delay_mean = x.new_tensor(0.0)
        episodic_replay_branch_alignment_mean = x.new_tensor(0.0)
        episodic_replay_schema_alignment_mean = x.new_tensor(0.0)
        episodic_retention_mean = x.new_tensor(0.0)
        episodic_cool_ratio_mean = x.new_tensor(0.0)
        episodic_fade_ratio_mean = x.new_tensor(0.0)
        episodic_archive_ratio_mean = x.new_tensor(0.0)
        episodic_prune_ratio_mean = x.new_tensor(0.0)
        episodic_multi_merge_ratio_mean = x.new_tensor(0.0)
        episodic_partial_merge_ratio_mean = x.new_tensor(0.0)
        episodic_key_focus_mean = x.new_tensor(0.0)
        episodic_key_consolidation_mean = x.new_tensor(0.0)
        episodic_key_overwrite_ratio_mean = x.new_tensor(0.0)
        episodic_key_strength_mean = x.new_tensor(0.0)
        episodic_key_usage_mean = x.new_tensor(0.0)
        episodic_key_read_score_mean = x.new_tensor(0.0)
        episodic_key_read_active_ratio_mean = x.new_tensor(0.0)
        episodic_key_persistent_preference_mean = x.new_tensor(0.0)
        summary_read_score_mean = x.new_tensor(0.0)
        summary_read_active_ratio_mean = x.new_tensor(0.0)
        summary_strength_mean = x.new_tensor(0.0)
        summary_branch_alignment_mean = x.new_tensor(0.0)
        summary_schema_alignment_mean = x.new_tensor(0.0)
        summary_overwrite_ratio_mean = x.new_tensor(0.0)
        summary_merge_preference_mean = x.new_tensor(0.0)
        scene_read_score_mean = x.new_tensor(0.0)
        scene_read_active_ratio_mean = x.new_tensor(0.0)
        scene_strength_mean = x.new_tensor(0.0)
        scene_branch_alignment_mean = x.new_tensor(0.0)
        scene_schema_alignment_mean = x.new_tensor(0.0)
        scene_overwrite_ratio_mean = x.new_tensor(0.0)
        scene_merge_preference_mean = x.new_tensor(0.0)
        drill_score_mean = x.new_tensor(0.0)
        drill_active_ratio_mean = x.new_tensor(0.0)
        drill_trigger_mean = x.new_tensor(0.0)
        drill_alignment_mean = x.new_tensor(0.0)
        summary_boundary_mean = x.new_tensor(0.0)
        summary_flush_ratio_mean = x.new_tensor(0.0)
        scene_boundary_mean = x.new_tensor(0.0)
        scene_flush_ratio_mean = x.new_tensor(0.0)
        abstraction_gate_mean = x.new_tensor(0.0)
        abstraction_delta_mean = x.new_tensor(0.0)
        abstraction_entropy_mean = x.new_tensor(0.0)
        abstraction_anchor_entropy_mean = x.new_tensor(0.0)
        abstraction_stop_depth_mean = x.new_tensor(0.0)
        abstraction_stopped_ratio_mean = x.new_tensor(0.0)
        abstraction_adequacy_mean = x.new_tensor(0.0)
        abstraction_entity_weight_mean = x.new_tensor(0.0)
        abstraction_relation_weight_mean = x.new_tensor(0.0)
        abstraction_event_weight_mean = x.new_tensor(0.0)
        abstraction_schema_active_ratio_mean = x.new_tensor(0.0)
        abstraction_schema_peak_mean = x.new_tensor(0.0)
        abstraction_schema_widen_mean = x.new_tensor(0.0)
        abstraction_schema_narrow_mean = x.new_tensor(0.0)
        abstraction_schema_split_mean = x.new_tensor(0.0)
        abstraction_schema_merge_mean = x.new_tensor(0.0)
        abstraction_schema_suspend_mean = x.new_tensor(0.0)
        abstraction_schema_suspend_mass_mean = x.new_tensor(0.0)
        abstraction_schema_temperature_mean = x.new_tensor(0.0)
        abstraction_residual_scale_mean = x.new_tensor(0.0)
        overwrite_ratio_mean = x.new_tensor(0.0)
        protection_mean = x.new_tensor(0.0)
        usage_mean = x.new_tensor(0.0)
        age_mean = x.new_tensor(0.0)
        total_steps = 0

        for step in range(seq_len):
            current = x[:, step, :]
            for layer_index, layer in enumerate(self.layers):
                abstract_signal, abstraction_stats = layer.abstraction(current, hidden[layer_index])
                current_branch = torch.stack(
                    [
                        abstraction_stats["entity_weight"],
                        abstraction_stats["relation_weight"],
                        abstraction_stats["event_weight"],
                    ],
                    dim=-1,
                )
                current_schema = abstraction_stats["schema_weights"]
                cell_input = 0.5 * (current + abstract_signal)
                routed_read, route = layer.router(
                    current=abstract_signal,
                    previous=hidden[layer_index],
                    working_keys=working_keys[layer_index],
                    working_values=working_values[layer_index],
                    semantic_keys=self.semantic_keys[layer_index],
                    semantic_values=self.semantic_values[layer_index],
                )
                episodic_read, episodic_read_stats = layer.episodic.read(
                    signal=abstract_signal,
                    previous=hidden[layer_index],
                    episodic_keys=episodic_keys[layer_index],
                    episodic_values=episodic_values[layer_index],
                    episodic_strength=episodic_strength[layer_index],
                    episodic_age=episodic_age[layer_index],
                )
                replay_query = layer.replay_query_builder(
                    signal=abstract_signal,
                    previous=hidden[layer_index],
                    branch_hint=current_branch,
                    schema_hint=current_schema,
                )
                entropy_confidence = torch.clamp(
                    1.0 - abstraction_stats["entropy"].unsqueeze(-1) / 1.0986123,
                    0.0,
                    1.0,
                )
                branch_shift = 0.5 * torch.abs(current_branch - previous_branch[layer_index]).sum(dim=-1, keepdim=True)
                schema_shift = 0.5 * torch.abs(current_schema - previous_schema[layer_index]).sum(
                    dim=-1,
                    keepdim=True,
                )
                replay_read, slot_replay_mass, replay_stats = self._episodic_replay_read(
                    query_key=replay_query["key"],
                    query_branch=replay_query["branch"],
                    query_schema=replay_query["schema"],
                    delay_gate=replay_query["delay_gate"],
                    salience_gate=replay_query["salience_gate"],
                    episodic_branch_mass=episodic_branch_mass[layer_index],
                    episodic_schema_mass=episodic_schema_mass[layer_index],
                    episodic_keys=episodic_keys[layer_index],
                    episodic_values=episodic_values[layer_index],
                    episodic_strength=episodic_strength[layer_index],
                    episodic_age=episodic_age[layer_index],
                )
                short_key_read, short_key_hits, short_key_read_stats = layer.short_key_memory.read(
                    query_key=replay_query["key"],
                    delay_gate=replay_query["delay_gate"],
                    salience_gate=replay_query["salience_gate"],
                    short_keys=episodic_key_keys[layer_index],
                    short_values=episodic_key_values[layer_index],
                    short_strength=episodic_key_strength[layer_index],
                    short_age=episodic_key_age[layer_index],
                    short_usage=episodic_key_usage[layer_index],
                )
                summary_read, summary_hits, summary_read_stats = layer.summary_memory.read(
                    query_key=replay_query["key"],
                    query_branch=replay_query["branch"],
                    query_schema=replay_query["schema"],
                    delay_gate=replay_query["delay_gate"],
                    summary_branch_mass=summary_branch_mass[layer_index],
                    summary_schema_mass=summary_schema_mass[layer_index],
                    summary_keys=summary_keys[layer_index],
                    summary_values=summary_values[layer_index],
                    summary_strength=summary_strength[layer_index],
                    summary_age=summary_age[layer_index],
                )
                scene_read, scene_hits, scene_read_stats = layer.scene_memory.read(
                    query_key=replay_query["key"],
                    query_branch=replay_query["branch"],
                    query_schema=replay_query["schema"],
                    delay_gate=replay_query["delay_gate"],
                    summary_branch_mass=scene_branch_mass[layer_index],
                    summary_schema_mass=scene_schema_mass[layer_index],
                    summary_keys=scene_keys[layer_index],
                    summary_values=scene_values[layer_index],
                    summary_strength=scene_strength[layer_index],
                    summary_age=scene_age[layer_index],
                )
                drill_read, drill_hits, drill_stats = self._summary_guided_drill_read(
                    query_key=replay_query["key"],
                    query_branch=replay_query["branch"],
                    query_schema=replay_query["schema"],
                    delay_gate=replay_query["delay_gate"],
                    salience_gate=replay_query["salience_gate"],
                    summary_read=summary_read,
                    scene_read=scene_read,
                    summary_buffer_state=summary_buffer_state[layer_index],
                    scene_buffer_state=scene_buffer_state[layer_index],
                    summary_buffer_mass=summary_buffer_mass[layer_index],
                    scene_buffer_mass=scene_buffer_mass[layer_index],
                    summary_hits=summary_hits,
                    scene_hits=scene_hits,
                    summary_strength=summary_strength[layer_index],
                    scene_strength=scene_strength[layer_index],
                    episodic_branch_mass=episodic_branch_mass[layer_index],
                    episodic_schema_mass=episodic_schema_mass[layer_index],
                    episodic_keys=episodic_keys[layer_index],
                    episodic_values=episodic_values[layer_index],
                    episodic_strength=episodic_strength[layer_index],
                    episodic_age=episodic_age[layer_index],
                )
                memory_mix = torch.sigmoid(
                    layer.memory_fusion(torch.cat([abstract_signal, hidden[layer_index]], dim=-1))
                    - 1.25 * replay_query["delay_gate"].unsqueeze(-1)
                )
                replay_mix = torch.sigmoid(
                    layer.replay_fusion(torch.cat([abstract_signal, hidden[layer_index]], dim=-1))
                    - 1.6 * replay_query["delay_gate"].unsqueeze(-1)
                )
                short_key_mix = torch.sigmoid(
                    layer.short_key_fusion(torch.cat([abstract_signal, hidden[layer_index]], dim=-1))
                    + 1.2 * replay_query["delay_gate"].unsqueeze(-1)
                )
                summary_mix = torch.sigmoid(
                    layer.summary_fusion(torch.cat([abstract_signal, hidden[layer_index]], dim=-1))
                    - 1.10
                    + 2.0 * (replay_query["delay_gate"].unsqueeze(-1) - 0.5)
                ) * gates.summary_read
                scene_mix = torch.sigmoid(
                    layer.scene_fusion(torch.cat([abstract_signal, hidden[layer_index]], dim=-1))
                    - 1.35
                    + 2.4 * (replay_query["delay_gate"].unsqueeze(-1) - 0.55)
                ) * gates.scene_read
                summary_support = (summary_hits * summary_strength[layer_index]).sum(dim=-1, keepdim=True)
                scene_support = (scene_hits * scene_strength[layer_index]).sum(dim=-1, keepdim=True)
                drill_mix = torch.sigmoid(
                    layer.drill_fusion(torch.cat([abstract_signal, hidden[layer_index]], dim=-1))
                    - 0.70
                    + 2.9 * (replay_query["delay_gate"].unsqueeze(-1) - 0.52)
                    + 1.35 * (summary_support + scene_support - 0.28)
                ) * gates.drill
                short_replay = short_key_mix * short_key_read + (1.0 - short_key_mix) * replay_read
                hierarchical_summary = scene_mix * scene_read + (1.0 - scene_mix) * summary_read
                drill_gate = drill_mix * drill_stats["trigger"].unsqueeze(-1)
                guided_summary = drill_gate * drill_read + (1.0 - drill_gate) * hierarchical_summary
                protected_replay = summary_mix * guided_summary + (1.0 - summary_mix) * short_replay
                episodic_context = replay_mix * episodic_read + (1.0 - replay_mix) * protected_replay
                memory_read = memory_mix * routed_read + (1.0 - memory_mix) * episodic_context
                hidden[layer_index] = layer.cell(cell_input, hidden[layer_index], memory_read)
                (
                    working_keys[layer_index],
                    working_values[layer_index],
                    working_protection[layer_index],
                    writer_stats,
                ) = layer.writer(
                    hidden[layer_index],
                    working_keys[layer_index],
                    working_values[layer_index],
                    working_protection[layer_index],
                    working_usage[layer_index],
                    working_age[layer_index],
                )
                (
                    episodic_keys[layer_index],
                    episodic_values[layer_index],
                    episodic_strength[layer_index],
                    episodic_age[layer_index],
                    episodic_write_stats,
                ) = layer.episodic.write(
                    signal=abstract_signal,
                    hidden=hidden[layer_index],
                    episodic_keys=episodic_keys[layer_index],
                    episodic_values=episodic_values[layer_index],
                    episodic_strength=episodic_strength[layer_index],
                    episodic_age=episodic_age[layer_index],
                )
                (
                    episodic_key_keys[layer_index],
                    episodic_key_values[layer_index],
                    episodic_key_strength[layer_index],
                    episodic_key_age[layer_index],
                    episodic_key_usage[layer_index],
                    short_key_write_stats,
                ) = layer.short_key_memory.write(
                    signal=abstract_signal,
                    hidden=hidden[layer_index],
                    branch_hint=current_branch,
                    abstraction_entropy=abstraction_stats["entropy"],
                    delay_gate=replay_query["delay_gate"],
                    episodic_keys=episodic_keys[layer_index],
                    episodic_values=episodic_values[layer_index],
                    episodic_strength=episodic_strength[layer_index],
                    episodic_replay_hits=episodic_replay_hits[layer_index],
                    episodic_age=episodic_age[layer_index],
                    short_keys=episodic_key_keys[layer_index],
                    short_values=episodic_key_values[layer_index],
                    short_strength=episodic_key_strength[layer_index],
                    short_age=episodic_key_age[layer_index],
                    short_usage=episodic_key_usage[layer_index],
                )
                summary_prev_mass = summary_buffer_mass[layer_index]
                summary_next_mass = summary_prev_mass * self.config.summary_buffer_decay + 1.0
                summary_buffer_state[layer_index] = (
                    summary_buffer_state[layer_index] * summary_prev_mass * self.config.summary_buffer_decay
                    + hidden[layer_index]
                ) / summary_next_mass.clamp_min(1e-6)
                summary_buffer_branch[layer_index] = (
                    summary_buffer_branch[layer_index] * summary_prev_mass * self.config.summary_buffer_decay
                    + current_branch
                ) / summary_next_mass.clamp_min(1e-6)
                summary_buffer_schema[layer_index] = (
                    summary_buffer_schema[layer_index] * summary_prev_mass * self.config.summary_buffer_decay
                    + current_schema
                ) / summary_next_mass.clamp_min(1e-6)
                summary_buffer_mass[layer_index] = summary_next_mass
                summary_branch_hint = summary_buffer_branch[layer_index] / summary_buffer_branch[layer_index].sum(
                    dim=-1, keepdim=True
                ).clamp_min(1e-6)
                summary_schema_hint = summary_buffer_schema[layer_index] / summary_buffer_schema[layer_index].sum(
                    dim=-1, keepdim=True
                ).clamp_min(1e-6)
                summary_boundary = torch.clamp(
                    0.30 * branch_shift
                    + 0.20 * schema_shift
                    + 0.20 * current_branch[:, 2:3]
                    + 0.18 * replay_query["delay_gate"].unsqueeze(-1)
                    + 0.12 * entropy_confidence,
                    0.0,
                    1.0,
                )
                summary_flush_mask = (
                    (summary_boundary > self.config.summary_boundary_threshold)
                    | (step == seq_len - 1)
                ).to(hidden[layer_index].dtype) * gates.summary_write
                flush_summary_state = summary_buffer_state[layer_index]
                flush_summary_branch = summary_branch_hint
                (
                    next_summary_keys,
                    next_summary_values,
                    next_summary_strength,
                    next_summary_age,
                    next_summary_branch_mass,
                    next_summary_schema_mass,
                    summary_write_stats,
                ) = layer.summary_memory.write(
                    signal=flush_summary_state,
                    hidden=hidden[layer_index],
                    branch_hint=flush_summary_branch,
                    schema_hint=summary_schema_hint,
                    episodic_values=episodic_values[layer_index],
                    episodic_strength=episodic_strength[layer_index],
                    short_values=episodic_key_values[layer_index],
                    short_strength=episodic_key_strength[layer_index],
                    summary_keys=summary_keys[layer_index],
                    summary_values=summary_values[layer_index],
                    summary_strength=summary_strength[layer_index],
                    summary_age=summary_age[layer_index],
                    summary_branch_mass=summary_branch_mass[layer_index],
                    summary_schema_mass=summary_schema_mass[layer_index],
                )
                summary_slot_mask = summary_flush_mask.unsqueeze(-1)
                summary_scalar_mask = summary_flush_mask
                summary_keys[layer_index] = (
                    summary_slot_mask * next_summary_keys + (1.0 - summary_slot_mask) * summary_keys[layer_index]
                )
                summary_values[layer_index] = (
                    summary_slot_mask * next_summary_values + (1.0 - summary_slot_mask) * summary_values[layer_index]
                )
                summary_strength[layer_index] = (
                    summary_scalar_mask * next_summary_strength
                    + (1.0 - summary_scalar_mask) * summary_strength[layer_index]
                )
                summary_age[layer_index] = (
                    summary_scalar_mask * next_summary_age + (1.0 - summary_scalar_mask) * summary_age[layer_index]
                )
                summary_branch_mass[layer_index] = (
                    summary_slot_mask * next_summary_branch_mass
                    + (1.0 - summary_slot_mask) * summary_branch_mass[layer_index]
                )
                summary_schema_mass[layer_index] = (
                    summary_slot_mask * next_summary_schema_mass
                    + (1.0 - summary_slot_mask) * summary_schema_mass[layer_index]
                )

                scene_prev_mass = scene_buffer_mass[layer_index]
                scene_buffer_increment = summary_flush_mask
                scene_next_mass = scene_prev_mass * self.config.scene_buffer_decay + scene_buffer_increment
                scene_buffer_state[layer_index] = torch.where(
                    scene_next_mass > 0.0,
                    (
                        scene_buffer_state[layer_index] * scene_prev_mass * self.config.scene_buffer_decay
                        + scene_buffer_increment * flush_summary_state
                    ) / scene_next_mass.clamp_min(1e-6),
                    scene_buffer_state[layer_index],
                )
                scene_buffer_branch[layer_index] = torch.where(
                    scene_next_mass > 0.0,
                    (
                        scene_buffer_branch[layer_index] * scene_prev_mass * self.config.scene_buffer_decay
                        + scene_buffer_increment * flush_summary_branch
                    ) / scene_next_mass.clamp_min(1e-6),
                    scene_buffer_branch[layer_index],
                )
                scene_buffer_schema[layer_index] = torch.where(
                    scene_next_mass > 0.0,
                    (
                        scene_buffer_schema[layer_index] * scene_prev_mass * self.config.scene_buffer_decay
                        + scene_buffer_increment * summary_schema_hint
                    ) / scene_next_mass.clamp_min(1e-6),
                    scene_buffer_schema[layer_index],
                )
                scene_buffer_mass[layer_index] = scene_next_mass
                scene_branch_hint = scene_buffer_branch[layer_index] / scene_buffer_branch[layer_index].sum(
                    dim=-1, keepdim=True
                ).clamp_min(1e-6)
                scene_schema_hint = scene_buffer_schema[layer_index] / scene_buffer_schema[layer_index].sum(
                    dim=-1, keepdim=True
                ).clamp_min(1e-6)
                scene_boundary = torch.clamp(
                    0.34 * summary_boundary
                    + 0.16 * schema_shift
                    + 0.22 * replay_query["delay_gate"].unsqueeze(-1)
                    + 0.18 * torch.clamp(scene_buffer_mass[layer_index] / 2.0, 0.0, 1.0)
                    + 0.10 * current_branch[:, 2:3],
                    0.0,
                    1.0,
                )
                scene_flush_mask = (
                    ((scene_boundary > self.config.scene_boundary_threshold) & (scene_buffer_mass[layer_index] > 0.5))
                    | (step == seq_len - 1)
                ).to(hidden[layer_index].dtype) * gates.scene_write
                (
                    next_scene_keys,
                    next_scene_values,
                    next_scene_strength,
                    next_scene_age,
                    next_scene_branch_mass,
                    next_scene_schema_mass,
                    scene_write_stats,
                ) = layer.scene_memory.write(
                    signal=scene_buffer_state[layer_index],
                    hidden=hidden[layer_index],
                    branch_hint=scene_branch_hint,
                    schema_hint=scene_schema_hint,
                    episodic_values=summary_values[layer_index],
                    episodic_strength=summary_strength[layer_index],
                    short_values=episodic_values[layer_index],
                    short_strength=episodic_strength[layer_index],
                    summary_keys=scene_keys[layer_index],
                    summary_values=scene_values[layer_index],
                    summary_strength=scene_strength[layer_index],
                    summary_age=scene_age[layer_index],
                    summary_branch_mass=scene_branch_mass[layer_index],
                    summary_schema_mass=scene_schema_mass[layer_index],
                )
                scene_slot_mask = scene_flush_mask.unsqueeze(-1)
                scene_scalar_mask = scene_flush_mask
                scene_keys[layer_index] = (
                    scene_slot_mask * next_scene_keys + (1.0 - scene_slot_mask) * scene_keys[layer_index]
                )
                scene_values[layer_index] = (
                    scene_slot_mask * next_scene_values + (1.0 - scene_slot_mask) * scene_values[layer_index]
                )
                scene_strength[layer_index] = (
                    scene_scalar_mask * next_scene_strength + (1.0 - scene_scalar_mask) * scene_strength[layer_index]
                )
                scene_age[layer_index] = (
                    scene_scalar_mask * next_scene_age + (1.0 - scene_scalar_mask) * scene_age[layer_index]
                )
                scene_branch_mass[layer_index] = (
                    scene_slot_mask * next_scene_branch_mass + (1.0 - scene_slot_mask) * scene_branch_mass[layer_index]
                )
                scene_schema_mass[layer_index] = (
                    scene_slot_mask * next_scene_schema_mass + (1.0 - scene_slot_mask) * scene_schema_mass[layer_index]
                )
                branch_write_mass = episodic_write_stats["slot_write_mass"].unsqueeze(-1)
                updated_branch_mass = (
                    episodic_branch_mass[layer_index] * (1.0 - branch_write_mass)
                    + branch_write_mass * current_branch.unsqueeze(1)
                )
                branch_mass_norm = updated_branch_mass.sum(dim=-1, keepdim=True).clamp_min(1e-6)
                episodic_branch_mass[layer_index] = updated_branch_mass / branch_mass_norm
                updated_schema_mass = (
                    episodic_schema_mass[layer_index] * (1.0 - branch_write_mass)
                    + branch_write_mass * current_schema.unsqueeze(1)
                )
                schema_mass_norm = updated_schema_mass.sum(dim=-1, keepdim=True).clamp_min(1e-6)
                episodic_schema_mass[layer_index] = updated_schema_mass / schema_mass_norm
                episodic_replay_hits[layer_index] = torch.clamp(
                    episodic_replay_hits[layer_index] * self.context_spec.config.replay_decay
                    + slot_replay_mass
                    + 0.24 * gates.drill * drill_hits
                    + 0.18
                    * gates.summary_read
                    * summary_hits.mean(dim=-1, keepdim=True).expand_as(episodic_replay_hits[layer_index])
                    + 0.12
                    * gates.scene_read
                    * scene_hits.mean(dim=-1, keepdim=True).expand_as(episodic_replay_hits[layer_index]),
                    0.0,
                    6.0,
                )
                previous_values = episodic_values[layer_index]
                previous_strength = episodic_strength[layer_index]
                previous_age = episodic_age[layer_index]
                previous_cold_steps = episodic_cold_steps[layer_index]
                (
                    episodic_values[layer_index],
                    episodic_strength[layer_index],
                    episodic_age[layer_index],
                    episodic_cold_steps[layer_index],
                    forgetting_stats,
                ) = self._apply_episodic_forgetting(
                    episodic_values=episodic_values[layer_index],
                    episodic_strength=episodic_strength[layer_index],
                    episodic_age=episodic_age[layer_index],
                    episodic_replay_hits=episodic_replay_hits[layer_index],
                    episodic_cold_steps=episodic_cold_steps[layer_index],
                )
                episodic_values[layer_index] = (
                    gates.forgetting * episodic_values[layer_index]
                    + (1.0 - gates.forgetting) * previous_values
                )
                episodic_strength[layer_index] = (
                    gates.forgetting * episodic_strength[layer_index]
                    + (1.0 - gates.forgetting) * previous_strength
                )
                episodic_age[layer_index] = (
                    gates.forgetting * episodic_age[layer_index] + (1.0 - gates.forgetting) * previous_age
                )
                episodic_cold_steps[layer_index] = (
                    gates.forgetting * episodic_cold_steps[layer_index]
                    + (1.0 - gates.forgetting) * previous_cold_steps
                )
                episodic_keys[layer_index] = episodic_keys[layer_index].detach()
                episodic_values[layer_index] = episodic_values[layer_index].detach()
                episodic_strength[layer_index] = episodic_strength[layer_index].detach()
                episodic_age[layer_index] = episodic_age[layer_index].detach()
                episodic_branch_mass[layer_index] = episodic_branch_mass[layer_index].detach()
                episodic_schema_mass[layer_index] = episodic_schema_mass[layer_index].detach()
                episodic_replay_hits[layer_index] = episodic_replay_hits[layer_index].detach()
                episodic_cold_steps[layer_index] = episodic_cold_steps[layer_index].detach()
                episodic_key_keys[layer_index] = episodic_key_keys[layer_index].detach()
                episodic_key_values[layer_index] = episodic_key_values[layer_index].detach()
                episodic_key_strength[layer_index] = episodic_key_strength[layer_index].detach()
                episodic_key_age[layer_index] = episodic_key_age[layer_index].detach()
                episodic_key_usage[layer_index] = episodic_key_usage[layer_index].detach()
                summary_keys[layer_index] = summary_keys[layer_index].detach()
                summary_values[layer_index] = summary_values[layer_index].detach()
                summary_strength[layer_index] = summary_strength[layer_index].detach()
                summary_age[layer_index] = summary_age[layer_index].detach()
                summary_branch_mass[layer_index] = summary_branch_mass[layer_index].detach()
                summary_schema_mass[layer_index] = summary_schema_mass[layer_index].detach()
                scene_keys[layer_index] = scene_keys[layer_index].detach()
                scene_values[layer_index] = scene_values[layer_index].detach()
                scene_strength[layer_index] = scene_strength[layer_index].detach()
                scene_age[layer_index] = scene_age[layer_index].detach()
                scene_branch_mass[layer_index] = scene_branch_mass[layer_index].detach()
                scene_schema_mass[layer_index] = scene_schema_mass[layer_index].detach()
                summary_keep = 1.0 - summary_flush_mask
                summary_buffer_state[layer_index] = (summary_buffer_state[layer_index] * summary_keep).detach()
                summary_buffer_branch[layer_index] = (summary_buffer_branch[layer_index] * summary_keep).detach()
                summary_buffer_schema[layer_index] = (summary_buffer_schema[layer_index] * summary_keep).detach()
                summary_buffer_mass[layer_index] = (summary_buffer_mass[layer_index] * summary_keep).detach()
                scene_keep = 1.0 - scene_flush_mask
                scene_buffer_state[layer_index] = (scene_buffer_state[layer_index] * scene_keep).detach()
                scene_buffer_branch[layer_index] = (scene_buffer_branch[layer_index] * scene_keep).detach()
                scene_buffer_schema[layer_index] = (scene_buffer_schema[layer_index] * scene_keep).detach()
                scene_buffer_mass[layer_index] = (scene_buffer_mass[layer_index] * scene_keep).detach()
                previous_branch[layer_index] = current_branch.detach()
                previous_schema[layer_index] = current_schema.detach()
                read_weights = torch.zeros_like(working_usage[layer_index])
                working_mask = route["top_indices"] < self.config.working_memory_slots
                if working_mask.any():
                    safe_indices = route["top_indices"].masked_fill(~working_mask, 0)
                    read_weights.scatter_add_(
                        1,
                        safe_indices,
                        route["weights"] * working_mask.to(route["weights"].dtype),
                    )
                slot_write_mass = writer_stats["slot_write_mass"].detach()
                working_usage[layer_index] = torch.clamp(
                    working_usage[layer_index] * self.config.working_usage_decay
                    + 0.7 * read_weights.detach()
                    + 0.9 * slot_write_mass,
                    0.0,
                    1.0,
                ).detach()
                working_age[layer_index] = torch.clamp(
                    (working_age[layer_index] + self.config.working_age_increment) * (1.0 - slot_write_mass),
                    0.0,
                    1.0,
                ).detach()
                working_protection[layer_index] = working_protection[layer_index].detach()
                current = hidden[layer_index]
                weights = route["weights"]
                route_entropy = route_entropy + (-(weights * torch.log(weights + 1e-8)).sum(dim=-1).mean())
                working_ratio = working_ratio + route["working_ratio"]
                write_gate_mean = write_gate_mean + writer_stats["write_strength"].mean()
                merge_preference_mean = merge_preference_mean + writer_stats["merge_preference"].mean()
                binding_strength_mean = binding_strength_mean + writer_stats["binding_strength"].mean()
                episodic_read_mix_mean = episodic_read_mix_mean + episodic_read_stats["read_mix"].mean()
                episodic_write_gate_mean = episodic_write_gate_mean + episodic_write_stats["write_strength"].mean()
                episodic_persistence_mean = episodic_persistence_mean + episodic_write_stats["persistence"].mean()
                episodic_overwrite_ratio_mean = (
                    episodic_overwrite_ratio_mean + episodic_write_stats["overwrite_ratio"].mean()
                )
                episodic_strength_mean = episodic_strength_mean + episodic_write_stats["strength_mean"].mean()
                episodic_similarity_mean = episodic_similarity_mean + episodic_read_stats["max_similarity"].mean()
                episodic_replay_score_mean = episodic_replay_score_mean + replay_stats["score_mean"].mean()
                episodic_replay_active_ratio_mean = (
                    episodic_replay_active_ratio_mean + replay_stats["active_ratio"].mean()
                )
                episodic_replay_delay_mean = episodic_replay_delay_mean + replay_stats["delay_gate"].mean()
                episodic_replay_branch_alignment_mean = (
                    episodic_replay_branch_alignment_mean + replay_stats["branch_alignment"].mean()
                )
                episodic_replay_schema_alignment_mean = (
                    episodic_replay_schema_alignment_mean + replay_stats["schema_alignment"].mean()
                )
                episodic_retention_mean = episodic_retention_mean + forgetting_stats["retention_mean"].mean()
                episodic_cool_ratio_mean = episodic_cool_ratio_mean + forgetting_stats["cool_ratio"].mean()
                episodic_fade_ratio_mean = episodic_fade_ratio_mean + forgetting_stats["fade_ratio"].mean()
                episodic_archive_ratio_mean = (
                    episodic_archive_ratio_mean + forgetting_stats["archive_ratio"].mean()
                )
                episodic_prune_ratio_mean = episodic_prune_ratio_mean + forgetting_stats["prune_ratio"].mean()
                episodic_multi_merge_ratio_mean = (
                    episodic_multi_merge_ratio_mean + episodic_write_stats["multi_merge_ratio"].mean()
                )
                episodic_partial_merge_ratio_mean = (
                    episodic_partial_merge_ratio_mean + episodic_write_stats["partial_merge_ratio"].mean()
                )
                episodic_key_focus_mean = episodic_key_focus_mean + short_key_write_stats["key_focus"].mean()
                episodic_key_consolidation_mean = (
                    episodic_key_consolidation_mean + short_key_write_stats["consolidation"].mean()
                )
                episodic_key_overwrite_ratio_mean = (
                    episodic_key_overwrite_ratio_mean + short_key_write_stats["overwrite_ratio"].mean()
                )
                episodic_key_strength_mean = (
                    episodic_key_strength_mean + short_key_write_stats["strength_mean"].mean()
                )
                episodic_key_usage_mean = episodic_key_usage_mean + short_key_write_stats["usage_mean"].mean()
                episodic_key_read_score_mean = (
                    episodic_key_read_score_mean + short_key_read_stats["score_mean"].mean()
                )
                episodic_key_read_active_ratio_mean = (
                    episodic_key_read_active_ratio_mean + short_key_read_stats["active_ratio"].mean()
                )
                episodic_key_persistent_preference_mean = (
                    episodic_key_persistent_preference_mean
                    + short_key_read_stats["persistent_preference"].mean()
                )
                summary_read_score_mean = summary_read_score_mean + summary_read_stats["score_mean"].mean()
                summary_read_active_ratio_mean = (
                    summary_read_active_ratio_mean + summary_read_stats["active_ratio"].mean()
                )
                summary_strength_mean = summary_strength_mean + summary_read_stats["strength_mean"].mean()
                summary_branch_alignment_mean = (
                    summary_branch_alignment_mean + summary_read_stats["branch_alignment"].mean()
                )
                summary_schema_alignment_mean = (
                    summary_schema_alignment_mean + summary_read_stats["schema_alignment"].mean()
                )
                summary_overwrite_ratio_mean = (
                    summary_overwrite_ratio_mean
                    + (summary_write_stats["overwrite_ratio"] * summary_flush_mask.squeeze(-1)).mean()
                )
                summary_merge_preference_mean = (
                    summary_merge_preference_mean
                    + (summary_write_stats["merge_preference"] * summary_flush_mask.squeeze(-1)).mean()
                )
                scene_read_score_mean = scene_read_score_mean + scene_read_stats["score_mean"].mean()
                scene_read_active_ratio_mean = (
                    scene_read_active_ratio_mean + scene_read_stats["active_ratio"].mean()
                )
                scene_strength_mean = scene_strength_mean + scene_read_stats["strength_mean"].mean()
                scene_branch_alignment_mean = (
                    scene_branch_alignment_mean + scene_read_stats["branch_alignment"].mean()
                )
                scene_schema_alignment_mean = (
                    scene_schema_alignment_mean + scene_read_stats["schema_alignment"].mean()
                )
                scene_overwrite_ratio_mean = (
                    scene_overwrite_ratio_mean
                    + (scene_write_stats["overwrite_ratio"] * scene_flush_mask.squeeze(-1)).mean()
                )
                scene_merge_preference_mean = (
                    scene_merge_preference_mean
                    + (scene_write_stats["merge_preference"] * scene_flush_mask.squeeze(-1)).mean()
                )
                drill_score_mean = drill_score_mean + drill_stats["score_mean"].mean()
                drill_active_ratio_mean = drill_active_ratio_mean + drill_stats["active_ratio"].mean()
                drill_trigger_mean = drill_trigger_mean + drill_stats["trigger"].mean()
                drill_alignment_mean = drill_alignment_mean + drill_stats["alignment_mean"].mean()
                summary_boundary_mean = summary_boundary_mean + summary_boundary.mean()
                summary_flush_ratio_mean = summary_flush_ratio_mean + summary_flush_mask.mean()
                scene_boundary_mean = scene_boundary_mean + scene_boundary.mean()
                scene_flush_ratio_mean = scene_flush_ratio_mean + scene_flush_mask.mean()
                abstraction_gate_mean = abstraction_gate_mean + abstraction_stats["gate_mean"].mean()
                abstraction_delta_mean = abstraction_delta_mean + abstraction_stats["delta_mean"].mean()
                abstraction_entropy_mean = abstraction_entropy_mean + abstraction_stats["entropy"].mean()
                abstraction_anchor_entropy_mean = (
                    abstraction_anchor_entropy_mean + abstraction_stats["anchor_entropy"].mean()
                )
                abstraction_stop_depth_mean = abstraction_stop_depth_mean + abstraction_stats["stop_depth"].mean()
                abstraction_stopped_ratio_mean = (
                    abstraction_stopped_ratio_mean + abstraction_stats["stopped_ratio"].mean()
                )
                abstraction_adequacy_mean = abstraction_adequacy_mean + abstraction_stats["adequacy_mean"].mean()
                abstraction_entity_weight_mean = (
                    abstraction_entity_weight_mean + abstraction_stats["entity_weight"].mean()
                )
                abstraction_relation_weight_mean = (
                    abstraction_relation_weight_mean + abstraction_stats["relation_weight"].mean()
                )
                abstraction_event_weight_mean = (
                    abstraction_event_weight_mean + abstraction_stats["event_weight"].mean()
                )
                abstraction_schema_active_ratio_mean = (
                    abstraction_schema_active_ratio_mean + abstraction_stats["schema_active_ratio"].mean()
                )
                abstraction_schema_peak_mean = (
                    abstraction_schema_peak_mean + abstraction_stats["schema_peak"].mean()
                )
                abstraction_schema_widen_mean = (
                    abstraction_schema_widen_mean + abstraction_stats["schema_widen_mean"].mean()
                )
                abstraction_schema_narrow_mean = (
                    abstraction_schema_narrow_mean + abstraction_stats["schema_narrow_mean"].mean()
                )
                abstraction_schema_split_mean = (
                    abstraction_schema_split_mean + abstraction_stats["schema_split_mean"].mean()
                )
                abstraction_schema_merge_mean = (
                    abstraction_schema_merge_mean + abstraction_stats["schema_merge_mean"].mean()
                )
                abstraction_schema_suspend_mean = (
                    abstraction_schema_suspend_mean + abstraction_stats["schema_suspend_mean"].mean()
                )
                abstraction_schema_suspend_mass_mean = (
                    abstraction_schema_suspend_mass_mean + abstraction_stats["schema_suspend_mass_mean"].mean()
                )
                abstraction_schema_temperature_mean = (
                    abstraction_schema_temperature_mean + abstraction_stats["schema_temperature_mean"].mean()
                )
                abstraction_residual_scale_mean = (
                    abstraction_residual_scale_mean + abstraction_stats["residual_scale"].mean()
                )
                overwrite_ratio_mean = overwrite_ratio_mean + writer_stats["overwrite_ratio"].mean()
                protection_mean = protection_mean + writer_stats["protection_mean"].mean()
                usage_mean = usage_mean + working_usage[layer_index].mean()
                age_mean = age_mean + working_age[layer_index].mean()
                total_steps += 1
            outputs.append(self.final_norm(self.dropout(current)))

        stacked = torch.stack(outputs, dim=1)
        logits = self.lm_head(stacked)

        aux = {
            "route_entropy": float((route_entropy / max(total_steps, 1)).detach().cpu()),
            "working_read_ratio": float((working_ratio / max(total_steps, 1)).detach().cpu()),
            "write_gate_mean": float((write_gate_mean / max(total_steps, 1)).detach().cpu()),
            "merge_preference_mean": float((merge_preference_mean / max(total_steps, 1)).detach().cpu()),
            "binding_strength_mean": float((binding_strength_mean / max(total_steps, 1)).detach().cpu()),
            "episodic_read_mix_mean": float((episodic_read_mix_mean / max(total_steps, 1)).detach().cpu()),
            "episodic_write_gate_mean": float((episodic_write_gate_mean / max(total_steps, 1)).detach().cpu()),
            "episodic_persistence_mean": float((episodic_persistence_mean / max(total_steps, 1)).detach().cpu()),
            "episodic_overwrite_ratio_mean": float(
                (episodic_overwrite_ratio_mean / max(total_steps, 1)).detach().cpu()
            ),
            "episodic_strength_mean": float((episodic_strength_mean / max(total_steps, 1)).detach().cpu()),
            "episodic_similarity_mean": float((episodic_similarity_mean / max(total_steps, 1)).detach().cpu()),
            "episodic_replay_score_mean": float((episodic_replay_score_mean / max(total_steps, 1)).detach().cpu()),
            "episodic_replay_active_ratio_mean": float(
                (episodic_replay_active_ratio_mean / max(total_steps, 1)).detach().cpu()
            ),
            "episodic_replay_delay_mean": float((episodic_replay_delay_mean / max(total_steps, 1)).detach().cpu()),
            "episodic_replay_branch_alignment_mean": float(
                (episodic_replay_branch_alignment_mean / max(total_steps, 1)).detach().cpu()
            ),
            "episodic_replay_schema_alignment_mean": float(
                (episodic_replay_schema_alignment_mean / max(total_steps, 1)).detach().cpu()
            ),
            "episodic_retention_mean": float((episodic_retention_mean / max(total_steps, 1)).detach().cpu()),
            "episodic_cool_ratio_mean": float((episodic_cool_ratio_mean / max(total_steps, 1)).detach().cpu()),
            "episodic_fade_ratio_mean": float((episodic_fade_ratio_mean / max(total_steps, 1)).detach().cpu()),
            "episodic_archive_ratio_mean": float(
                (episodic_archive_ratio_mean / max(total_steps, 1)).detach().cpu()
            ),
            "episodic_prune_ratio_mean": float((episodic_prune_ratio_mean / max(total_steps, 1)).detach().cpu()),
            "episodic_multi_merge_ratio_mean": float(
                (episodic_multi_merge_ratio_mean / max(total_steps, 1)).detach().cpu()
            ),
            "episodic_partial_merge_ratio_mean": float(
                (episodic_partial_merge_ratio_mean / max(total_steps, 1)).detach().cpu()
            ),
            "episodic_key_focus_mean": float((episodic_key_focus_mean / max(total_steps, 1)).detach().cpu()),
            "episodic_key_consolidation_mean": float(
                (episodic_key_consolidation_mean / max(total_steps, 1)).detach().cpu()
            ),
            "episodic_key_overwrite_ratio_mean": float(
                (episodic_key_overwrite_ratio_mean / max(total_steps, 1)).detach().cpu()
            ),
            "episodic_key_strength_mean": float(
                (episodic_key_strength_mean / max(total_steps, 1)).detach().cpu()
            ),
            "episodic_key_usage_mean": float((episodic_key_usage_mean / max(total_steps, 1)).detach().cpu()),
            "episodic_key_read_score_mean": float(
                (episodic_key_read_score_mean / max(total_steps, 1)).detach().cpu()
            ),
            "episodic_key_read_active_ratio_mean": float(
                (episodic_key_read_active_ratio_mean / max(total_steps, 1)).detach().cpu()
            ),
            "episodic_key_persistent_preference_mean": float(
                (episodic_key_persistent_preference_mean / max(total_steps, 1)).detach().cpu()
            ),
            "summary_read_score_mean": float((summary_read_score_mean / max(total_steps, 1)).detach().cpu()),
            "summary_read_active_ratio_mean": float(
                (summary_read_active_ratio_mean / max(total_steps, 1)).detach().cpu()
            ),
            "summary_strength_mean": float((summary_strength_mean / max(total_steps, 1)).detach().cpu()),
            "summary_branch_alignment_mean": float(
                (summary_branch_alignment_mean / max(total_steps, 1)).detach().cpu()
            ),
            "summary_schema_alignment_mean": float(
                (summary_schema_alignment_mean / max(total_steps, 1)).detach().cpu()
            ),
            "summary_overwrite_ratio_mean": float(
                (summary_overwrite_ratio_mean / max(total_steps, 1)).detach().cpu()
            ),
            "summary_merge_preference_mean": float(
                (summary_merge_preference_mean / max(total_steps, 1)).detach().cpu()
            ),
            "scene_read_score_mean": float((scene_read_score_mean / max(total_steps, 1)).detach().cpu()),
            "scene_read_active_ratio_mean": float(
                (scene_read_active_ratio_mean / max(total_steps, 1)).detach().cpu()
            ),
            "scene_strength_mean": float((scene_strength_mean / max(total_steps, 1)).detach().cpu()),
            "scene_branch_alignment_mean": float(
                (scene_branch_alignment_mean / max(total_steps, 1)).detach().cpu()
            ),
            "scene_schema_alignment_mean": float(
                (scene_schema_alignment_mean / max(total_steps, 1)).detach().cpu()
            ),
            "scene_overwrite_ratio_mean": float(
                (scene_overwrite_ratio_mean / max(total_steps, 1)).detach().cpu()
            ),
            "scene_merge_preference_mean": float(
                (scene_merge_preference_mean / max(total_steps, 1)).detach().cpu()
            ),
            "drill_score_mean": float((drill_score_mean / max(total_steps, 1)).detach().cpu()),
            "drill_active_ratio_mean": float((drill_active_ratio_mean / max(total_steps, 1)).detach().cpu()),
            "drill_trigger_mean": float((drill_trigger_mean / max(total_steps, 1)).detach().cpu()),
            "drill_alignment_mean": float((drill_alignment_mean / max(total_steps, 1)).detach().cpu()),
            "summary_boundary_mean": float((summary_boundary_mean / max(total_steps, 1)).detach().cpu()),
            "summary_flush_ratio_mean": float((summary_flush_ratio_mean / max(total_steps, 1)).detach().cpu()),
            "scene_boundary_mean": float((scene_boundary_mean / max(total_steps, 1)).detach().cpu()),
            "scene_flush_ratio_mean": float((scene_flush_ratio_mean / max(total_steps, 1)).detach().cpu()),
            "abstraction_gate_mean": float((abstraction_gate_mean / max(total_steps, 1)).detach().cpu()),
            "abstraction_delta_mean": float((abstraction_delta_mean / max(total_steps, 1)).detach().cpu()),
            "abstraction_entropy_mean": float((abstraction_entropy_mean / max(total_steps, 1)).detach().cpu()),
            "abstraction_anchor_entropy_mean": float(
                (abstraction_anchor_entropy_mean / max(total_steps, 1)).detach().cpu()
            ),
            "abstraction_stop_depth_mean": float(
                (abstraction_stop_depth_mean / max(total_steps, 1)).detach().cpu()
            ),
            "abstraction_stopped_ratio_mean": float(
                (abstraction_stopped_ratio_mean / max(total_steps, 1)).detach().cpu()
            ),
            "abstraction_adequacy_mean": float(
                (abstraction_adequacy_mean / max(total_steps, 1)).detach().cpu()
            ),
            "abstraction_entity_weight_mean": float(
                (abstraction_entity_weight_mean / max(total_steps, 1)).detach().cpu()
            ),
            "abstraction_relation_weight_mean": float(
                (abstraction_relation_weight_mean / max(total_steps, 1)).detach().cpu()
            ),
            "abstraction_event_weight_mean": float(
                (abstraction_event_weight_mean / max(total_steps, 1)).detach().cpu()
            ),
            "abstraction_schema_active_ratio_mean": float(
                (abstraction_schema_active_ratio_mean / max(total_steps, 1)).detach().cpu()
            ),
            "abstraction_schema_peak_mean": float(
                (abstraction_schema_peak_mean / max(total_steps, 1)).detach().cpu()
            ),
            "abstraction_schema_widen_mean": float(
                (abstraction_schema_widen_mean / max(total_steps, 1)).detach().cpu()
            ),
            "abstraction_schema_narrow_mean": float(
                (abstraction_schema_narrow_mean / max(total_steps, 1)).detach().cpu()
            ),
            "abstraction_schema_split_mean": float(
                (abstraction_schema_split_mean / max(total_steps, 1)).detach().cpu()
            ),
            "abstraction_schema_merge_mean": float(
                (abstraction_schema_merge_mean / max(total_steps, 1)).detach().cpu()
            ),
            "abstraction_schema_suspend_mean": float(
                (abstraction_schema_suspend_mean / max(total_steps, 1)).detach().cpu()
            ),
            "abstraction_schema_suspend_mass_mean": float(
                (abstraction_schema_suspend_mass_mean / max(total_steps, 1)).detach().cpu()
            ),
            "abstraction_schema_temperature_mean": float(
                (abstraction_schema_temperature_mean / max(total_steps, 1)).detach().cpu()
            ),
            "abstraction_residual_scale_mean": float(
                (abstraction_residual_scale_mean / max(total_steps, 1)).detach().cpu()
            ),
            "runtime_summary_read_gate": float(gates.summary_read),
            "runtime_summary_write_gate": float(gates.summary_write),
            "runtime_scene_read_gate": float(gates.scene_read),
            "runtime_scene_write_gate": float(gates.scene_write),
            "runtime_drill_gate": float(gates.drill),
            "runtime_forgetting_gate": float(gates.forgetting),
            "overwrite_ratio_mean": float((overwrite_ratio_mean / max(total_steps, 1)).detach().cpu()),
            "protection_mean": float((protection_mean / max(total_steps, 1)).detach().cpu()),
            "usage_mean": float((usage_mean / max(total_steps, 1)).detach().cpu()),
            "age_mean": float((age_mean / max(total_steps, 1)).detach().cpu()),
            "avg_active_slots": float(self.config.router_top_k),
        }
        next_state = state.detached() if return_state else None
        if not return_aux:
            payload: Dict[str, Tensor | Dict[str, float] | SBCoreMemoryState] = {"logits": logits}
            if next_state is not None:
                payload["state"] = next_state
            return payload
        payload = {"logits": logits, "aux": aux}
        if next_state is not None:
            payload["state"] = next_state
        return payload


def next_token_loss(logits: Tensor, target_ids: Tensor, focus_mask: Tensor | None = None) -> Tensor:
    vocab_size = logits.shape[-1]
    losses = F.cross_entropy(logits.reshape(-1, vocab_size), target_ids.reshape(-1), reduction="none")
    if focus_mask is None:
        return losses.mean()

    mask = focus_mask.reshape(-1).to(dtype=losses.dtype)
    masked_count = mask.sum()
    if float(masked_count.detach().cpu()) <= 0.0:
        return losses.mean()
    return (losses * mask).sum() / masked_count
