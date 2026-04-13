from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .hierarchical_context import HierarchicalContextSpec, MemoryLevel


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
    signal_abstraction_levels: int = 3
    signal_stop_threshold: float = 0.62
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
        if self.signal_abstraction_levels <= 0:
            raise ValueError("signal_abstraction_levels 必须大于 0。")
        if not 0.0 < self.signal_stop_threshold < 1.0:
            raise ValueError("signal_stop_threshold 必须在 (0, 1) 之间。")


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
    BRANCH_NAMES = ("entity", "relation", "event")

    def __init__(self, state_dim: int, levels: int, stop_threshold: float) -> None:
        super().__init__()
        self.branches = nn.ModuleDict(
            {
                name: SBSignalBranch(
                    state_dim=state_dim,
                    levels=levels,
                    stop_threshold=stop_threshold,
                )
                for name in self.BRANCH_NAMES
            }
        )
        self.branch_router = nn.Linear(state_dim * 2, len(self.BRANCH_NAMES))
        self.residual_scale = nn.Linear(state_dim * 2, 1)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.zeros_(self.branch_router.weight)
        nn.init.zeros_(self.branch_router.bias)
        nn.init.zeros_(self.residual_scale.weight)
        nn.init.constant_(self.residual_scale.bias, -1.5)

    def forward(self, signal: Tensor, previous: Tensor) -> Tuple[Tensor, Dict[str, Tensor]]:
        joined = torch.cat([signal, previous], dim=-1)
        branch_weights = F.softmax(self.branch_router(joined), dim=-1)

        branch_outputs: List[Tensor] = []
        branch_gate_stats: List[Tensor] = []
        branch_delta_stats: List[Tensor] = []
        branch_depth_stats: List[Tensor] = []
        branch_stop_stats: List[Tensor] = []
        branch_adequacy_stats: List[Tensor] = []

        for name in self.BRANCH_NAMES:
            branch_output, branch_stats = self.branches[name](signal, previous)
            branch_outputs.append(branch_output)
            branch_gate_stats.append(branch_stats["gate_mean"])
            branch_delta_stats.append(branch_stats["delta_mean"])
            branch_depth_stats.append(branch_stats["stop_depth"])
            branch_stop_stats.append(branch_stats["stopped_ratio"])
            branch_adequacy_stats.append(branch_stats["adequacy_mean"])

        stacked_outputs = torch.stack(branch_outputs, dim=1)
        residuals = stacked_outputs - signal.unsqueeze(1)
        residual_scale = torch.sigmoid(self.residual_scale(joined))
        fused = signal + residual_scale * torch.sum(residuals * branch_weights.unsqueeze(-1), dim=1)

        branch_gate_tensor = torch.stack(branch_gate_stats, dim=1)
        branch_delta_tensor = torch.stack(branch_delta_stats, dim=1)
        branch_depth_tensor = torch.stack(branch_depth_stats, dim=1)
        branch_stop_tensor = torch.stack(branch_stop_stats, dim=1)
        branch_adequacy_tensor = torch.stack(branch_adequacy_stats, dim=1)

        stats = {
            "gate_mean": torch.sum(branch_gate_tensor * branch_weights, dim=1),
            "delta_mean": torch.sum(branch_delta_tensor * branch_weights, dim=1),
            "entropy": (-(branch_weights * torch.log(branch_weights + 1e-8)).sum(dim=-1)),
            "stop_depth": torch.sum(branch_depth_tensor * branch_weights, dim=1),
            "stopped_ratio": torch.sum(branch_stop_tensor * branch_weights, dim=1),
            "adequacy_mean": torch.sum(branch_adequacy_tensor * branch_weights, dim=1),
            "entity_weight": branch_weights[:, 0],
            "relation_weight": branch_weights[:, 1],
            "event_weight": branch_weights[:, 2],
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
    def __init__(self, state_dim: int) -> None:
        super().__init__()
        joined_dim = state_dim * 2 + 3
        self.key_query = nn.Linear(joined_dim, state_dim, bias=False)
        self.branch_query = nn.Linear(joined_dim, 3)
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

    def forward(self, signal: Tensor, previous: Tensor, branch_hint: Tensor) -> Dict[str, Tensor]:
        joined = torch.cat([signal, previous, branch_hint], dim=-1)
        return {
            "key": F.normalize(self.key_query(joined), dim=-1, eps=1e-6),
            "branch": F.softmax(self.branch_query(joined), dim=-1),
            "delay_gate": torch.sigmoid(self.delay_gate(joined)).squeeze(-1),
            "salience_gate": torch.sigmoid(self.salience_gate(joined)).squeeze(-1),
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
        read = torch.sum(selected_values * top_weights.unsqueeze(-1), dim=1)

        slot_hits = torch.zeros_like(short_strength)
        slot_hits.scatter_add_(1, top_indices, top_weights)
        stats = {
            "score_mean": scores.mean(dim=-1),
            "active_ratio": (slot_hits > 0.0).float().mean(dim=-1),
            "strength_mean": short_strength.mean(dim=-1),
            "usage_mean": short_usage.mean(dim=-1),
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
            self.consolidation_gate(routed).squeeze(-1) + 2.2 * (source_confidence - 0.5)
        )
        persistence = torch.sigmoid(self.persistence_gate(routed)).squeeze(-1)
        key_focus = torch.clamp(0.45 * focus_base + 0.30 * compactness + 0.25 * delay_gate, 0.0, 1.0)

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

        overwrite = target_weights * (0.10 + 0.80 * key_focus.unsqueeze(-1)) * (0.55 + 0.45 * compactness.unsqueeze(-1))
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


class SBMiniLayer(nn.Module):
    def __init__(
        self,
        state_dim: int,
        working_slots: int,
        top_k: int,
        episodic_slots: int,
        episodic_key_slots: int,
        *,
        protection_decay: float,
        temperature: float,
        episodic_strength_decay: float,
        episodic_age_increment: float,
        episodic_temperature: float,
        episodic_key_decay: float,
        episodic_key_age_increment: float,
        episodic_key_temperature: float,
        abstraction_levels: int,
        stop_threshold: float,
    ) -> None:
        super().__init__()
        self.abstraction = SBSignalAbstraction(
            state_dim=state_dim,
            levels=abstraction_levels,
            stop_threshold=stop_threshold,
        )
        self.router = SBMemoryRouter(state_dim=state_dim, top_k=top_k)
        self.replay_query_builder = SBKeyCentricReplayQueryBuilder(state_dim=state_dim)
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
        self.memory_fusion = nn.Linear(state_dim * 2, 1)
        self.replay_fusion = nn.Linear(state_dim * 2, 1)
        self.short_key_fusion = nn.Linear(state_dim * 2, 1)
        nn.init.zeros_(self.replay_fusion.weight)
        nn.init.constant_(self.replay_fusion.bias, 1.5)
        nn.init.zeros_(self.short_key_fusion.weight)
        nn.init.constant_(self.short_key_fusion.bias, 0.4)
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
                    protection_decay=config.working_protection_decay,
                    temperature=config.working_memory_temperature,
                    episodic_strength_decay=config.episodic_strength_decay,
                    episodic_age_increment=config.episodic_age_increment,
                    episodic_temperature=config.episodic_memory_temperature,
                    episodic_key_decay=config.episodic_key_decay,
                    episodic_key_age_increment=config.episodic_key_age_increment,
                    episodic_key_temperature=config.episodic_key_temperature,
                    abstraction_levels=config.signal_abstraction_levels,
                    stop_threshold=config.signal_stop_threshold,
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
        if config.tie_weights and config.d_model == config.state_dim:
            self.lm_head.weight = self.embedding.weight

    def _episodic_replay_read(
        self,
        *,
        query_key: Tensor,
        query_branch: Tensor,
        delay_gate: Tensor,
        salience_gate: Tensor,
        episodic_branch_mass: Tensor,
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
        task_match = torch.sigmoid(key_scores)
        salience = salience_gate.unsqueeze(-1) * episodic_strength
        delay_match = delay_gate.unsqueeze(-1) * episodic_age
        replay_scores = (
            weights["task"] * task_match
            + weights["entity"] * entity_match
            + weights["relation_event"] * relation_event_match
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
        }
        return replay_read, slot_replay_mass, stats

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

    def forward(self, input_ids: Tensor, return_aux: bool = True) -> Dict[str, Tensor | Dict[str, float]]:
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        if seq_len > self.config.max_seq_len:
            raise ValueError(f"seq_len={seq_len} 超过 max_seq_len={self.config.max_seq_len}")
        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        x = self.embedding(input_ids) + self.position_embedding(positions)
        x = self.input_proj(x)

        hidden = [torch.zeros(batch_size, self.config.state_dim, device=device) for _ in range(self.config.num_layers)]
        working_keys = [
            torch.zeros(batch_size, self.config.working_memory_slots, self.config.state_dim, device=device)
            for _ in range(self.config.num_layers)
        ]
        working_values = [
            torch.zeros(batch_size, self.config.working_memory_slots, self.config.state_dim, device=device)
            for _ in range(self.config.num_layers)
        ]
        working_protection = [
            torch.zeros(batch_size, self.config.working_memory_slots, device=device)
            for _ in range(self.config.num_layers)
        ]
        working_usage = [
            torch.zeros(batch_size, self.config.working_memory_slots, device=device)
            for _ in range(self.config.num_layers)
        ]
        working_age = [
            torch.zeros(batch_size, self.config.working_memory_slots, device=device)
            for _ in range(self.config.num_layers)
        ]
        episodic_keys = [
            torch.zeros(batch_size, self.config.episodic_memory_slots, self.config.state_dim, device=device)
            for _ in range(self.config.num_layers)
        ]
        episodic_values = [
            torch.zeros(batch_size, self.config.episodic_memory_slots, self.config.state_dim, device=device)
            for _ in range(self.config.num_layers)
        ]
        episodic_strength = [
            torch.zeros(batch_size, self.config.episodic_memory_slots, device=device)
            for _ in range(self.config.num_layers)
        ]
        episodic_age = [
            torch.zeros(batch_size, self.config.episodic_memory_slots, device=device)
            for _ in range(self.config.num_layers)
        ]
        episodic_branch_mass = [
            torch.zeros(batch_size, self.config.episodic_memory_slots, 3, device=device)
            for _ in range(self.config.num_layers)
        ]
        episodic_replay_hits = [
            torch.zeros(batch_size, self.config.episodic_memory_slots, device=device)
            for _ in range(self.config.num_layers)
        ]
        episodic_cold_steps = [
            torch.zeros(batch_size, self.config.episodic_memory_slots, device=device)
            for _ in range(self.config.num_layers)
        ]
        episodic_key_keys = [
            torch.zeros(batch_size, self.config.episodic_key_slots, self.config.state_dim, device=device)
            for _ in range(self.config.num_layers)
        ]
        episodic_key_values = [
            torch.zeros(batch_size, self.config.episodic_key_slots, self.config.state_dim, device=device)
            for _ in range(self.config.num_layers)
        ]
        episodic_key_strength = [
            torch.zeros(batch_size, self.config.episodic_key_slots, device=device)
            for _ in range(self.config.num_layers)
        ]
        episodic_key_age = [
            torch.zeros(batch_size, self.config.episodic_key_slots, device=device)
            for _ in range(self.config.num_layers)
        ]
        episodic_key_usage = [
            torch.zeros(batch_size, self.config.episodic_key_slots, device=device)
            for _ in range(self.config.num_layers)
        ]

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
        abstraction_gate_mean = x.new_tensor(0.0)
        abstraction_delta_mean = x.new_tensor(0.0)
        abstraction_entropy_mean = x.new_tensor(0.0)
        abstraction_stop_depth_mean = x.new_tensor(0.0)
        abstraction_stopped_ratio_mean = x.new_tensor(0.0)
        abstraction_adequacy_mean = x.new_tensor(0.0)
        abstraction_entity_weight_mean = x.new_tensor(0.0)
        abstraction_relation_weight_mean = x.new_tensor(0.0)
        abstraction_event_weight_mean = x.new_tensor(0.0)
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
                )
                replay_read, slot_replay_mass, replay_stats = self._episodic_replay_read(
                    query_key=replay_query["key"],
                    query_branch=replay_query["branch"],
                    delay_gate=replay_query["delay_gate"],
                    salience_gate=replay_query["salience_gate"],
                    episodic_branch_mass=episodic_branch_mass[layer_index],
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
                memory_mix = torch.sigmoid(layer.memory_fusion(torch.cat([abstract_signal, hidden[layer_index]], dim=-1)))
                replay_mix = torch.sigmoid(layer.replay_fusion(torch.cat([abstract_signal, hidden[layer_index]], dim=-1)))
                short_key_mix = torch.sigmoid(
                    layer.short_key_fusion(torch.cat([abstract_signal, hidden[layer_index]], dim=-1))
                )
                protected_replay = short_key_mix * short_key_read + (1.0 - short_key_mix) * replay_read
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
                branch_write_mass = episodic_write_stats["slot_write_mass"].unsqueeze(-1)
                updated_branch_mass = (
                    episodic_branch_mass[layer_index] * (1.0 - branch_write_mass)
                    + branch_write_mass * current_branch.unsqueeze(1)
                )
                branch_mass_norm = updated_branch_mass.sum(dim=-1, keepdim=True).clamp_min(1e-6)
                episodic_branch_mass[layer_index] = updated_branch_mass / branch_mass_norm
                episodic_replay_hits[layer_index] = torch.clamp(
                    episodic_replay_hits[layer_index] * self.context_spec.config.replay_decay + slot_replay_mass,
                    0.0,
                    6.0,
                )
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
                episodic_keys[layer_index] = episodic_keys[layer_index].detach()
                episodic_values[layer_index] = episodic_values[layer_index].detach()
                episodic_strength[layer_index] = episodic_strength[layer_index].detach()
                episodic_age[layer_index] = episodic_age[layer_index].detach()
                episodic_branch_mass[layer_index] = episodic_branch_mass[layer_index].detach()
                episodic_replay_hits[layer_index] = episodic_replay_hits[layer_index].detach()
                episodic_cold_steps[layer_index] = episodic_cold_steps[layer_index].detach()
                episodic_key_keys[layer_index] = episodic_key_keys[layer_index].detach()
                episodic_key_values[layer_index] = episodic_key_values[layer_index].detach()
                episodic_key_strength[layer_index] = episodic_key_strength[layer_index].detach()
                episodic_key_age[layer_index] = episodic_key_age[layer_index].detach()
                episodic_key_usage[layer_index] = episodic_key_usage[layer_index].detach()
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
                abstraction_gate_mean = abstraction_gate_mean + abstraction_stats["gate_mean"].mean()
                abstraction_delta_mean = abstraction_delta_mean + abstraction_stats["delta_mean"].mean()
                abstraction_entropy_mean = abstraction_entropy_mean + abstraction_stats["entropy"].mean()
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
            "abstraction_gate_mean": float((abstraction_gate_mean / max(total_steps, 1)).detach().cpu()),
            "abstraction_delta_mean": float((abstraction_delta_mean / max(total_steps, 1)).detach().cpu()),
            "abstraction_entropy_mean": float((abstraction_entropy_mean / max(total_steps, 1)).detach().cpu()),
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
            "abstraction_residual_scale_mean": float(
                (abstraction_residual_scale_mean / max(total_steps, 1)).detach().cpu()
            ),
            "overwrite_ratio_mean": float((overwrite_ratio_mean / max(total_steps, 1)).detach().cpu()),
            "protection_mean": float((protection_mean / max(total_steps, 1)).detach().cpu()),
            "usage_mean": float((usage_mean / max(total_steps, 1)).detach().cpu()),
            "age_mean": float((age_mean / max(total_steps, 1)).detach().cpu()),
            "avg_active_slots": float(self.config.router_top_k),
        }
        if not return_aux:
            return {"logits": logits}
        return {"logits": logits, "aux": aux}


def next_token_loss(logits: Tensor, target_ids: Tensor) -> Tensor:
    vocab_size = logits.shape[-1]
    return F.cross_entropy(logits.reshape(-1, vocab_size), target_ids.reshape(-1))
