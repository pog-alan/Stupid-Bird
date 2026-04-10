from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass(frozen=True)
class SBCoreMiniTorchConfig:
    vocab_size: int
    d_model: int = 96
    state_dim: int = 128
    num_layers: int = 3
    semantic_memory_slots: int = 64
    working_memory_slots: int = 16
    router_top_k: int = 4
    dropout: float = 0.1
    tie_weights: bool = False
    pad_token_id: int = 0
    max_seq_len: int = 256
    working_protection_decay: float = 0.96
    working_usage_decay: float = 0.92
    working_age_increment: float = 0.05
    working_memory_temperature: float = 0.35
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
        if not 0.0 < self.working_protection_decay <= 1.0:
            raise ValueError("working_protection_decay 必须在 (0, 1] 之间。")
        if not 0.0 < self.working_usage_decay <= 1.0:
            raise ValueError("working_usage_decay 必须在 (0, 1] 之间。")
        if not 0.0 <= self.working_age_increment <= 1.0:
            raise ValueError("working_age_increment 必须在 [0, 1] 之间。")
        if self.working_memory_temperature <= 0.0:
            raise ValueError("working_memory_temperature 必须大于 0。")


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


class SBMiniLayer(nn.Module):
    def __init__(
        self,
        state_dim: int,
        working_slots: int,
        top_k: int,
        *,
        protection_decay: float,
        temperature: float,
    ) -> None:
        super().__init__()
        self.router = SBMemoryRouter(state_dim=state_dim, top_k=top_k)
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
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        self.input_proj = nn.Linear(config.d_model, config.state_dim)
        self.layers = nn.ModuleList(
            [
                SBMiniLayer(
                    state_dim=config.state_dim,
                    working_slots=config.working_memory_slots,
                    top_k=config.router_top_k,
                    protection_decay=config.working_protection_decay,
                    temperature=config.working_memory_temperature,
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

        outputs: List[Tensor] = []
        route_entropy = x.new_tensor(0.0)
        working_ratio = x.new_tensor(0.0)
        write_gate_mean = x.new_tensor(0.0)
        merge_preference_mean = x.new_tensor(0.0)
        binding_strength_mean = x.new_tensor(0.0)
        overwrite_ratio_mean = x.new_tensor(0.0)
        protection_mean = x.new_tensor(0.0)
        usage_mean = x.new_tensor(0.0)
        age_mean = x.new_tensor(0.0)
        total_steps = 0

        for step in range(seq_len):
            current = x[:, step, :]
            for layer_index, layer in enumerate(self.layers):
                memory_read, route = layer.router(
                    current=current,
                    previous=hidden[layer_index],
                    working_keys=working_keys[layer_index],
                    working_values=working_values[layer_index],
                    semantic_keys=self.semantic_keys[layer_index],
                    semantic_values=self.semantic_values[layer_index],
                )
                hidden[layer_index] = layer.cell(current, hidden[layer_index], memory_read)
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
