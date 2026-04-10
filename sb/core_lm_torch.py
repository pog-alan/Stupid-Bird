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
    def __init__(self, state_dim: int, working_slots: int) -> None:
        super().__init__()
        self.working_slots = working_slots
        self.key_proj = nn.Linear(state_dim, state_dim)
        self.value_proj = nn.Linear(state_dim, state_dim)
        self.write_gate = nn.Linear(state_dim, 1)

    def forward(
        self,
        hidden: Tensor,
        working_keys: Tensor,
        working_values: Tensor,
        step: int,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        batch_size, _, state_dim = working_keys.shape
        slot_index = step % self.working_slots
        slot_mask = F.one_hot(
            torch.full((batch_size,), slot_index, device=hidden.device, dtype=torch.long),
            num_classes=self.working_slots,
        ).to(hidden.dtype).unsqueeze(-1)

        gate = torch.sigmoid(self.write_gate(hidden)).unsqueeze(-1)
        new_key = torch.tanh(self.key_proj(hidden)).unsqueeze(1)
        new_value = torch.tanh(self.value_proj(hidden)).unsqueeze(1)

        current_key = torch.sum(working_keys * slot_mask, dim=1, keepdim=True)
        current_value = torch.sum(working_values * slot_mask, dim=1, keepdim=True)

        blended_key = current_key * (1.0 - gate) + new_key * gate
        blended_value = current_value * (1.0 - gate) + new_value * gate

        updated_keys = working_keys * (1.0 - slot_mask) + blended_key * slot_mask
        updated_values = working_values * (1.0 - slot_mask) + blended_value * slot_mask
        return updated_keys, updated_values, gate.squeeze(-1).squeeze(-1)


class SBMiniLayer(nn.Module):
    def __init__(self, state_dim: int, working_slots: int, top_k: int) -> None:
        super().__init__()
        self.router = SBMemoryRouter(state_dim=state_dim, top_k=top_k)
        self.cell = SBRecurrentCell(state_dim=state_dim)
        self.writer = SBMemoryWriter(state_dim=state_dim, working_slots=working_slots)


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

        outputs: List[Tensor] = []
        route_entropy = x.new_tensor(0.0)
        working_ratio = x.new_tensor(0.0)
        write_gate_mean = x.new_tensor(0.0)
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
                working_keys[layer_index], working_values[layer_index], gate = layer.writer(
                    hidden[layer_index],
                    working_keys[layer_index],
                    working_values[layer_index],
                    step=step,
                )
                current = hidden[layer_index]
                weights = route["weights"]
                route_entropy = route_entropy + (-(weights * torch.log(weights + 1e-8)).sum(dim=-1).mean())
                working_ratio = working_ratio + route["working_ratio"]
                write_gate_mean = write_gate_mean + gate.mean()
                total_steps += 1
            outputs.append(self.final_norm(self.dropout(current)))

        stacked = torch.stack(outputs, dim=1)
        logits = self.lm_head(stacked)

        aux = {
            "route_entropy": float((route_entropy / max(total_steps, 1)).detach().cpu()),
            "working_read_ratio": float((working_ratio / max(total_steps, 1)).detach().cpu()),
            "write_gate_mean": float((write_gate_mean / max(total_steps, 1)).detach().cpu()),
            "avg_active_slots": float(self.config.router_top_k),
        }
        if not return_aux:
            return {"logits": logits}
        return {"logits": logits, "aux": aux}


def next_token_loss(logits: Tensor, target_ids: Tensor) -> Tensor:
    vocab_size = logits.shape[-1]
    return F.cross_entropy(logits.reshape(-1, vocab_size), target_ids.reshape(-1))
