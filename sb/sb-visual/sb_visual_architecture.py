from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from sb.signal_schema import DynamicSchemaConfig, DynamicSchemaOperator


@dataclass(frozen=True)
class SBVisualConfig:
    input_channels: int = 3
    image_size: int = 224
    patch_size: int = 16
    patch_stride: int = 16
    d_model: int = 128
    state_dim: int = 128
    schema_slots: int = 9
    object_memory_slots: int = 24
    relation_memory_slots: int = 48
    scene_memory_slots: int = 8
    summary_memory_slots: int = 8
    max_patches: int = 196
    relation_top_k: int = 4
    relation_vocab_size: int = 16
    scene_classes: int = 12
    answer_vocab_size: int = 128
    question_vocab_size: int = 512
    question_max_len: int = 64
    schema_anchor_names: Tuple[str, ...] = ("object", "attribute", "relation", "event", "scene")

    def validate(self) -> None:
        if self.input_channels <= 0:
            raise ValueError("input_channels must be positive.")
        if self.image_size <= 0:
            raise ValueError("image_size must be positive.")
        if self.patch_size <= 0 or self.patch_stride <= 0:
            raise ValueError("patch_size and patch_stride must be positive.")
        if self.d_model <= 0 or self.state_dim <= 0:
            raise ValueError("d_model and state_dim must be positive.")
        if self.schema_slots <= 0:
            raise ValueError("schema_slots must be positive.")
        if self.object_memory_slots <= 0 or self.relation_memory_slots <= 0:
            raise ValueError("visual memory slots must be positive.")
        if self.scene_memory_slots <= 0 or self.summary_memory_slots <= 0:
            raise ValueError("scene and summary memory slots must be positive.")
        if self.relation_top_k <= 0:
            raise ValueError("relation_top_k must be positive.")
        if self.relation_vocab_size <= 0:
            raise ValueError("relation_vocab_size must be positive.")
        if self.scene_classes <= 0 or self.answer_vocab_size <= 0:
            raise ValueError("task head dimensions must be positive.")
        if self.question_vocab_size <= 0 or self.question_max_len <= 0:
            raise ValueError("question vocabulary settings must be positive.")
        if len(self.schema_anchor_names) == 0:
            raise ValueError("schema_anchor_names must not be empty.")


@dataclass(frozen=True)
class VisualSignalBatch:
    signals: Tensor
    positions: Tensor
    patch_grid: Tuple[int, int]


@dataclass(frozen=True)
class VisualMemoryState:
    object_keys: Tensor
    object_values: Tensor
    object_schema_mass: Tensor
    relation_keys: Tensor
    relation_values: Tensor
    relation_schema_mass: Tensor
    scene_keys: Tensor
    scene_values: Tensor
    scene_schema_mass: Tensor
    summary_keys: Tensor
    summary_values: Tensor
    summary_schema_mass: Tensor

    def detached(self) -> "VisualMemoryState":
        return VisualMemoryState(
            object_keys=self.object_keys.detach(),
            object_values=self.object_values.detach(),
            object_schema_mass=self.object_schema_mass.detach(),
            relation_keys=self.relation_keys.detach(),
            relation_values=self.relation_values.detach(),
            relation_schema_mass=self.relation_schema_mass.detach(),
            scene_keys=self.scene_keys.detach(),
            scene_values=self.scene_values.detach(),
            scene_schema_mass=self.scene_schema_mass.detach(),
            summary_keys=self.summary_keys.detach(),
            summary_values=self.summary_values.detach(),
            summary_schema_mass=self.summary_schema_mass.detach(),
        )

    def moved_to(self, device: torch.device | str) -> "VisualMemoryState":
        return VisualMemoryState(
            object_keys=self.object_keys.to(device),
            object_values=self.object_values.to(device),
            object_schema_mass=self.object_schema_mass.to(device),
            relation_keys=self.relation_keys.to(device),
            relation_values=self.relation_values.to(device),
            relation_schema_mass=self.relation_schema_mass.to(device),
            scene_keys=self.scene_keys.to(device),
            scene_values=self.scene_values.to(device),
            scene_schema_mass=self.scene_schema_mass.to(device),
            summary_keys=self.summary_keys.to(device),
            summary_values=self.summary_values.to(device),
            summary_schema_mass=self.summary_schema_mass.to(device),
        )


class VisualPatchEncoder(nn.Module):
    def __init__(self, config: SBVisualConfig) -> None:
        super().__init__()
        self.config = config
        self.patch_embed = nn.Conv2d(
            in_channels=config.input_channels,
            out_channels=config.d_model,
            kernel_size=config.patch_size,
            stride=config.patch_stride,
            bias=False,
        )
        self.channel_norm = nn.LayerNorm(config.d_model)
        self.position_mlp = nn.Sequential(
            nn.Linear(2, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model),
        )

    def forward(self, image: Tensor) -> VisualSignalBatch:
        features = self.patch_embed(image)
        batch_size, channels, height, width = features.shape
        patch_tokens = features.flatten(2).transpose(1, 2)
        patch_tokens = self.channel_norm(patch_tokens)

        ys = torch.linspace(-1.0, 1.0, steps=height, device=image.device, dtype=patch_tokens.dtype)
        xs = torch.linspace(-1.0, 1.0, steps=width, device=image.device, dtype=patch_tokens.dtype)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        positions = torch.stack([grid_x, grid_y], dim=-1).reshape(1, height * width, 2)
        positions = positions.expand(batch_size, -1, -1)
        position_bias = self.position_mlp(positions)

        return VisualSignalBatch(
            signals=patch_tokens + position_bias,
            positions=positions,
            patch_grid=(height, width),
        )


class VisualSignalAbstraction(nn.Module):
    def __init__(self, config: SBVisualConfig) -> None:
        super().__init__()
        self.config = config
        self.signal_proj = nn.Linear(config.d_model, config.state_dim)
        self.previous_proj = nn.Linear(config.d_model, config.state_dim)
        self.schema = DynamicSchemaOperator(
            DynamicSchemaConfig(
                state_dim=config.state_dim,
                schema_slots=config.schema_slots,
                anchor_names=config.schema_anchor_names,
            )
        )
        self.residual_gate = nn.Linear(config.state_dim * 2, config.state_dim)

    def forward(self, signals: Tensor) -> Dict[str, Tensor]:
        previous = torch.roll(signals, shifts=1, dims=1)
        previous[:, 0] = signals[:, 0]
        lifted = self.signal_proj(signals)
        previous_lifted = self.previous_proj(previous)
        flat_lifted = lifted.reshape(-1, lifted.shape[-1])
        flat_previous = previous_lifted.reshape(-1, previous_lifted.shape[-1])
        schema_state = self.schema(flat_lifted, flat_previous)
        schema_embedding = schema_state["schema_embedding"].reshape_as(lifted)
        schema_weights = schema_state["schema_weights"].reshape(
            signals.shape[0], signals.shape[1], self.config.schema_slots
        )
        anchor_weights = schema_state["anchor_weights"].reshape(
            signals.shape[0], signals.shape[1], len(self.config.schema_anchor_names)
        )
        gate = torch.sigmoid(self.residual_gate(torch.cat([lifted, schema_embedding], dim=-1)))
        abstracted = lifted + gate * (schema_embedding - lifted)
        return {
            "abstracted_signals": abstracted,
            "schema_weights": schema_weights,
            "anchor_weights": anchor_weights,
            "schema_embedding": schema_embedding,
            "schema_entropy": schema_state["schema_entropy"].reshape(signals.shape[0], signals.shape[1]),
        }


class VisualQuestionEncoder(nn.Module):
    def __init__(self, config: SBVisualConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.question_vocab_size, config.d_model)
        self.proj = nn.Sequential(
            nn.Linear(config.d_model, config.state_dim),
            nn.GELU(),
            nn.Linear(config.state_dim, config.state_dim),
        )

    def forward(self, question_ids: Tensor, question_mask: Tensor | None = None) -> Tensor:
        embedded = self.embedding(question_ids)
        if question_mask is None:
            question_mask = torch.ones_like(question_ids, dtype=embedded.dtype)
        else:
            question_mask = question_mask.to(dtype=embedded.dtype)
        masked = embedded * question_mask.unsqueeze(-1)
        pooled = masked.sum(dim=1) / question_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        return self.proj(pooled)


class SpatialRelationBinder(nn.Module):
    def __init__(self, config: SBVisualConfig) -> None:
        super().__init__()
        self.config = config
        self.relation_mlp = nn.Sequential(
            nn.Linear(config.state_dim * 2 + 2, config.state_dim),
            nn.GELU(),
            nn.Linear(config.state_dim, config.state_dim),
        )

    def forward(self, abstracted_signals: Tensor, positions: Tensor) -> Dict[str, Tensor]:
        pair_distance = torch.cdist(positions, positions)
        topk = min(self.config.relation_top_k + 1, pair_distance.shape[-1])
        neighbor_index = pair_distance.topk(k=topk, largest=False).indices[..., 1:]

        gather_index = neighbor_index.unsqueeze(-1).expand(-1, -1, -1, abstracted_signals.shape[-1])
        neighbor_signals = torch.gather(
            abstracted_signals.unsqueeze(1).expand(-1, abstracted_signals.shape[1], -1, -1),
            2,
            gather_index,
        )
        neighbor_positions = torch.gather(
            positions.unsqueeze(1).expand(-1, positions.shape[1], -1, -1),
            2,
            neighbor_index.unsqueeze(-1).expand(-1, -1, -1, positions.shape[-1]),
        )
        source = abstracted_signals.unsqueeze(2).expand_as(neighbor_signals)
        relative_position = neighbor_positions - positions.unsqueeze(2)
        relation_inputs = torch.cat([source, neighbor_signals, relative_position], dim=-1)
        relation_messages = self.relation_mlp(relation_inputs)
        relation_state = relation_messages.mean(dim=2)
        return {
            "relation_state": relation_state,
            "neighbor_index": neighbor_index,
            "relation_distance_mean": pair_distance.mean(dim=(-1, -2)),
        }


class VisualMemoryRouter(nn.Module):
    def __init__(self, state_dim: int, schema_slots: int) -> None:
        super().__init__()
        self.query_proj = nn.Linear(state_dim * 2 + schema_slots, state_dim)

    def forward(self, visual_state: Tensor, relation_state: Tensor, schema_weights: Tensor) -> Tensor:
        query = self.query_proj(torch.cat([visual_state, relation_state, schema_weights], dim=-1))
        return F.normalize(query, dim=-1, eps=1e-6)


class VisualMemoryBank(nn.Module):
    def __init__(self, config: SBVisualConfig) -> None:
        super().__init__()
        self.config = config

    def initialize_state(self, batch_size: int, device: torch.device | str, dtype: torch.dtype) -> VisualMemoryState:
        schema_fill = 1.0 / float(self.config.schema_slots)
        return VisualMemoryState(
            object_keys=torch.zeros(batch_size, self.config.object_memory_slots, self.config.state_dim, device=device, dtype=dtype),
            object_values=torch.zeros(batch_size, self.config.object_memory_slots, self.config.state_dim, device=device, dtype=dtype),
            object_schema_mass=torch.full(
                (batch_size, self.config.object_memory_slots, self.config.schema_slots),
                schema_fill,
                device=device,
                dtype=dtype,
            ),
            relation_keys=torch.zeros(batch_size, self.config.relation_memory_slots, self.config.state_dim, device=device, dtype=dtype),
            relation_values=torch.zeros(batch_size, self.config.relation_memory_slots, self.config.state_dim, device=device, dtype=dtype),
            relation_schema_mass=torch.full(
                (batch_size, self.config.relation_memory_slots, self.config.schema_slots),
                schema_fill,
                device=device,
                dtype=dtype,
            ),
            scene_keys=torch.zeros(batch_size, self.config.scene_memory_slots, self.config.state_dim, device=device, dtype=dtype),
            scene_values=torch.zeros(batch_size, self.config.scene_memory_slots, self.config.state_dim, device=device, dtype=dtype),
            scene_schema_mass=torch.full(
                (batch_size, self.config.scene_memory_slots, self.config.schema_slots),
                schema_fill,
                device=device,
                dtype=dtype,
            ),
            summary_keys=torch.zeros(batch_size, self.config.summary_memory_slots, self.config.state_dim, device=device, dtype=dtype),
            summary_values=torch.zeros(batch_size, self.config.summary_memory_slots, self.config.state_dim, device=device, dtype=dtype),
            summary_schema_mass=torch.full(
                (batch_size, self.config.summary_memory_slots, self.config.schema_slots),
                schema_fill,
                device=device,
                dtype=dtype,
            ),
        )

    def _read_bank(self, query: Tensor, keys: Tensor, values: Tensor, schema_weights: Tensor, schema_mass: Tensor) -> Dict[str, Tensor]:
        score = torch.einsum("bd,bnd->bn", query, F.normalize(keys + 1e-6, dim=-1, eps=1e-6))
        schema_score = torch.einsum("bd,bnd->bn", schema_weights, schema_mass)
        combined = score + 0.2 * schema_score
        weights = F.softmax(combined, dim=-1)
        readout = torch.einsum("bn,bnd->bd", weights, values)
        return {
            "readout": readout,
            "alignment": (weights * combined).sum(dim=-1),
            "schema_alignment": (weights * schema_score).sum(dim=-1),
        }

    def _write_bank(self, content: Tensor, schema_weights: Tensor, keys: Tensor, values: Tensor, schema_mass: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        similarity = torch.einsum("bd,bnd->bn", F.normalize(content, dim=-1, eps=1e-6), F.normalize(keys + 1e-6, dim=-1, eps=1e-6))
        target_index = similarity.argmax(dim=-1)
        target_one_hot = F.one_hot(target_index, num_classes=keys.shape[1]).to(content.dtype)
        updated_keys = keys * (1.0 - target_one_hot.unsqueeze(-1)) + target_one_hot.unsqueeze(-1) * content.unsqueeze(1)
        updated_values = values * (1.0 - target_one_hot.unsqueeze(-1)) + target_one_hot.unsqueeze(-1) * content.unsqueeze(1)
        updated_schema = schema_mass * (1.0 - target_one_hot.unsqueeze(-1)) + target_one_hot.unsqueeze(-1) * schema_weights.unsqueeze(1)
        updated_schema = updated_schema / updated_schema.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        return updated_keys, updated_values, updated_schema

    def forward(
        self,
        query: Tensor,
        content: Tensor,
        schema_weights: Tensor,
        state: VisualMemoryState,
    ) -> Dict[str, Tensor | VisualMemoryState]:
        object_read = self._read_bank(query, state.object_keys, state.object_values, schema_weights, state.object_schema_mass)
        relation_read = self._read_bank(query, state.relation_keys, state.relation_values, schema_weights, state.relation_schema_mass)
        scene_read = self._read_bank(query, state.scene_keys, state.scene_values, schema_weights, state.scene_schema_mass)
        summary_read = self._read_bank(query, state.summary_keys, state.summary_values, schema_weights, state.summary_schema_mass)

        object_keys, object_values, object_schema = self._write_bank(content, schema_weights, state.object_keys, state.object_values, state.object_schema_mass)
        relation_keys, relation_values, relation_schema = self._write_bank(content, schema_weights, state.relation_keys, state.relation_values, state.relation_schema_mass)
        scene_keys, scene_values, scene_schema = self._write_bank(content, schema_weights, state.scene_keys, state.scene_values, state.scene_schema_mass)
        summary_keys, summary_values, summary_schema = self._write_bank(content, schema_weights, state.summary_keys, state.summary_values, state.summary_schema_mass)

        next_state = VisualMemoryState(
            object_keys=object_keys,
            object_values=object_values,
            object_schema_mass=object_schema,
            relation_keys=relation_keys,
            relation_values=relation_values,
            relation_schema_mass=relation_schema,
            scene_keys=scene_keys,
            scene_values=scene_values,
            scene_schema_mass=scene_schema,
            summary_keys=summary_keys,
            summary_values=summary_values,
            summary_schema_mass=summary_schema,
        )
        return {
            "object_read": object_read["readout"],
            "relation_read": relation_read["readout"],
            "scene_read": scene_read["readout"],
            "summary_read": summary_read["readout"],
            "object_schema_alignment": object_read["schema_alignment"],
            "scene_schema_alignment": scene_read["schema_alignment"],
            "summary_schema_alignment": summary_read["schema_alignment"],
            "state": next_state,
        }


class SceneGraphHead(nn.Module):
    def __init__(self, config: SBVisualConfig) -> None:
        super().__init__()
        self.scene_classifier = nn.Linear(config.state_dim, config.scene_classes)
        self.objectness = nn.Linear(config.state_dim, 1)
        self.box_regressor = nn.Linear(config.state_dim, 4)
        self.relationness = nn.Linear(config.state_dim, 1)
        self.relation_predicate = nn.Linear(config.state_dim, config.relation_vocab_size)
        self.answer_head = nn.Linear(config.state_dim, config.answer_vocab_size)

    def forward(self, fused_state: Tensor, relation_state: Tensor, scene_state: Tensor) -> Dict[str, Tensor]:
        pooled = fused_state.mean(dim=1)
        relation_pooled = relation_state.mean(dim=1)
        scene_logits = self.scene_classifier(scene_state + pooled)
        object_scores = self.objectness(fused_state).squeeze(-1)
        box_deltas = torch.sigmoid(self.box_regressor(fused_state))
        relation_scores = self.relationness(relation_state).squeeze(-1)
        relation_predicate_logits = self.relation_predicate(relation_state)
        answer_logits = self.answer_head(scene_state + 0.5 * pooled + 0.5 * relation_pooled)
        return {
            "scene_logits": scene_logits,
            "object_scores": object_scores,
            "box_deltas": box_deltas,
            "relation_scores": relation_scores,
            "relation_predicate_logits": relation_predicate_logits,
            "answer_logits": answer_logits,
        }


class SBVisualCore(nn.Module):
    def __init__(self, config: SBVisualConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.patch_encoder = VisualPatchEncoder(config)
        self.abstraction = VisualSignalAbstraction(config)
        self.question_encoder = VisualQuestionEncoder(config)
        self.binding = SpatialRelationBinder(config)
        self.router = VisualMemoryRouter(config.state_dim, config.schema_slots)
        self.memory_bank = VisualMemoryBank(config)
        self.scene_pool = nn.Sequential(
            nn.Linear(config.state_dim * 4, config.state_dim),
            nn.GELU(),
            nn.Linear(config.state_dim, config.state_dim),
        )
        self.task_head = SceneGraphHead(config)

    def initialize_memory_state(
        self,
        batch_size: int,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> VisualMemoryState:
        return self.memory_bank.initialize_state(batch_size=batch_size, device=device, dtype=dtype)

    def forward(
        self,
        image: Tensor,
        *,
        text_query: Tensor | None = None,
        question_ids: Tensor | None = None,
        question_mask: Tensor | None = None,
        memory_state: VisualMemoryState | None = None,
        return_state: bool = False,
        return_aux: bool = False,
    ) -> Dict[str, Tensor | VisualMemoryState | Dict[str, float]]:
        patch_batch = self.patch_encoder(image)
        abstraction = self.abstraction(patch_batch.signals)
        relation_pack = self.binding(abstraction["abstracted_signals"], patch_batch.positions)

        visual_state = abstraction["abstracted_signals"]
        if text_query is None and question_ids is not None:
            text_query = self.question_encoder(question_ids, question_mask)
        if text_query is not None:
            visual_state = visual_state + text_query.unsqueeze(1)

        query = self.router(
            visual_state.mean(dim=1),
            relation_pack["relation_state"].mean(dim=1),
            abstraction["schema_weights"].mean(dim=1),
        )
        current_state = memory_state
        if current_state is None:
            current_state = self.initialize_memory_state(
                image.shape[0],
                device=image.device,
                dtype=visual_state.dtype,
            )
        memory = self.memory_bank(
            query=query,
            content=visual_state.mean(dim=1),
            schema_weights=abstraction["schema_weights"].mean(dim=1),
            state=current_state,
        )

        scene_state = self.scene_pool(
            torch.cat(
                [
                    visual_state.mean(dim=1),
                    relation_pack["relation_state"].mean(dim=1),
                    memory["scene_read"],
                    memory["summary_read"],
                ],
                dim=-1,
            )
        )
        heads = self.task_head(visual_state, relation_pack["relation_state"], scene_state)

        aux = {
            "patch_count_mean": float(visual_state.shape[1]),
            "schema_entropy_mean": float(abstraction["schema_entropy"].mean().detach().cpu()),
            "object_schema_alignment_mean": float(memory["object_schema_alignment"].mean().detach().cpu()),
            "summary_schema_alignment_mean": float(memory["summary_schema_alignment"].mean().detach().cpu()),
            "scene_schema_alignment_mean": float(memory["scene_schema_alignment"].mean().detach().cpu()),
            "relation_distance_mean": float(relation_pack["relation_distance_mean"].mean().detach().cpu()),
        }
        payload: Dict[str, Tensor | VisualMemoryState | Dict[str, float]] = {
            "scene_logits": heads["scene_logits"],
            "object_scores": heads["object_scores"],
            "box_deltas": heads["box_deltas"],
            "relation_scores": heads["relation_scores"],
            "relation_predicate_logits": heads["relation_predicate_logits"],
            "answer_logits": heads["answer_logits"],
            "patch_positions": patch_batch.positions,
            "schema_weights": abstraction["schema_weights"],
            "anchor_weights": abstraction["anchor_weights"],
            "scene_state": scene_state,
        }
        if return_aux:
            payload["aux"] = aux
        if return_state:
            payload["state"] = memory["state"]
        return payload
