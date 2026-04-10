from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class TinyTransformerConfig:
    vocab_size: int
    d_model: int = 96
    num_layers: int = 2
    num_heads: int = 4
    ff_multiplier: int = 4
    dropout: float = 0.1
    max_seq_len: int = 256
    pad_token_id: int = 0
    tie_weights: bool = False

    def validate(self) -> None:
        if self.d_model % self.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive.")
        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive.")


class TinyTransformerLM(nn.Module):
    def __init__(self, config: TinyTransformerConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=config.d_model,
                    nhead=config.num_heads,
                    dim_feedforward=config.d_model * config.ff_multiplier,
                    dropout=config.dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_weights:
            self.lm_head.weight = self.embedding.weight

    @staticmethod
    def _causal_mask(seq_len: int, device: torch.device) -> Tensor:
        return torch.triu(
            torch.ones((seq_len, seq_len), dtype=torch.bool, device=device),
            diagonal=1,
        )

    def forward(self, input_ids: Tensor, return_aux: bool = True) -> Dict[str, Tensor | Dict[str, float]]:
        batch_size, seq_len = input_ids.shape
        if seq_len > self.config.max_seq_len:
            raise ValueError(f"seq_len={seq_len} exceeds max_seq_len={self.config.max_seq_len}.")

        device = input_ids.device
        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        hidden = self.embedding(input_ids) + self.position_embedding(positions)
        hidden = self.dropout(hidden)

        attn_mask = self._causal_mask(seq_len, device=device)
        padding_mask = input_ids.eq(self.config.pad_token_id)
        for layer in self.layers:
            hidden = layer(
                hidden,
                src_mask=attn_mask,
                src_key_padding_mask=padding_mask,
            )

        hidden = self.final_norm(hidden)
        logits = self.lm_head(hidden)
        if not return_aux:
            return {"logits": logits}
        return {
            "logits": logits,
            "aux": {
                "attention_tokens_per_step": float(seq_len),
                "num_layers": float(self.config.num_layers),
            },
        }
