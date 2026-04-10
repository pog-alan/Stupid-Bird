from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass(frozen=True)
class SBCoreConfig:
    vocab_size: int
    d_model: int = 512
    num_layers: int = 8
    state_dim: int = 1024
    memory_slots: int = 2048
    router_top_k: int = 8
    use_attention: bool = False
    use_kv_cache: bool = False

    def validate(self) -> None:
        if self.use_attention:
            raise ValueError("SB-Core 实验线不允许开启 self-attention。")
        if self.use_kv_cache:
            raise ValueError("SB-Core 实验线不允许依赖 KV cache。")
        if self.router_top_k <= 0:
            raise ValueError("router_top_k 必须大于 0。")


@dataclass
class SBLayerState:
    values: List[float]


@dataclass
class SBMemoryRead:
    active_slots: List[int] = field(default_factory=list)
    slot_scores: Dict[int, float] = field(default_factory=dict)
    summary: List[float] = field(default_factory=list)


@dataclass
class SBTokenStep:
    token_id: int
    embedding_dim: int
    layer_states: List[SBLayerState]
    memory_reads: List[SBMemoryRead]


class SBCoreModelSpec:
    """SB-Core 的架构说明对象，不包含具体张量实现。"""

    def __init__(self, config: SBCoreConfig) -> None:
        self.config = config
        self.config.validate()

    def recurrent_update_equation(self) -> str:
        return (
            "u_t^l = W_u[h_{t-1}^l; h_t^{l-1}; r_t^l; e_t], "
            "g_t^l = sigmoid(W_g u_t^l), "
            "c_t^l = phi(W_c u_t^l), "
            "h_t^l = (1-g_t^l) * h_{t-1}^l + g_t^l * c_t^l"
        )

    def memory_flow_equation(self) -> str:
        return "q_t = Router(h_t), I_t = topk(score(q_t, K), k), r_t = Read(M, I_t), M <- Update(M, I_t, z_t)"

    def output_equation(self) -> str:
        return "logits_t = W_o z_t, p(x_{t+1}) = softmax(logits_t)"

    def build_empty_step(self, token_id: int) -> SBTokenStep:
        return SBTokenStep(
            token_id=token_id,
            embedding_dim=self.config.d_model,
            layer_states=[SBLayerState(values=[]) for _ in range(self.config.num_layers)],
            memory_reads=[SBMemoryRead() for _ in range(self.config.num_layers)],
        )
