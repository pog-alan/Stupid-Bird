from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


CORE_OBJECTIVE_NON_ATTENTION_RECALL = "non_attention_long_range_recall"
DEFAULT_RECALL_BANKS = ("working", "episodic", "key", "summary", "scene")


@dataclass(frozen=True)
class SBCoreConfig:
    vocab_size: int
    d_model: int = 512
    num_layers: int = 8
    state_dim: int = 1024
    memory_slots: int = 2048
    router_top_k: int = 8
    recall_horizon: int = 4096
    objective: str = CORE_OBJECTIVE_NON_ATTENTION_RECALL
    memory_banks: Tuple[str, ...] = DEFAULT_RECALL_BANKS
    use_attention: bool = False
    use_kv_cache: bool = False

    def validate(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError("vocab_size 必须大于 0。")
        if self.d_model <= 0 or self.state_dim <= 0:
            raise ValueError("d_model 和 state_dim 必须大于 0。")
        if self.num_layers <= 0:
            raise ValueError("num_layers 必须大于 0。")
        if self.memory_slots <= 0:
            raise ValueError("memory_slots 必须大于 0。")
        if self.router_top_k <= 0:
            raise ValueError("router_top_k 必须大于 0。")
        if self.recall_horizon <= 0:
            raise ValueError("recall_horizon 必须大于 0。")
        if not self.memory_banks:
            raise ValueError("memory_banks 不能为空。")
        if self.objective != CORE_OBJECTIVE_NON_ATTENTION_RECALL:
            raise ValueError(f"SB-Core 当前只收窄为目标：{CORE_OBJECTIVE_NON_ATTENTION_RECALL}。")
        if self.use_attention:
            raise ValueError("SB-Core 实验线不允许启用 self-attention。")
        if self.use_kv_cache:
            raise ValueError("SB-Core 实验线不允许依赖 KV cache。")


@dataclass(frozen=True)
class SBRecallBankSpec:
    name: str
    symbol: str
    role: str
    slot_count_symbol: str
    decay_symbol: str
    writable: bool = True

    def as_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "symbol": self.symbol,
            "role": self.role,
            "slot_count_symbol": self.slot_count_symbol,
            "decay_symbol": self.decay_symbol,
            "writable": self.writable,
        }


@dataclass(frozen=True)
class SBRecallMathSpec:
    """SB-Core 当前收窄后的数学对象：非 attention 长程召回网络。

    这里描述的是架构约束与计算语义，不绑定具体 PyTorch 实现。
    """

    name: str = "SB-Core Non-Attention Long-Range Recall Network"
    objective: str = CORE_OBJECTIVE_NON_ATTENTION_RECALL
    banks: Tuple[SBRecallBankSpec, ...] = (
        SBRecallBankSpec("working", "M^W", "短期局部状态、最近 signal 的快速覆盖", "N_W", "rho_W"),
        SBRecallBankSpec("episodic", "M^E", "跨段事件片段、延迟召回内容", "N_E", "rho_E"),
        SBRecallBankSpec("key", "M^K", "passkey / delayed recall 的键中心索引", "N_K", "rho_K"),
        SBRecallBankSpec("summary", "M^S", "多 token 合并后的分层摘要", "N_S", "rho_S"),
        SBRecallBankSpec("scene", "M^C", "更长周期的场景级/任务级稳定表征", "N_C", "rho_C"),
    )
    branches: Tuple[str, ...] = ("entity", "relation", "event")
    forbidden_operations: Tuple[str, ...] = ("self_attention", "cross_token_attention_matrix", "transformer_kv_cache")

    def axioms(self) -> List[str]:
        return [
            "输入序列为 x_1:T，x_t 属于离散词表 V；模型按时间递推处理，不构造 T×T token 注意力矩阵。",
            "任意时刻 t 的输出只依赖当前输入 x_t、上一时刻状态 S_{t-1}、以及有界记忆库 M_{t-1}。",
            "跨长程信息通过稀疏召回 read(M, q, top_k) 与受控写入 write(M, z) 传递。",
            "禁止使用 Transformer KV cache；可使用 SB State Cache 保存融合后的 S_t，但它不是 key/value attention 历史。",
            "训练目标暂时收窄为长程召回：给定前文中的 key/value、needle 或 delayed signal，在查询位置恢复目标 y_t。",
        ]

    def symbols(self) -> Dict[str, str]:
        return {
            "x_t": "第 t 个输入 token 或 signal id",
            "e_t": "输入嵌入 E[x_t]",
            "a_t^m": "第 m 层 signal 抽象结果",
            "c_t": "动态 schema 分布",
            "h_t^l": "第 l 层递归隐藏状态",
            "M_t^b": "第 b 类记忆库，b 属于 {working, episodic, key, summary, scene}",
            "K_t^b,V_t^b": "记忆库 b 的键和值矩阵",
            "s_t^b,u_t^b,g_t^b": "记忆强度、使用度、年龄/冷却状态",
            "q_t^{l,b}": "第 l 层面向记忆库 b 的召回查询",
            "I_t^{l,b}": "top-k 激活槽集合",
            "r_t^{l,b}": "从记忆库 b 读出的召回向量",
            "z_t^l": "融合当前 signal、递归状态与召回结果后的层输出",
            "S_t": "完整连续状态，包括所有 h、记忆库、schema buffer 与统计量",
        }

    def state_definition(self) -> str:
        return (
            "S_t = ({h_t^l}_{l=1..L}, {M_t^b}_{b in B}, B_t^summary, B_t^scene, c_t, meta_t), "
            "M_t^b = (K_t^b, V_t^b, strength_t^b, age_t^b, usage_t^b, schema_mass_t^b)"
        )

    def signal_abstraction_equation(self) -> str:
        return (
            "a_t^0 = P_e E[x_t]; "
            "a_t^{m+1}, c_t^{m+1}, stop_t^m = A_m(a_t^m, c_t^m), "
            "m = 0..M-1; "
            "a_t = a_t^{m*}, where m* = min{m | stop_t^m >= tau_stop}"
        )

    def recall_read_equation(self) -> str:
        return (
            "q_t^{l,b} = Q_b[h_{t-1}^l; z_t^{l-1}; a_t; c_t]; "
            "score_i^{l,b} = sim(q_t^{l,b}, K_{t-1,i}^b)/tau_b "
            "+ alpha_b strength_{t-1,i}^b - beta_b age_{t-1,i}^b + gamma_b schema_align(c_t, schema_i^b); "
            "I_t^{l,b} = TopK(score^{l,b}, k); "
            "r_t^{l,b} = sum_{i in I_t^{l,b}} softmax(score_i^{l,b}) V_{t-1,i}^b"
        )

    def recurrent_fusion_equation(self) -> str:
        return (
            "r_t^l = Fuse_b({r_t^{l,b}}_{b in B}); "
            "u_t^l = [h_{t-1}^l; z_t^{l-1}; a_t; r_t^l; c_t]; "
            "g_t^l = sigmoid(W_g^l u_t^l); "
            "candidate_t^l = phi(W_c^l u_t^l); "
            "h_t^l = (1 - g_t^l) * h_{t-1}^l + g_t^l * candidate_t^l; "
            "z_t^l = Norm(W_z^l[h_t^l; r_t^l; a_t])"
        )

    def memory_write_equation(self) -> str:
        return (
            "write_t^b = sigmoid(W_w^b[z_t^L; a_t; c_t]); "
            "hat_k_t^b = K_b[z_t^L; a_t; c_t], hat_v_t^b = V_b[z_t^L; a_t; c_t]; "
            "j_t^b = argmin_i overwrite_cost_i^b, "
            "overwrite_cost_i^b = strength_i^b + protection_i^b - eta_b age_i^b; "
            "K_{t,j}^b = (1-write_t^b)K_{t-1,j}^b + write_t^b hat_k_t^b; "
            "V_{t,j}^b = (1-write_t^b)V_{t-1,j}^b + write_t^b hat_v_t^b; "
            "strength_t^b = rho_b strength_{t-1}^b + write_t^b one_hot(j_t^b)"
        )

    def key_centric_replay_equation(self) -> str:
        return (
            "q_t^replay = R(h_t^L, a_t, c_t, key_focus_t, delay_t); "
            "I_t^K = TopK(sim(q_t^replay, K_{t-1}^K) + key_usage_bonus - age_penalty, k_K); "
            "r_t^replay = Read(M_{t-1}^K, I_t^K); "
            "z_t^L <- z_t^L + lambda_replay Gate(z_t^L, r_t^replay) * r_t^replay"
        )

    def summary_scene_equation(self) -> str:
        return (
            "B_t^summary = rho_buf B_{t-1}^summary + delta_t^summary z_t^L; "
            "if boundary_t^summary > tau_summary: write(M^S, B_t^summary); "
            "B_t^scene = rho_scene B_{t-1}^scene + delta_t^scene Read(M_t^S); "
            "if boundary_t^scene > tau_scene: write(M^C, B_t^scene)"
        )

    def output_equation(self) -> str:
        return "logits_t = W_o z_t^L; p_theta(y_t | x_1:t, S_0) = softmax(logits_t)"

    def objective_function(self) -> str:
        return (
            "L = CE(y_t, p_theta) "
            "+ lambda_recall L_recall "
            "+ lambda_route L_sparse_route "
            "+ lambda_schema L_schema_align "
            "+ lambda_write L_write_stability "
            "+ lambda_forget L_forgetting_control"
        )

    def complexity_claim(self) -> str:
        return (
            "不构造 T×T attention，因此没有 O(T^2 d) 注意力项。"
            "若每步对每个记忆库全槽打分，复杂度为 O(T * L * sum_b N_b * d)。"
            "若记忆检索实现为 ANN 或分桶 top-k，目标复杂度为 O(T * L * |B| * k * d + 检索索引开销)。"
            "状态空间为 O(L * sum_b N_b * d)，SB State Cache 只保存 S_t，不随历史 token 线性保存 KV。"
        )

    def acceptance_criteria(self) -> List[str]:
        return [
            "配置层面 use_attention=False 且 use_kv_cache=False，开启即报错。",
            "passkey_retrieval / delayed_recall / needle_in_haystack 至少一项 carry on 优于 carry off。",
            "长上下文阶段 summary_schema_alignment_mean 和 scene_schema_alignment_mean 稳定大于 0。",
            "State Cache 的 reused_tokens 增长时，computed_tokens 只覆盖新增 suffix。",
            "实验报告必须同时输出召回准确率、精确匹配率、cache 命中率、schema 链路激活率。",
        ]

    def equations(self) -> Dict[str, str]:
        return {
            "state": self.state_definition(),
            "signal_abstraction": self.signal_abstraction_equation(),
            "recall_read": self.recall_read_equation(),
            "recurrent_fusion": self.recurrent_fusion_equation(),
            "memory_write": self.memory_write_equation(),
            "key_centric_replay": self.key_centric_replay_equation(),
            "summary_scene": self.summary_scene_equation(),
            "output": self.output_equation(),
            "objective": self.objective_function(),
            "complexity": self.complexity_claim(),
        }

    def as_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "objective": self.objective,
            "branches": list(self.branches),
            "forbidden_operations": list(self.forbidden_operations),
            "banks": [bank.as_dict() for bank in self.banks],
            "axioms": self.axioms(),
            "symbols": self.symbols(),
            "equations": self.equations(),
            "acceptance_criteria": self.acceptance_criteria(),
        }


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
    """SB-Core 的轻量形式化说明对象，不包含具体张量实现。"""

    def __init__(self, config: SBCoreConfig, math_spec: SBRecallMathSpec | None = None) -> None:
        self.config = config
        self.config.validate()
        self.math_spec = math_spec or SBRecallMathSpec()

    def target_statement(self) -> str:
        return (
            "SB-Core 当前目标收窄为非 attention 长程召回网络："
            "在不使用 self-attention 与 KV cache 的前提下，依靠稀疏记忆读写、"
            "key-centric replay、分层摘要和 state cache 完成长程信息恢复。"
        )

    def recurrent_update_equation(self) -> str:
        return self.math_spec.recurrent_fusion_equation()

    def memory_flow_equation(self) -> str:
        return self.math_spec.recall_read_equation() + "; " + self.math_spec.memory_write_equation()

    def output_equation(self) -> str:
        return self.math_spec.output_equation()

    def formal_summary(self) -> Dict[str, object]:
        return {
            "target": self.target_statement(),
            "config": {
                "vocab_size": self.config.vocab_size,
                "d_model": self.config.d_model,
                "num_layers": self.config.num_layers,
                "state_dim": self.config.state_dim,
                "memory_slots": self.config.memory_slots,
                "router_top_k": self.config.router_top_k,
                "recall_horizon": self.config.recall_horizon,
                "objective": self.config.objective,
                "use_attention": self.config.use_attention,
                "use_kv_cache": self.config.use_kv_cache,
            },
            "math": self.math_spec.as_dict(),
        }

    def build_empty_step(self, token_id: int) -> SBTokenStep:
        return SBTokenStep(
            token_id=token_id,
            embedding_dim=self.config.d_model,
            layer_states=[SBLayerState(values=[]) for _ in range(self.config.num_layers)],
            memory_reads=[SBMemoryRead() for _ in range(self.config.num_layers)],
        )
