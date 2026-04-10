from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass(frozen=True)
class LossWeights:
    lm: float = 1.0
    sparse: float = 0.02
    balance: float = 0.01
    write_stability: float = 0.01


@dataclass(frozen=True)
class TrainLMConfig:
    total_tokens: int = 100_000_000
    batch_size_tokens: int = 131_072
    learning_rate: float = 3e-4
    warmup_steps: int = 2000
    max_steps: int = 100_000
    gradient_clip: float = 1.0
    loss_weights: LossWeights = field(default_factory=LossWeights)


@dataclass(frozen=True)
class ExperimentStage:
    stage_id: str
    goal: str
    tasks: List[str]
    acceptance: List[str]


@dataclass(frozen=True)
class CurriculumStage:
    stage_id: str
    prefix_len: int
    filler_len: int
    key_length: int = 2
    steps: int = 120

    @property
    def sequence_length(self) -> int:
        return 1 + self.prefix_len + 1 + self.key_length + self.filler_len + 1 + self.key_length + 1


class SBCoreTrainingPlan:
    """定义 SB-Core 从玩具任务到小型语言模型的训练路线。"""

    def __init__(self, config: TrainLMConfig | None = None) -> None:
        self.config = config or TrainLMConfig()

    def build_stages(self) -> List[ExperimentStage]:
        return [
            ExperimentStage(
                stage_id="P0",
                goal="验证架构能收敛、能路由、能读写记忆。",
                tasks=["copy", "reverse", "passkey retrieval", "long-range matching"],
                acceptance=[
                    "训练不发散",
                    "路由不塌缩",
                    "长序列准确率高于随机基线",
                ],
            ),
            ExperimentStage(
                stage_id="P1",
                goal="验证 SB-Core 能做标准语言建模。",
                tasks=["tiny next-token prediction", "small corpus perplexity"],
                acceptance=[
                    "val loss 稳定下降",
                    "与同规模 tiny Transformer 可比较",
                ],
            ),
            ExperimentStage(
                stage_id="P2",
                goal="验证长上下文效率与记忆稳定性。",
                tasks=["needle in haystack", "multi-turn fact recall", "event chain replay"],
                acceptance=[
                    "长序列下性能下降慢于基线",
                    "推理内存增长更平滑",
                ],
            ),
        ]

    def baseline_rules(self) -> List[str]:
        return [
            "相同 tokenizer。",
            "相同训练 token 数。",
            "相同参数量级。",
            "相同优化器与学习率调度。",
            "相同硬件预算。",
        ]

    def tracked_metrics(self) -> List[str]:
        return [
            "train loss",
            "validation loss",
            "perplexity",
            "tokens/s",
            "memory usage",
            "active slots per token",
            "slot usage entropy",
        ]

    def passkey_curriculum(self) -> List[CurriculumStage]:
        return [
            CurriculumStage(stage_id="C0", prefix_len=4, filler_len=4, steps=120),
            CurriculumStage(stage_id="C1", prefix_len=6, filler_len=8, steps=120),
            CurriculumStage(stage_id="C2", prefix_len=10, filler_len=14, steps=120),
            CurriculumStage(stage_id="C3", prefix_len=14, filler_len=20, steps=120),
        ]

    def fixed_passkey_baseline(self) -> CurriculumStage:
        return CurriculumStage(stage_id="B0", prefix_len=6, filler_len=8, steps=220)
