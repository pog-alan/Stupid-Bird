from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class LongContextTask:
    name: str
    description: str
    primary_metric: str


@dataclass(frozen=True)
class LongContextScenario:
    name: str
    train_length: int
    eval_length: int
    focus_description: str


@dataclass(frozen=True)
class LongContextMeasurement:
    model_name: str
    task_name: str
    parameter_count: int
    train_sequence_length: int
    long_sequence_length: int
    train_loss: float
    in_distribution_token_acc: float
    in_distribution_exact_match: float
    long_context_token_acc: float
    long_context_exact_match: float
    eval_ms_per_batch: float


class LongContextEvaluationSuite:
    """Defines the synthetic long-context tasks used to compare SB-Core against Transformer baselines."""

    def default_tasks(self) -> List[LongContextTask]:
        return [
            LongContextTask(
                name="needle_in_haystack",
                description="Recover a value token span after a long haystack and a query key.",
                primary_metric="exact_match",
            ),
            LongContextTask(
                name="passkey_retrieval",
                description="Retain a short key that appears far earlier than the query cue.",
                primary_metric="exact_match",
            ),
            LongContextTask(
                name="multi_turn_fact_update",
                description="Track whether the most recent fact overrides stale context.",
                primary_metric="updated_fact_accuracy",
            ),
            LongContextTask(
                name="event_chain_replay",
                description="Recover a long event chain in the correct order.",
                primary_metric="sequence_recall",
            ),
        ]

    def report_template(self) -> List[str]:
        return [
            "任务名称",
            "模型",
            "参数量",
            "训练长度",
            "测试长度",
            "训练损失",
            "同分布 token 准确率",
            "同分布 exact match",
            "长上下文 token 准确率",
            "长上下文 exact match",
            "评测耗时（ms / batch）",
        ]
