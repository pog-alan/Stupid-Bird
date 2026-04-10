from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Tuple


@dataclass(frozen=True)
class RouterConfig:
    top_k: int = 8
    min_score: float = 0.0
    load_balance_weight: float = 0.01
    route_entropy_weight: float = 0.001


@dataclass(frozen=True)
class RoutingDecision:
    selected_slots: Tuple[int, ...]
    scores: Dict[int, float]


class SparseRouterSpec:
    def __init__(self, config: RouterConfig | None = None) -> None:
        self.config = config or RouterConfig()

    def select_top_k(self, scores: Mapping[int, float]) -> RoutingDecision:
        filtered = {
            slot_id: score
            for slot_id, score in scores.items()
            if score >= self.config.min_score
        }
        ranked = sorted(filtered.items(), key=lambda item: item[1], reverse=True)[: self.config.top_k]
        return RoutingDecision(
            selected_slots=tuple(slot_id for slot_id, _ in ranked),
            scores={slot_id: score for slot_id, score in ranked},
        )

    def routing_losses(self) -> List[str]:
        return [
            "稀疏损失：约束单步激活槽位数量。",
            "负载均衡损失：避免大量 token 固定拥塞在同一组槽位。",
            "路由熵约束：避免过早塌缩成单一路径。",
        ]

    def explain(self, scores: Mapping[int, float]) -> str:
        decision = self.select_top_k(scores)
        if not decision.selected_slots:
            return "当前没有槽位超过路由阈值。"
        return f"当前激活槽位：{list(decision.selected_slots)}"
