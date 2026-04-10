from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass(frozen=True)
class MemoryBankConfig:
    working_slots: int = 256
    episodic_slots: int = 1024
    semantic_slots: int = 2048
    slot_dim: int = 512
    write_threshold: float = 0.7
    decay: float = 0.995


@dataclass
class MemorySlot:
    slot_id: int
    memory_type: str
    key: List[float] = field(default_factory=list)
    value: List[float] = field(default_factory=list)
    usage: float = 0.0
    last_update_step: int = 0
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass
class MemoryUpdatePlan:
    read_slots: List[int] = field(default_factory=list)
    write_slots: List[int] = field(default_factory=list)
    write_gate: float = 0.0
    reason: str = ""


class SparseMemoryBankSpec:
    """描述 SB-Core 应如何组织和更新外部记忆。"""

    def __init__(self, config: MemoryBankConfig | None = None) -> None:
        self.config = config or MemoryBankConfig()

    def slot_budget(self) -> Dict[str, int]:
        return {
            "working": self.config.working_slots,
            "episodic": self.config.episodic_slots,
            "semantic": self.config.semantic_slots,
        }

    def write_rule(self) -> str:
        return (
            "只允许在 write_gate >= write_threshold 时写入；"
            "working memory 快写快忘，semantic memory 慢写慢忘。"
        )

    def anti_pollution_constraints(self) -> List[str]:
        return [
            "高频低置信输入不能直接进入 semantic memory。",
            "写入前需要检查槽位冲突和重复度。",
            "长期低使用率槽位允许衰减和重分配。",
        ]
