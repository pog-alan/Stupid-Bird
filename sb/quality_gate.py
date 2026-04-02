from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

from .extractor import ExtractedCandidate


@dataclass(frozen=True)
class QualityGateConfig:
    promote_to_candidate: float = 0.85
    promote_to_temporary: float = 0.70
    min_sources_for_stable: int = 3
    conflict_penalty: float = 0.15


@dataclass
class CandidateDecision:
    key: Tuple[str, str]
    status: str
    score: float
    evidence: str


class QualityGate:
    def __init__(self, config: QualityGateConfig | None = None) -> None:
        self.config = config or QualityGateConfig()

    def evaluate(
        self,
        candidates: Iterable[ExtractedCandidate],
        conflict_keys: Iterable[Tuple[str, str]] = (),
    ) -> List[CandidateDecision]:
        conflicts = set(conflict_keys)
        decisions: List[CandidateDecision] = []
        for item in candidates:
            score = item.confidence
            key = (item.item_type, item.normalized)
            if key in conflicts:
                score = max(0.0, score - self.config.conflict_penalty)
            if score >= self.config.promote_to_candidate:
                status = "candidate"
            elif score >= self.config.promote_to_temporary:
                status = "temporary"
            else:
                status = "drop"
            decisions.append(
                CandidateDecision(
                    key=key,
                    status=status,
                    score=score,
                    evidence=item.evidence,
                )
            )
        return decisions
