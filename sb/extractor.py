from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

from .ontology import SBV01Ontology
from .parser import SBV01Parser
from .v01_types import ParsedScene


@dataclass(frozen=True)
class ExtractedCandidate:
    item_type: str
    label: str
    normalized: str
    confidence: float
    evidence: str
    source_url: Optional[str] = None


class SimpleExtractor:
    def __init__(self, ontology: SBV01Ontology) -> None:
        self.ontology = ontology
        self.parser = SBV01Parser(ontology)

    def extract(self, text: str, source_url: Optional[str] = None) -> List[ExtractedCandidate]:
        parsed = self.parser.parse(text)
        return self._from_parsed(parsed, source_url)

    def _from_parsed(self, parsed: ParsedScene, source_url: Optional[str]) -> List[ExtractedCandidate]:
        candidates: List[ExtractedCandidate] = []
        for obj in parsed.objects:
            candidates.append(
                ExtractedCandidate(
                    item_type="object",
                    label=obj.label,
                    normalized=obj.normalized,
                    confidence=obj.confidence,
                    evidence=obj.label,
                    source_url=source_url,
                )
            )
        for attr in parsed.attributes:
            candidates.append(
                ExtractedCandidate(
                    item_type="attribute",
                    label=attr.label,
                    normalized=attr.normalized,
                    confidence=attr.confidence,
                    evidence=attr.label,
                    source_url=source_url,
                )
            )
        for rel in parsed.relations:
            candidates.append(
                ExtractedCandidate(
                    item_type="relation",
                    label=rel.label,
                    normalized=rel.normalized,
                    confidence=rel.confidence,
                    evidence=f"{rel.source_label}->{rel.label}->{rel.target_label}",
                    source_url=source_url,
                )
            )
        for event in parsed.events:
            candidates.append(
                ExtractedCandidate(
                    item_type="event",
                    label=event.event_type,
                    normalized=event.event_type,
                    confidence=event.confidence,
                    evidence=event.event_type,
                    source_url=source_url,
                )
            )
        for hint in parsed.scene_hints:
            candidates.append(
                ExtractedCandidate(
                    item_type="scene_hint",
                    label=hint.label,
                    normalized=hint.label,
                    confidence=hint.confidence,
                    evidence=hint.label,
                    source_url=source_url,
                )
            )
        return candidates


def merge_candidates(candidates: Iterable[ExtractedCandidate]) -> List[ExtractedCandidate]:
    deduped = {}
    for item in candidates:
        key = (item.item_type, item.normalized)
        existing = deduped.get(key)
        if existing is None or item.confidence > existing.confidence:
            deduped[key] = item
    return list(deduped.values())
