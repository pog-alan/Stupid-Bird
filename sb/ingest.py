from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from .extractor import ExtractedCandidate
from .quality_gate import CandidateDecision, QualityGate, QualityGateConfig


@dataclass
class KnowledgeEntry:
    item_type: str
    label: str
    normalized: str
    status: str
    occurrences: int = 0
    sources: List[str] = field(default_factory=list)
    last_seen: str = ""


@dataclass
class KnowledgeStore:
    path: Path
    entries: Dict[Tuple[str, str], KnowledgeEntry] = field(default_factory=dict)
    history: List[Dict[str, str]] = field(default_factory=list)

    @classmethod
    def load(cls, path: str | Path) -> "KnowledgeStore":
        path = Path(path)
        if not path.exists():
            return cls(path=path)
        raw = json.loads(path.read_text(encoding="utf-8"))
        entries: Dict[Tuple[str, str], KnowledgeEntry] = {}
        for item in raw.get("entries", []):
            key = (item["item_type"], item["normalized"])
            entries[key] = KnowledgeEntry(
                item_type=item["item_type"],
                label=item["label"],
                normalized=item["normalized"],
                status=item.get("status", "temporary"),
                occurrences=item.get("occurrences", 0),
                sources=list(item.get("sources", [])),
                last_seen=item.get("last_seen", ""),
            )
        return cls(path=path, entries=entries, history=list(raw.get("history", [])))

    def save(self) -> None:
        payload = {
            "entries": [
                {
                    "item_type": entry.item_type,
                    "label": entry.label,
                    "normalized": entry.normalized,
                    "status": entry.status,
                    "occurrences": entry.occurrences,
                    "sources": entry.sources,
                    "last_seen": entry.last_seen,
                }
                for entry in self.entries.values()
            ],
            "history": self.history[-200:],
        }
        self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def stable_entries(self) -> List[KnowledgeEntry]:
        return [entry for entry in self.entries.values() if entry.status == "stable"]


@dataclass
class IngestReport:
    promoted: List[KnowledgeEntry]
    updated: List[KnowledgeEntry]
    dropped: List[Tuple[str, str]]


class Ingestor:
    def __init__(
        self,
        store: KnowledgeStore,
        gate_config: QualityGateConfig | None = None,
    ) -> None:
        self.store = store
        self.gate = QualityGate(gate_config)

    def ingest(
        self,
        candidates: Iterable[ExtractedCandidate],
        conflict_keys: Iterable[Tuple[str, str]] = (),
    ) -> IngestReport:
        decisions = self.gate.evaluate(candidates, conflict_keys=conflict_keys)
        promoted: List[KnowledgeEntry] = []
        updated: List[KnowledgeEntry] = []
        dropped: List[Tuple[str, str]] = []
        now = datetime.now(timezone.utc).isoformat()

        for decision, candidate in zip(decisions, candidates):
            if decision.status == "drop":
                dropped.append(decision.key)
                continue
            entry = self.store.entries.get(decision.key)
            if entry is None:
                entry = KnowledgeEntry(
                    item_type=candidate.item_type,
                    label=candidate.label,
                    normalized=candidate.normalized,
                    status=decision.status,
                    occurrences=1,
                    sources=[candidate.source_url] if candidate.source_url else [],
                    last_seen=now,
                )
                self.store.entries[decision.key] = entry
                updated.append(entry)
            else:
                entry.occurrences += 1
                entry.last_seen = now
                if candidate.label and entry.label == entry.normalized:
                    entry.label = candidate.label
                if candidate.source_url and candidate.source_url not in entry.sources:
                    entry.sources.append(candidate.source_url)
                previous = entry.status
                if previous != "stable":
                    if entry.occurrences >= self.gate.config.min_sources_for_stable and len(entry.sources) >= 2:
                        entry.status = "stable"
                    elif decision.status == "candidate":
                        entry.status = "candidate"
                if entry.status != previous:
                    promoted.append(entry)
                updated.append(entry)

        self.store.history.append(
            {
                "timestamp": now,
                "updated": str(len(updated)),
                "promoted": str(len(promoted)),
                "dropped": str(len(dropped)),
            }
        )
        return IngestReport(promoted=promoted, updated=updated, dropped=dropped)


def apply_stable_entries_to_config(
    store: KnowledgeStore,
    config_path: str | Path,
    output_path: Optional[str | Path] = None,
) -> Path:
    config_path = Path(config_path)
    output_path = Path(output_path) if output_path else config_path
    raw = json.loads(config_path.read_text(encoding="utf-8"))

    object_lexicon = raw.get("object_lexicon", {})
    attribute_lexicon = raw.get("attribute_lexicon", {})
    scene_hint_terms = raw.get("scene_hint_terms", {})

    for entry in store.entries.values():
        if entry.status != "stable":
            continue
        if entry.item_type == "object":
            object_lexicon.setdefault(entry.normalized, [])
        elif entry.item_type == "attribute":
            attribute_lexicon.setdefault(entry.normalized, [])
        elif entry.item_type == "scene_hint":
            scene_hint_terms.setdefault(entry.normalized, [])

    raw["object_lexicon"] = object_lexicon
    raw["attribute_lexicon"] = attribute_lexicon
    raw["scene_hint_terms"] = scene_hint_terms
    output_path.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path
