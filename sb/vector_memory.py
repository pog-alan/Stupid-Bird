from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

from .embedding_backends import EmbeddingEncoder, HashedVectorEncoder
from .ingest import KnowledgeEntry
from .ontology import SBV01Ontology


@dataclass(frozen=True)
class MemoryRecord:
    memory_id: str
    space_id: str
    space_type: str
    label: str
    text: str
    supports: Tuple[str, ...] = ()
    aliases: Tuple[str, ...] = ()


@dataclass(frozen=True)
class VectorHit:
    memory_id: str
    space_id: str
    space_type: str
    label: str
    text: str
    score: float
    supports: Tuple[str, ...] = ()
    aliases: Tuple[str, ...] = ()


class VectorMemoryIndex:
    def __init__(
        self,
        records: Sequence[MemoryRecord],
        encoder: EmbeddingEncoder | None = None,
    ) -> None:
        self.records = list(records)
        self.encoder = encoder or HashedVectorEncoder()
        encoded = self.encoder.encode_documents([record.text for record in self.records])
        self._vectors: Dict[str, List[float]] = {
            record.memory_id: vector
            for record, vector in zip(self.records, encoded)
        }

    @classmethod
    def from_ontology(
        cls,
        ontology: SBV01Ontology,
        stable_entries: Sequence[KnowledgeEntry] | None = None,
        encoder: EmbeddingEncoder | None = None,
    ) -> "VectorMemoryIndex":
        records: List[MemoryRecord] = []

        for label, aliases in ontology.object_lexicon.items():
            records.append(
                MemoryRecord(
                    memory_id=f"mem_object_{label}",
                    space_id=ontology.concept_space_id(label),
                    space_type="concept",
                    label=label,
                    text=" ".join((label, *aliases)),
                    supports=tuple(
                        scene.label
                        for scene in ontology.scene_spaces.values()
                        if label in scene.support_objects
                    ),
                    aliases=aliases,
                )
            )

        for label, aliases in ontology.attribute_lexicon.items():
            records.append(
                MemoryRecord(
                    memory_id=f"mem_attribute_{label}",
                    space_id=ontology.attribute_space_id(label),
                    space_type="attribute",
                    label=label,
                    text=" ".join((label, *aliases)),
                    supports=tuple(
                        scene.label
                        for scene in ontology.scene_spaces.values()
                        if label in scene.support_attributes
                    ),
                    aliases=aliases,
                )
            )

        relation_labels = sorted(
            {
                relation["relation"]
                for relation in ontology.relation_patterns
                if "relation" in relation
            }
            | {label for scene in ontology.scene_spaces.values() for label in scene.support_relations}
        )
        for label in relation_labels:
            records.append(
                MemoryRecord(
                    memory_id=f"mem_relation_{label}",
                    space_id=ontology.relation_space_id(label),
                    space_type="relation",
                    label=label,
                    text=label,
                    supports=tuple(
                        scene.label
                        for scene in ontology.scene_spaces.values()
                        if label in scene.support_relations
                    ),
                )
            )

        event_labels = sorted(
            {
                event["event"]
                for event in ontology.event_patterns
                if "event" in event
            }
            | {label for scene in ontology.scene_spaces.values() for label in scene.support_events}
        )
        for label in event_labels:
            records.append(
                MemoryRecord(
                    memory_id=f"mem_event_{label}",
                    space_id=f"event_{label}",
                    space_type="event",
                    label=label,
                    text=label,
                    supports=tuple(
                        scene.label
                        for scene in ontology.scene_spaces.values()
                        if label in scene.support_events
                    ),
                )
            )

        for scene in ontology.scene_spaces.values():
            scene_text_parts = [
                scene.label,
                *scene.support_objects,
                *scene.support_attributes,
                *scene.support_relations,
                *scene.support_events,
            ]
            records.append(
                MemoryRecord(
                    memory_id=f"mem_scene_{scene.label}",
                    space_id=ontology.scene_space_id(scene.label),
                    space_type="scene",
                    label=scene.label,
                    text=" ".join(scene_text_parts),
                    supports=(scene.label,),
                )
            )

        if stable_entries:
            existing_keys = {
                (record.space_type, record.label)
                for record in records
            }
            for entry in stable_entries:
                record = _record_from_knowledge_entry(entry, ontology)
                if record is None:
                    continue
                key = (record.space_type, record.label)
                if key in existing_keys:
                    continue
                existing_keys.add(key)
                records.append(record)

        return cls(records, encoder=encoder)

    def search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.15,
        allowed_types: Iterable[str] | None = None,
    ) -> List[VectorHit]:
        if not query.strip():
            return []
        query_vector = self.encoder.encode_query(query)
        allowed = set(allowed_types or ())
        hits: List[VectorHit] = []
        for record in self.records:
            if allowed and record.space_type not in allowed:
                continue
            score = _cosine(query_vector, self._vectors[record.memory_id])
            if score < min_score:
                continue
            hits.append(
                VectorHit(
                    memory_id=record.memory_id,
                    space_id=record.space_id,
                    space_type=record.space_type,
                    label=record.label,
                    text=record.text,
                    score=score,
                    supports=record.supports,
                    aliases=record.aliases,
                )
            )
        hits.sort(key=lambda item: item.score, reverse=True)
        return hits[:top_k]


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm <= 0.0 or right_norm <= 0.0:
        return 0.0
    return numerator / (left_norm * right_norm)
def _record_from_knowledge_entry(
    entry: KnowledgeEntry,
    ontology: SBV01Ontology,
) -> MemoryRecord | None:
    if entry.item_type == "object":
        aliases = ontology.object_lexicon.get(entry.normalized, ())
        supports = tuple(
            scene.label
            for scene in ontology.scene_spaces.values()
            if entry.normalized in scene.support_objects
        )
        return MemoryRecord(
            memory_id=f"mem_store_object_{entry.normalized}",
            space_id=ontology.concept_space_id(entry.normalized),
            space_type="concept",
            label=entry.normalized,
            text=" ".join(filter(None, (entry.normalized, entry.label, *aliases))),
            supports=supports,
            aliases=aliases,
        )

    if entry.item_type == "attribute":
        aliases = ontology.attribute_lexicon.get(entry.normalized, ())
        supports = tuple(
            scene.label
            for scene in ontology.scene_spaces.values()
            if entry.normalized in scene.support_attributes
        )
        return MemoryRecord(
            memory_id=f"mem_store_attribute_{entry.normalized}",
            space_id=ontology.attribute_space_id(entry.normalized),
            space_type="attribute",
            label=entry.normalized,
            text=" ".join(filter(None, (entry.normalized, entry.label, *aliases))),
            supports=supports,
            aliases=aliases,
        )

    if entry.item_type == "relation":
        supports = tuple(
            scene.label
            for scene in ontology.scene_spaces.values()
            if entry.normalized in scene.support_relations
        )
        return MemoryRecord(
            memory_id=f"mem_store_relation_{entry.normalized}",
            space_id=ontology.relation_space_id(entry.normalized),
            space_type="relation",
            label=entry.normalized,
            text=" ".join(filter(None, (entry.normalized, entry.label))),
            supports=supports,
        )

    if entry.item_type == "event":
        supports = tuple(
            scene.label
            for scene in ontology.scene_spaces.values()
            if entry.normalized in scene.support_events
        )
        return MemoryRecord(
            memory_id=f"mem_store_event_{entry.normalized}",
            space_id=f"event_{entry.normalized}",
            space_type="event",
            label=entry.normalized,
            text=" ".join(filter(None, (entry.normalized, entry.label))),
            supports=supports,
        )

    if entry.item_type == "scene_hint":
        supports = ontology.get_scene_hint_targets(entry.normalized)
        label = supports[0] if supports else entry.normalized
        return MemoryRecord(
            memory_id=f"mem_store_scene_hint_{entry.normalized}",
            space_id=ontology.scene_space_id(label),
            space_type="scene",
            label=label,
            text=" ".join(filter(None, (entry.normalized, entry.label, *supports))),
            supports=supports or (label,),
        )

    return None
