from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Sequence

from .embedding_backends import EmbeddingEncoder
from .ingest import KnowledgeEntry
from .ontology import SBV01Ontology, load_default_ontology
from .llm_bridge import build_llm_context, retrieve_for_llm
from .output import build_output
from .parser import SBV01Parser
from .scorer import SBV01Scorer
from .state_update import apply_state_updates
from .v01_types import ActivationHit, AnalysisResult, ExtractedSignal, ParsedScene, ReasoningStep
from .vector_memory import VectorMemoryIndex


class SBV01Engine:
    def __init__(
        self,
        ontology: SBV01Ontology | None = None,
        stable_entries: Sequence[KnowledgeEntry] | None = None,
        embedding_encoder: EmbeddingEncoder | None = None,
        vector_top_k: int | None = None,
        vector_min_score: float = 0.20,
    ) -> None:
        self.ontology = ontology or load_default_ontology()
        self.parser = SBV01Parser(self.ontology)
        self.scorer = SBV01Scorer(self.ontology)
        self.vector_top_k = vector_top_k or self.ontology.limits["max_spaces_per_signal"]
        self.vector_min_score = vector_min_score
        self.vector_index = VectorMemoryIndex.from_ontology(
            self.ontology,
            stable_entries=stable_entries,
            encoder=embedding_encoder,
        )

    @classmethod
    def from_default_config(cls) -> "SBV01Engine":
        return cls(load_default_ontology())

    def analyze(self, text: str) -> Dict[str, object]:
        parsed = self.parser.parse(text)
        self._merge_entities(parsed)
        state_inference, derived_attributes, derived_relations = apply_state_updates(parsed)
        parsed.attributes = self.parser._dedupe_attributes([*parsed.attributes, *derived_attributes])
        parsed.relations = self.parser._dedupe_relations([*parsed.relations, *derived_relations])
        parsed.events = self.parser._dedupe_events(parsed.events)
        parsed.signals = self.parser.build_signals(
            parsed.objects,
            parsed.attributes,
            parsed.relations,
            parsed.events,
            parsed.scene_hints,
        )

        activation_hits = self._activate_signals(parsed.signals)
        hypotheses = self.scorer.score_scene_hypotheses(parsed, activation_hits, state_inference)
        best = hypotheses[0] if hypotheses else None
        reasoning_path = self._build_reasoning_path(best.label if best else "", activation_hits)

        result = AnalysisResult(
            input_text=parsed.input_text,
            objects=parsed.objects,
            attributes=parsed.attributes,
            relations=parsed.relations,
            events=parsed.events,
            scene_hypotheses=hypotheses,
            best_hypothesis=best,
            reasoning_path=reasoning_path,
            state_inference=state_inference,
        )
        return build_output(result)

    def retrieve_memories(self, text: str, analysis: Dict[str, object], top_k: int = 6) -> List[Dict[str, object]]:
        hits = retrieve_for_llm(self.vector_index, text, analysis, top_k=top_k)
        return [
            {
                "memory_id": hit.memory_id,
                "label": hit.label,
                "space_type": hit.space_type,
                "score": round(hit.score, 3),
                "supports": list(hit.supports),
                "text": hit.text,
            }
            for hit in hits
        ]

    def build_llm_payload(self, text: str, analysis: Dict[str, object], top_k: int = 6) -> Dict[str, object]:
        hits = retrieve_for_llm(self.vector_index, text, analysis, top_k=top_k)
        return build_llm_context(text, analysis, hits)

    def _merge_entities(self, parsed: ParsedScene) -> None:
        canonical_by_label = {}
        merged_objects = []
        object_id_map: Dict[str, str] = {}

        for item in sorted(parsed.objects, key=lambda value: (value.span[0], value.span[1], value.object_id)):
            canonical = canonical_by_label.get(item.normalized)
            if canonical is None:
                canonical_by_label[item.normalized] = item
                merged_objects.append(item)
                object_id_map[item.object_id] = item.object_id
                continue

            object_id_map[item.object_id] = canonical.object_id
            canonical.confidence = max(canonical.confidence, item.confidence)

        parsed.objects = merged_objects
        object_lookup = {item.object_id: item for item in parsed.objects}

        for item in parsed.attributes:
            if item.target_id is None:
                continue
            item.target_id = object_id_map.get(item.target_id, item.target_id)
            target = object_lookup.get(item.target_id)
            if target is not None:
                item.target_label = target.normalized

        for item in parsed.relations:
            item.source_id = object_id_map.get(item.source_id, item.source_id)
            item.target_id = object_id_map.get(item.target_id, item.target_id)
            source = object_lookup.get(item.source_id)
            target = object_lookup.get(item.target_id)
            if source is not None:
                item.source_label = source.normalized
            if target is not None:
                item.target_label = target.normalized

        for item in parsed.events:
            if item.actor_id is not None:
                item.actor_id = object_id_map.get(item.actor_id, item.actor_id)
                actor = object_lookup.get(item.actor_id)
                if actor is not None:
                    item.actor_label = actor.normalized
            if item.target_id is not None:
                item.target_id = object_id_map.get(item.target_id, item.target_id)
                target = object_lookup.get(item.target_id)
                if target is not None:
                    item.target_label = target.normalized
            if item.location_id is not None:
                item.location_id = object_id_map.get(item.location_id, item.location_id)
                location = object_lookup.get(item.location_id)
                if location is not None:
                    item.location_label = location.normalized
            if item.target_ids:
                remapped_ids = []
                for target_id in item.target_ids:
                    mapped = object_id_map.get(target_id, target_id)
                    if mapped not in remapped_ids:
                        remapped_ids.append(mapped)
                item.target_ids = tuple(remapped_ids)
                item.target_labels = tuple(
                    object_lookup[target_id].normalized
                    if target_id in object_lookup
                    else label
                    for target_id, label in zip(item.target_ids, item.target_labels or item.target_ids)
                )

        parsed.attributes = self.parser._dedupe_attributes(parsed.attributes)
        parsed.relations = self.parser._dedupe_relations(parsed.relations)
        parsed.events = self.parser._dedupe_events(parsed.events)

    def _activate_signals(self, signals: Sequence[ExtractedSignal]) -> List[ActivationHit]:
        hits: List[ActivationHit] = []
        transition_memory: Dict[str, float] = defaultdict(float)

        for signal in signals:
            candidates = self._candidate_spaces(signal)
            ranked: List[ActivationHit] = []
            for candidate in candidates:
                transition_prior = transition_memory.get(candidate["space_id"], 0.0)
                score = self.scorer.local_activation_score(
                    signal.signal_type,
                    signal.normalized,
                    candidate["space_type"],
                    candidate["label"],
                    aliases=candidate.get("aliases", ()),
                    context_compatibility=candidate.get("context_compatibility", 1.0),
                    transition_prior=transition_prior,
                    conflict_penalty=0.0,
                )
                if score < self.ontology.thresholds["activation"]:
                    continue
                ranked.append(
                    ActivationHit(
                        signal_id=signal.signal_id,
                        signal_label=signal.normalized,
                        signal_type=signal.signal_type,
                        space_id=candidate["space_id"],
                        space_type=candidate["space_type"],
                        space_label=candidate["label"],
                        score=score,
                        supports=tuple(candidate.get("supports", ())),
                    )
                )

            ranked.sort(key=lambda item: item.score, reverse=True)
            limited = ranked[: self.ontology.limits["max_spaces_per_signal"]]
            hits.extend(limited)
            for item in limited:
                for supported in item.supports[: self.ontology.limits["max_propagation_targets"]]:
                    transition_memory[self.ontology.scene_space_id(supported)] = max(
                        transition_memory[self.ontology.scene_space_id(supported)],
                        item.score * 0.5,
                    )
        return hits

    def _candidate_spaces(self, signal: ExtractedSignal) -> List[Dict[str, object]]:
        candidates: List[Dict[str, object]] = []

        if signal.signal_type == "object":
            normalized = self.ontology.normalize_object(signal.normalized) or signal.normalized
            candidates.extend([
                {
                    "space_id": self.ontology.concept_space_id(normalized),
                    "space_type": "concept",
                    "label": normalized,
                    "aliases": self.ontology.object_lexicon.get(normalized, ()),
                    "supports": self._scene_supports_for_object(normalized),
                    "context_compatibility": 1.0,
                }
            ])

        elif signal.signal_type == "attribute":
            normalized = self.ontology.normalize_attribute(signal.normalized) or signal.normalized
            candidates.extend([
                {
                    "space_id": self.ontology.attribute_space_id(normalized),
                    "space_type": "attribute",
                    "label": normalized,
                    "aliases": self.ontology.attribute_lexicon.get(normalized, ()),
                    "supports": self._scene_supports_for_attribute(normalized),
                    "context_compatibility": 1.0,
                }
            ])

        elif signal.signal_type == "relation":
            candidates.extend([
                {
                    "space_id": self.ontology.relation_space_id(signal.normalized),
                    "space_type": "relation",
                    "label": signal.normalized,
                    "supports": self._scene_supports_for_relation(signal.normalized),
                    "context_compatibility": 1.0,
                }
            ])

        elif signal.signal_type == "scene_hint":
            candidates.extend([
                {
                    "space_id": self.ontology.scene_space_id(scene),
                    "space_type": "scene",
                    "label": scene,
                    "supports": (scene,),
                    "context_compatibility": 0.8,
                }
                for scene in self.ontology.get_scene_hint_targets(signal.normalized)
            ])

        elif signal.signal_type == "event":
            candidates.extend([
                {
                    "space_id": self.ontology.scene_space_id(scene),
                    "space_type": "scene",
                    "label": scene,
                    "supports": (scene,),
                    "context_compatibility": 0.75,
                }
                for scene in self._scene_supports_for_event(signal.normalized)
            ])

        candidates.extend(self._vector_candidates(signal))
        return self._dedupe_candidates(candidates)

    def _vector_candidates(self, signal: ExtractedSignal) -> List[Dict[str, object]]:
        allowed_map = {
            "object": ("concept", "scene"),
            "attribute": ("attribute", "scene"),
            "relation": ("relation", "scene"),
            "event": ("event", "scene"),
            "scene_hint": ("scene",),
        }
        hits = self.vector_index.search(
            signal.normalized,
            top_k=self.vector_top_k,
            min_score=self.vector_min_score,
            allowed_types=allowed_map.get(signal.signal_type, ()),
        )
        candidates: List[Dict[str, object]] = []
        for hit in hits:
            mapped_type = "scene" if hit.space_type == "scene" else hit.space_type
            if mapped_type == "event":
                mapped_type = "scene"
            supports = hit.supports if hit.supports else ((hit.label,) if mapped_type == "scene" else ())
            candidates.append(
                {
                    "space_id": hit.space_id if mapped_type != "scene" else self._scene_space_id_for_hit(hit, supports),
                    "space_type": mapped_type,
                    "label": hit.label,
                    "aliases": hit.aliases,
                    "supports": supports,
                    "context_compatibility": min(0.95, 0.45 + hit.score * 0.5),
                }
            )
        return candidates

    def _scene_space_id_for_hit(self, hit: object, supports: Sequence[str]) -> str:
        label = supports[0] if supports else getattr(hit, "label")
        return self.ontology.scene_space_id(label)

    @staticmethod
    def _dedupe_candidates(candidates: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
        deduped: Dict[tuple[str, str], Dict[str, object]] = {}
        for item in candidates:
            key = (str(item["space_type"]), str(item["space_id"]))
            current = deduped.get(key)
            if current is None or float(item.get("context_compatibility", 0.0)) > float(
                current.get("context_compatibility", 0.0)
            ):
                deduped[key] = item
        return list(deduped.values())

    def _scene_supports_for_object(self, label: str) -> tuple[str, ...]:
        return tuple(
            scene.label
            for scene in self.ontology.scene_spaces.values()
            if label in scene.support_objects
        )

    def _scene_supports_for_attribute(self, label: str) -> tuple[str, ...]:
        return tuple(
            scene.label
            for scene in self.ontology.scene_spaces.values()
            if label in scene.support_attributes
        )

    def _scene_supports_for_relation(self, label: str) -> tuple[str, ...]:
        return tuple(
            scene.label
            for scene in self.ontology.scene_spaces.values()
            if label in scene.support_relations
        )

    def _scene_supports_for_event(self, label: str) -> tuple[str, ...]:
        return tuple(
            scene.label
            for scene in self.ontology.scene_spaces.values()
            if label in scene.support_events
        )

    def _build_reasoning_path(
        self,
        best_label: str,
        activation_hits: Sequence[ActivationHit],
    ) -> List[ReasoningStep]:
        if not best_label:
            return []

        relevant = [hit for hit in activation_hits if best_label in hit.supports]
        relevant.sort(key=lambda item: item.score, reverse=True)

        steps: List[ReasoningStep] = []
        seen = set()
        step_index = 1
        for hit in relevant:
            if hit.signal_label in seen:
                continue
            seen.add(hit.signal_label)
            steps.append(
                ReasoningStep(
                    step=step_index,
                    signal=hit.signal_label,
                    activated_spaces=(hit.space_id, self.ontology.scene_space_id(best_label)),
                    contribution=f"{hit.signal_label} 支持 {best_label}",
                )
            )
            step_index += 1
            if step_index > 6:
                break
        return steps
