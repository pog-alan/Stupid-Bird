from __future__ import annotations

from math import exp
from typing import Dict, Iterable, List, Sequence, Tuple

from .ontology import SBV01Ontology, SceneSpace
from .v01_types import ActivationHit, ParsedScene, SceneHypothesis, StateInference


CONFLICTING_ATTRIBUTES = {
    "稳定": {"翻倒"},
    "翻倒": {"稳定"},
    "整齐": {"散乱", "不整齐"},
    "散乱": {"整齐"},
    "不整齐": {"整齐"},
    "干净": {"污染"},
    "污染": {"干净"},
    "完整": {"破碎"},
    "破碎": {"完整"},
}


class SBV01Scorer:
    def __init__(self, ontology: SBV01Ontology) -> None:
        self.ontology = ontology

    def local_activation_score(
        self,
        signal_type: str,
        signal_value: str,
        candidate_type: str,
        candidate_label: str,
        aliases: Iterable[str] = (),
        context_compatibility: float = 1.0,
        transition_prior: float = 0.0,
        conflict_penalty: float = 0.0,
    ) -> float:
        type_match = self._type_match(signal_type, candidate_type)
        content_similarity = self._content_similarity(signal_value, candidate_label, aliases)
        score = (
            self.ontology.local_activation_weights["type_match"] * type_match
            + self.ontology.local_activation_weights["content_similarity"] * content_similarity
            + self.ontology.local_activation_weights["context_compatibility"] * context_compatibility
            + self.ontology.local_activation_weights["transition_prior"] * min(transition_prior, 0.3)
            - conflict_penalty
        )
        return max(0.0, min(1.0, score))

    def score_scene_hypotheses(
        self,
        parsed: ParsedScene,
        activation_hits: Sequence[ActivationHit],
        state_inference: Sequence[StateInference],
    ) -> List[SceneHypothesis]:
        scene_support = self._collect_scene_support(activation_hits, parsed)
        hypotheses: List[SceneHypothesis] = []

        for label, scene in self.ontology.scene_spaces.items():
            support_entries = scene_support.get(label, [])
            matched_objects = tuple(sorted({entry[0] for entry in support_entries if entry[1] == "object"}))
            matched_attributes = tuple(sorted({entry[0] for entry in support_entries if entry[1] == "attribute"}))
            matched_relations = tuple(sorted({entry[0] for entry in support_entries if entry[1] == "relation"}))
            matched_events = tuple(sorted({entry[0] for entry in support_entries if entry[1] == "event"}))

            object_support = self._object_support(parsed, scene)
            attribute_consistency = self._attribute_consistency(parsed, scene)
            relation_consistency = self._relation_consistency(parsed, scene)
            scene_compatibility = self._scene_compatibility(parsed, scene, support_entries)
            event_plausibility = self._event_plausibility(parsed, scene, state_inference)
            conflict_penalty = self._conflict_penalty(parsed, state_inference, scene)
            mutual_exclusion_penalty = 0.0
            complexity_penalty = self._complexity_penalty(support_entries)

            weights = self.ontology.global_hypothesis_weights
            score = (
                weights["object_support"] * object_support
                + weights["attribute_consistency"] * attribute_consistency
                + weights["relation_consistency"] * relation_consistency
                + weights["scene_compatibility"] * scene_compatibility
                + weights["event_plausibility"] * event_plausibility
                - weights["conflict_penalty"] * conflict_penalty
                - weights["mutual_exclusion_penalty"] * mutual_exclusion_penalty
                - weights["complexity_penalty"] * complexity_penalty
            )

            hypotheses.append(
                SceneHypothesis(
                    label=label,
                    score=max(0.0, min(1.0, score)),
                    object_support=object_support,
                    attribute_consistency=attribute_consistency,
                    relation_consistency=relation_consistency,
                    scene_compatibility=scene_compatibility,
                    event_plausibility=event_plausibility,
                    conflict_penalty=conflict_penalty,
                    mutual_exclusion_penalty=mutual_exclusion_penalty,
                    complexity_penalty=complexity_penalty,
                    matched_objects=matched_objects,
                    matched_attributes=matched_attributes,
                    matched_relations=matched_relations,
                    matched_events=matched_events,
                )
            )

        hypotheses.sort(key=lambda item: item.score, reverse=True)
        return hypotheses[: self.ontology.limits["output_hypotheses"]]

    def _type_match(self, signal_type: str, candidate_type: str) -> float:
        exact_map = {
            "object": "concept",
            "attribute": "attribute",
            "relation": "relation",
            "scene_hint": "scene",
        }
        if exact_map.get(signal_type) == candidate_type:
            return 1.0
        if signal_type == "event" and candidate_type == "scene":
            return 0.6
        return 0.0

    def _content_similarity(self, signal_value: str, candidate_label: str, aliases: Iterable[str]) -> float:
        if signal_value == candidate_label:
            return 1.0
        if signal_value in aliases:
            return 0.9
        if signal_value in candidate_label or candidate_label in signal_value:
            return 0.7
        return 0.0

    def _collect_scene_support(
        self,
        activation_hits: Sequence[ActivationHit],
        parsed: ParsedScene,
    ) -> Dict[str, List[Tuple[str, str, float]]]:
        support: Dict[str, List[Tuple[str, str, float]]] = {}
        for hit in activation_hits:
            for scene_label in hit.supports:
                support.setdefault(scene_label, []).append((hit.signal_label, hit.signal_type, hit.score))

        for hint in parsed.scene_hints:
            for scene_label in self.ontology.get_scene_hint_targets(hint.label):
                support.setdefault(scene_label, []).append((hint.label, "scene_hint", hint.confidence * 0.6))

        for event in parsed.events:
            for scene_label, scene in self.ontology.scene_spaces.items():
                if event.event_type in scene.support_events:
                    support.setdefault(scene_label, []).append((event.event_type, "event", event.confidence * 0.8))
        return support

    def _object_support(self, parsed: ParsedScene, scene: SceneSpace) -> float:
        if not parsed.objects:
            return 0.0
        matched = sum(1 for item in parsed.objects if item.normalized in scene.support_objects)
        return matched / max(len(parsed.objects), 1)

    def _attribute_consistency(self, parsed: ParsedScene, scene: SceneSpace) -> float:
        if not parsed.attributes:
            return 0.5
        matched = sum(1 for item in parsed.attributes if item.normalized in scene.support_attributes)
        return matched / max(len(parsed.attributes), 1)

    def _relation_consistency(self, parsed: ParsedScene, scene: SceneSpace) -> float:
        if not parsed.relations:
            return 0.5
        matched = sum(1 for item in parsed.relations if item.normalized in scene.support_relations)
        return matched / max(len(parsed.relations), 1)

    def _scene_compatibility(
        self,
        parsed: ParsedScene,
        scene: SceneSpace,
        support_entries: Sequence[Tuple[str, str, float]],
    ) -> float:
        support_total = sum(item[2] for item in support_entries)
        support_score = 1.0 - exp(-support_total) if support_total > 0 else 0.0
        reject_penalty = 0.0
        for reject in scene.reject_signals:
            if reject in parsed.normalized_text:
                reject_penalty += 0.15
        compatibility = scene.base_score * 0.6 + support_score * 0.6 - reject_penalty
        return max(0.0, min(1.0, compatibility))

    def _event_plausibility(
        self,
        parsed: ParsedScene,
        scene: SceneSpace,
        state_inference: Sequence[StateInference],
    ) -> float:
        if not parsed.events:
            return 0.6 if not scene.support_events else 0.45
        matched = sum(1 for event in parsed.events if event.event_type in scene.support_events)
        state_bonus = 0.0
        for state in state_inference:
            if state.state in scene.support_attributes or state.state in scene.support_events:
                state_bonus += 0.15
        return max(0.0, min(1.0, matched / max(len(parsed.events), 1) + state_bonus))

    def _conflict_penalty(
        self,
        parsed: ParsedScene,
        state_inference: Sequence[StateInference],
        scene: SceneSpace,
    ) -> float:
        penalty = 0.0
        attribute_map: Dict[str, set[str]] = {}
        for item in parsed.attributes:
            if item.target_id is None:
                continue
            attribute_map.setdefault(item.target_id, set()).add(item.normalized)
        for state in state_inference:
            attribute_map.setdefault(state.target_id, set()).add(state.state)

        for values in attribute_map.values():
            for value in values:
                conflicting = CONFLICTING_ATTRIBUTES.get(value, set())
                if conflicting & values:
                    penalty += 0.3

        for reject in scene.reject_signals:
            if reject in parsed.normalized_text:
                penalty += 0.1
        return max(0.0, min(1.0, penalty))

    def _complexity_penalty(self, support_entries: Sequence[Tuple[str, str, float]]) -> float:
        if not support_entries:
            return 0.2
        hint_only = all(item[1] == "scene_hint" for item in support_entries)
        if hint_only:
            return 0.15
        if len(support_entries) <= 2:
            return 0.05
        return 0.0
