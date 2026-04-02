from __future__ import annotations

import re
from typing import Dict, Iterable, List, Sequence, Tuple

from .ontology import SBV01Ontology
from .v01_types import (
    AttributeEdge,
    EventNode,
    ExtractedSignal,
    ObjectNode,
    ParsedScene,
    RelationEdge,
    SceneHint,
    Segment,
)


STANDARD_REPLACEMENTS = {
    "边上": "旁边",
    "边儿上": "旁边",
    "附近有": "旁边有",
    "摆放杂乱": "摆放散乱",
    "没有整齐堆放": "不整齐堆放",
    "洒出来": "洒在",
    "洒落在": "洒在",
    "碰倒了": "被碰倒",
}


class SBV01Parser:
    def __init__(self, ontology: SBV01Ontology) -> None:
        self.ontology = ontology
        self._signal_counter = 0
        self._object_counter = 0
        self._attribute_counter = 0
        self._relation_counter = 0
        self._event_counter = 0
        self._hint_counter = 0

    def parse(self, text: str) -> ParsedScene:
        normalized_text = self._normalize_text(text)
        segments = self._split_segments(normalized_text)
        objects = self._extract_objects(segments)
        attributes = self._extract_attributes(segments, objects)
        relations = self._extract_relations(segments, objects)
        events = self._extract_events(segments, objects)
        scene_hints = self._extract_scene_hints(segments)
        signals = self.build_signals(objects, attributes, relations, events, scene_hints)

        return ParsedScene(
            input_text=text,
            normalized_text=normalized_text,
            segments=segments,
            signals=signals,
            objects=objects,
            attributes=attributes,
            relations=relations,
            events=events,
            scene_hints=scene_hints,
        )

    def build_signals(
        self,
        objects: Sequence[ObjectNode],
        attributes: Sequence[AttributeEdge],
        relations: Sequence[RelationEdge],
        events: Sequence[EventNode],
        scene_hints: Sequence[SceneHint],
    ) -> List[ExtractedSignal]:
        signals: List[ExtractedSignal] = []
        for item in objects:
            signals.append(
                ExtractedSignal(
                    signal_id=self._next_id("sig"),
                    signal_type="object",
                    surface=item.label,
                    normalized=item.normalized,
                    category=item.category,
                    segment_id=item.segment_id,
                    span=item.span,
                    confidence=item.confidence,
                    anchors=(item.object_id,),
                    provenance=("parser",),
                )
            )
        for item in attributes:
            signals.append(
                ExtractedSignal(
                    signal_id=self._next_id("sig"),
                    signal_type="attribute",
                    surface=item.label,
                    normalized=item.normalized,
                    category="属性",
                    segment_id=item.segment_id,
                    span=item.span,
                    confidence=item.confidence,
                    anchors=(item.target_id,) if item.target_id is not None else (),
                    provenance=("parser",),
                )
            )
        for item in relations:
            signals.append(
                ExtractedSignal(
                    signal_id=self._next_id("sig"),
                    signal_type="relation",
                    surface=item.label,
                    normalized=item.normalized,
                    category="关系",
                    segment_id=item.segment_id,
                    span=item.span,
                    confidence=item.confidence,
                    anchors=(item.source_id, item.target_id),
                    provenance=("parser",),
                )
            )
        for item in events:
            anchors = tuple(
                anchor
                for anchor in (item.actor_id, item.target_id, item.location_id)
                if anchor is not None
            )
            signals.append(
                ExtractedSignal(
                    signal_id=self._next_id("sig"),
                    signal_type="event",
                    surface=item.event_type,
                    normalized=item.event_type,
                    category="事件",
                    segment_id=item.segment_id,
                    span=item.span,
                    confidence=item.confidence,
                    anchors=anchors,
                    provenance=("parser",),
                )
            )
        for item in scene_hints:
            signals.append(
                ExtractedSignal(
                    signal_id=self._next_id("sig"),
                    signal_type="scene_hint",
                    surface=item.label,
                    normalized=item.label,
                    category="场景提示",
                    segment_id=item.segment_id,
                    span=item.span,
                    confidence=item.confidence,
                    provenance=("parser",),
                )
            )
        return signals

    def _normalize_text(self, text: str) -> str:
        normalized = text.strip()
        for source, target in STANDARD_REPLACEMENTS.items():
            normalized = normalized.replace(source, target)
        return normalized

    def _split_segments(self, text: str) -> List[Segment]:
        segments: List[Segment] = []
        start = 0
        index = 1
        for match in re.finditer(r"[，。！？；]", text):
            chunk = text[start:match.start()].strip()
            if chunk:
                segments.append(
                    Segment(
                        segment_id=f"seg_{index}",
                        text=chunk,
                        start=start,
                        end=match.start(),
                    )
                )
                index += 1
            start = match.end()
        tail = text[start:].strip()
        if tail:
            segments.append(
                Segment(
                    segment_id=f"seg_{index}",
                    text=tail,
                    start=start,
                    end=len(text),
                )
            )
        return segments

    def _extract_objects(self, segments: Sequence[Segment]) -> List[ObjectNode]:
        objects: List[ObjectNode] = []
        for segment in segments:
            matches = self._find_known_mentions(segment.text, self.ontology.object_terms, kind="object")
            for surface, label, span in matches:
                objects.append(
                    ObjectNode(
                        object_id=self._next_id("obj"),
                        label=surface,
                        normalized=label,
                        category=self.ontology.object_categories.get(label, "一般对象"),
                        segment_id=segment.segment_id,
                        span=(segment.start + span[0], segment.start + span[1]),
                        confidence=0.95 if surface == label else 0.85,
                    )
                )
        return self._dedupe_objects(objects)

    def _extract_attributes(
        self,
        segments: Sequence[Segment],
        objects: Sequence[ObjectNode],
    ) -> List[AttributeEdge]:
        attributes: List[AttributeEdge] = []
        objects_by_segment = self._group_by_segment(objects)
        scene_level_attributes = {"散乱", "不整齐", "整齐"}

        for segment in segments:
            matches = self._find_known_mentions(segment.text, self.ontology.attribute_terms, kind="attribute")
            local_objects = objects_by_segment.get(segment.segment_id, [])
            for surface, label, span in matches:
                target = self._resolve_attribute_target(
                    (segment.start + span[0], segment.start + span[1]),
                    local_objects,
                    label,
                    scene_level_attributes,
                )
                attributes.append(
                    AttributeEdge(
                        attribute_id=self._next_id("attr"),
                        label=surface,
                        normalized=label,
                        target_id=target.object_id if target is not None else None,
                        target_label=target.normalized if target is not None else "scene",
                        confidence=0.92 if surface == label else 0.82,
                        segment_id=segment.segment_id,
                        span=(segment.start + span[0], segment.start + span[1]),
                    )
                )
        return self._dedupe_attributes(attributes)

    def _extract_relations(
        self,
        segments: Sequence[Segment],
        objects: Sequence[ObjectNode],
    ) -> List[RelationEdge]:
        relations: List[RelationEdge] = []
        objects_by_segment = self._group_by_segment(objects)

        for segment in segments:
            local_objects = sorted(objects_by_segment.get(segment.segment_id, []), key=lambda item: item.span[0])
            if len(local_objects) >= 2:
                surface_pattern = self._build_surface_pattern(local_objects)
                if surface_pattern:
                    gap = r"(?:[^，。！？；]{0,6}?)"
                    patterns = [
                        (rf"(?P<x>{surface_pattern})左边(?:有|放着|是)?{gap}(?P<y>{surface_pattern})", "在……左侧", "在……旁边"),
                        (rf"(?P<y>{surface_pattern})在(?P<x>{surface_pattern})左边", "在……左侧", "在……旁边"),
                        (rf"(?P<x>{surface_pattern})右边(?:有|放着|是)?{gap}(?P<y>{surface_pattern})", "在……右侧", "在……旁边"),
                        (rf"(?P<y>{surface_pattern})在(?P<x>{surface_pattern})右边", "在……右侧", "在……旁边"),
                        (rf"(?P<x>{surface_pattern})(?:的)?旁边(?:有|停着|放着|是)?{gap}(?P<y>{surface_pattern})", "在……旁边", "在……旁边"),
                        (rf"(?P<y>{surface_pattern})在(?P<x>{surface_pattern})(?:的)?旁边", "在……旁边", "在……旁边"),
                        (rf"(?P<x>{surface_pattern})(?:里|里面)有{gap}(?P<y>{surface_pattern})", "在……里面", "在……里面"),
                        (rf"(?P<y>{surface_pattern})在(?P<x>{surface_pattern})(?:里|里面)", "在……里面", "在……里面"),
                        (rf"(?P<x>{surface_pattern})上(?:有|放着|堆着)?{gap}(?P<y>{surface_pattern})", "在……上", "在……上"),
                        (rf"(?P<y>{surface_pattern})在(?P<x>{surface_pattern})上", "在……上", "在……上"),
                        (rf"(?P<y>{surface_pattern})靠近(?P<x>{surface_pattern})", "靠近", "靠近"),
                        (rf"(?P<x>{surface_pattern})后面(?:有|放着)?{gap}(?P<y>{surface_pattern})", "在……后方", "在……旁边"),
                        (rf"(?P<y>{surface_pattern})在(?P<x>{surface_pattern})后面", "在……后方", "在……旁边"),
                    ]

                    for pattern, label, normalized in patterns:
                        for match in re.finditer(pattern, segment.text):
                            left = self._find_local_object(local_objects, match.group("x"), segment.start + match.start("x"))
                            right = self._find_local_object(local_objects, match.group("y"), segment.start + match.start("y"))
                            if left is None or right is None or left.object_id == right.object_id:
                                continue
                            relations.append(
                                RelationEdge(
                                    relation_id=self._next_id("rel"),
                                    label=label,
                                    normalized=normalized,
                                    source_id=right.object_id,
                                    source_label=right.normalized,
                                    target_id=left.object_id,
                                    target_label=left.normalized,
                                    confidence=0.88,
                                    segment_id=segment.segment_id,
                                    span=(segment.start + match.start(), segment.start + match.end()),
                                )
                            )

            contextual = self._extract_contextual_relation(segment, local_objects, segments, objects_by_segment)
            if contextual is not None:
                relations.append(contextual)
        return self._dedupe_relations(relations)

    def _extract_events(
        self,
        segments: Sequence[Segment],
        objects: Sequence[ObjectNode],
    ) -> List[EventNode]:
        events: List[EventNode] = []
        objects_by_segment = self._group_by_segment(objects)

        for segment in segments:
            local_objects = sorted(objects_by_segment.get(segment.segment_id, []), key=lambda item: item.span[0])
            surface_pattern = self._build_surface_pattern(local_objects)

            if surface_pattern:
                pair_patterns = [
                    (rf"(?P<x>{surface_pattern})碰到了(?P<y>{surface_pattern})", "撞击"),
                    (rf"(?P<x>{surface_pattern})碰到(?P<y>{surface_pattern})", "撞击"),
                    (rf"(?P<x>{surface_pattern})撞到了(?P<y>{surface_pattern})", "撞击"),
                    (rf"(?P<y>{surface_pattern})洒在(?P<x>{surface_pattern})(?:上|里|面上)?", "泄漏"),
                ]

                for pattern, event_type in pair_patterns:
                    for match in re.finditer(pattern, segment.text):
                        actor = self._find_local_object(local_objects, match.group("x"), segment.start + match.start("x"))
                        target = self._find_local_object(local_objects, match.group("y"), segment.start + match.start("y"))
                        if actor is None or target is None:
                            continue
                        if event_type == "泄漏":
                            actor, target = target, actor
                        events.append(
                            EventNode(
                                event_id=self._next_id("evt"),
                                event_type=event_type,
                                actor_id=actor.object_id,
                                actor_label=actor.normalized,
                                target_id=target.object_id,
                                target_label=target.normalized,
                                location_id=target.object_id if event_type == "泄漏" else None,
                                location_label=target.normalized if event_type == "泄漏" else None,
                                confidence=0.90,
                                segment_id=segment.segment_id,
                                span=(segment.start + match.start(), segment.start + match.end()),
                                provenance=("pattern",),
                            )
                        )

                single_patterns = [
                    (rf"(?P<x>{surface_pattern})翻倒了", "翻倒"),
                    (rf"(?P<x>{surface_pattern})被碰倒", "翻倒"),
                    (rf"(?P<x>{surface_pattern})倒了", "翻倒"),
                ]

                for pattern, event_type in single_patterns:
                    for match in re.finditer(pattern, segment.text):
                        target = self._find_local_object(local_objects, match.group("x"), segment.start + match.start("x"))
                        if target is None:
                            continue
                        events.append(
                            EventNode(
                                event_id=self._next_id("evt"),
                                event_type=event_type,
                                actor_id=None,
                                actor_label=None,
                                target_id=target.object_id,
                                target_label=target.normalized,
                                location_id=None,
                                location_label=None,
                                confidence=0.90,
                                segment_id=segment.segment_id,
                                span=(segment.start + match.start(), segment.start + match.end()),
                                provenance=("pattern",),
                            )
                        )

            if "堆着" in segment.text or "堆放" in segment.text:
                marker = segment.text.find("堆着")
                if marker < 0:
                    marker = segment.text.find("堆放")
                before = [item for item in local_objects if item.span[0] - segment.start < marker]
                after = [item for item in local_objects if item.span[0] - segment.start > marker]
                location = before[-1] if before else None
                if after:
                    events.append(
                        EventNode(
                            event_id=self._next_id("evt"),
                            event_type="堆放",
                            actor_id=None,
                            actor_label=None,
                            target_id=after[0].object_id,
                            target_label=after[0].normalized,
                            location_id=location.object_id if location is not None else None,
                            location_label=location.normalized if location is not None else None,
                            target_ids=tuple(item.object_id for item in after),
                            target_labels=tuple(item.normalized for item in after),
                            confidence=0.82,
                            segment_id=segment.segment_id,
                            span=(segment.start + marker, segment.start + len(segment.text)),
                            provenance=("pattern",),
                        )
                    )

            if "散落" in segment.text:
                marker = segment.text.find("散落")
                before = [item for item in local_objects if item.span[0] - segment.start < marker]
                after = [item for item in local_objects if item.span[0] - segment.start > marker]
                target = before[-1] if before else (after[0] if after else None)
                location = after[0] if after else None
                if target is not None:
                    events.append(
                        EventNode(
                            event_id=self._next_id("evt"),
                            event_type="散落",
                            actor_id=target.object_id,
                            actor_label=target.normalized,
                            target_id=target.object_id,
                            target_label=target.normalized,
                            location_id=location.object_id if location is not None else None,
                            location_label=location.normalized if location is not None else None,
                            confidence=0.84,
                            segment_id=segment.segment_id,
                            span=(segment.start + marker, segment.start + len(segment.text)),
                            provenance=("pattern",),
                        )
                    )

        return self._dedupe_events(events)

    def _extract_scene_hints(self, segments: Sequence[Segment]) -> List[SceneHint]:
        hints: List[SceneHint] = []
        hint_terms = sorted(self.ontology.scene_hint_terms, key=len, reverse=True)
        for segment in segments:
            for surface, _, span in self._find_known_mentions(segment.text, hint_terms, is_direct_label=True):
                hints.append(
                    SceneHint(
                        hint_id=self._next_id("hint"),
                        label=surface,
                        confidence=0.72,
                        segment_id=segment.segment_id,
                        span=(segment.start + span[0], segment.start + span[1]),
                    )
                )
        return self._dedupe_hints(hints)

    def _group_by_segment(self, items: Iterable[object]) -> Dict[str, List[object]]:
        grouped: Dict[str, List[object]] = {}
        for item in items:
            grouped.setdefault(item.segment_id, []).append(item)
        return grouped

    def _resolve_attribute_target(
        self,
        absolute_span: Tuple[int, int],
        local_objects: Sequence[ObjectNode],
        label: str,
        scene_level_attributes: set[str],
    ) -> ObjectNode | None:
        if label in scene_level_attributes and len(local_objects) > 1:
            return None

        best: ObjectNode | None = None
        best_distance = None
        for item in local_objects:
            distance = min(
                abs(item.span[0] - absolute_span[1]),
                abs(absolute_span[0] - item.span[1]),
            )
            if best_distance is None or distance < best_distance:
                best = item
                best_distance = distance
        if best_distance is None:
            return None
        return best if best_distance <= 8 else None

    def _find_known_mentions(
        self,
        text: str,
        terms: Sequence[str],
        kind: str = "object",
        is_direct_label: bool = False,
    ) -> List[Tuple[str, str, Tuple[int, int]]]:
        matches: List[Tuple[str, str, Tuple[int, int]]] = []
        candidates: List[Tuple[int, int, str, str]] = []
        for term in terms:
            for match in re.finditer(re.escape(term), text):
                if is_direct_label:
                    label = term
                elif kind == "object":
                    label = self.ontology.normalize_object(term) or term
                else:
                    label = self.ontology.normalize_attribute(term) or term
                candidates.append((match.start(), match.end(), match.group(), label))

        candidates.sort(key=lambda item: (item[0], -(item[1] - item[0])))
        occupied: List[Tuple[int, int]] = []
        for start, end, surface, label in candidates:
            span = (start, end)
            if any(not (span[1] <= current[0] or span[0] >= current[1]) for current in occupied):
                continue
            occupied.append(span)
            matches.append((surface, label, span))

        matches.sort(key=lambda item: item[2][0])
        return matches

    @staticmethod
    def _build_surface_pattern(local_objects: Sequence[ObjectNode]) -> str:
        surfaces = sorted({item.label for item in local_objects}, key=len, reverse=True)
        if not surfaces:
            return ""
        return "|".join(re.escape(surface) for surface in surfaces)

    @staticmethod
    def _find_local_object(
        local_objects: Sequence[ObjectNode],
        surface: str,
        absolute_start: int,
    ) -> ObjectNode | None:
        candidates = [
            item
            for item in local_objects
            if item.label == surface or item.normalized == surface
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda item: abs(item.span[0] - absolute_start))

    def _extract_contextual_relation(
        self,
        segment: Segment,
        local_objects: Sequence[ObjectNode],
        segments: Sequence[Segment],
        objects_by_segment: Dict[str, List[ObjectNode]],
    ) -> RelationEdge | None:
        if len(local_objects) != 1:
            return None

        anchor = self._find_context_anchor(segment.segment_id, segments, objects_by_segment)
        if anchor is None or anchor.object_id == local_objects[0].object_id:
            return None

        text = segment.text
        current = local_objects[0]
        if text.startswith("旁边"):
            label = "在……旁边"
            normalized = "在……旁边"
            source = current
            target = anchor
        elif text.startswith("左边"):
            label = "在……左侧"
            normalized = "在……旁边"
            source = current
            target = anchor
        elif text.startswith("右边"):
            label = "在……右侧"
            normalized = "在……旁边"
            source = current
            target = anchor
        elif text.startswith("后面"):
            label = "在……后方"
            normalized = "在……旁边"
            source = current
            target = anchor
        elif text.startswith("靠近"):
            label = "靠近"
            normalized = "靠近"
            source = anchor
            target = current
        else:
            return None

        return RelationEdge(
            relation_id=self._next_id("rel"),
            label=label,
            normalized=normalized,
            source_id=source.object_id,
            source_label=source.normalized,
            target_id=target.object_id,
            target_label=target.normalized,
            confidence=0.74,
            segment_id=segment.segment_id,
            span=(segment.start, segment.end),
        )

    def _find_context_anchor(
        self,
        segment_id: str,
        segments: Sequence[Segment],
        objects_by_segment: Dict[str, List[ObjectNode]],
    ) -> ObjectNode | None:
        segment_index = next(
            (index for index, item in enumerate(segments) if item.segment_id == segment_id),
            -1,
        )
        if segment_index <= 0:
            return None

        preferred_categories = {"环境类", "附属物类", "容器类"}

        def select(candidates: List[ObjectNode]) -> ObjectNode:
            preferred = [item for item in candidates if item.category in preferred_categories]
            return preferred[-1] if preferred else candidates[-1]

        for index in range(segment_index - 1, -1, -1):
            candidates = sorted(objects_by_segment.get(segments[index].segment_id, []), key=lambda item: item.span[0])
            if len(candidates) >= 2:
                return select(candidates)

        for index in range(segment_index - 1, -1, -1):
            candidates = sorted(objects_by_segment.get(segments[index].segment_id, []), key=lambda item: item.span[0])
            if candidates:
                return select(candidates)
        return None

    def _dedupe_objects(self, items: Sequence[ObjectNode]) -> List[ObjectNode]:
        seen = set()
        deduped: List[ObjectNode] = []
        for item in items:
            key = (item.segment_id, item.normalized, item.span)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
        return deduped

    def _dedupe_attributes(self, items: Sequence[AttributeEdge]) -> List[AttributeEdge]:
        seen = set()
        deduped: List[AttributeEdge] = []
        for item in items:
            key = (item.segment_id, item.normalized, item.target_id, item.span)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
        return deduped

    def _dedupe_relations(self, items: Sequence[RelationEdge]) -> List[RelationEdge]:
        seen = set()
        deduped: List[RelationEdge] = []
        for item in items:
            key = (item.segment_id, item.normalized, item.source_id, item.target_id)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
        return deduped

    def _dedupe_events(self, items: Sequence[EventNode]) -> List[EventNode]:
        seen = set()
        deduped: List[EventNode] = []
        for item in items:
            key = (item.segment_id, item.event_type, item.actor_id, item.target_id, item.location_id, item.target_ids)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
        return deduped

    def _dedupe_hints(self, items: Sequence[SceneHint]) -> List[SceneHint]:
        seen = set()
        deduped: List[SceneHint] = []
        for item in items:
            key = (item.segment_id, item.label, item.span)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
        return deduped

    def _next_id(self, prefix: str) -> str:
        if prefix == "sig":
            self._signal_counter += 1
            return f"sig_{self._signal_counter:03d}"
        if prefix == "obj":
            self._object_counter += 1
            return f"obj_{self._object_counter:03d}"
        if prefix == "attr":
            self._attribute_counter += 1
            return f"attr_{self._attribute_counter:03d}"
        if prefix == "rel":
            self._relation_counter += 1
            return f"rel_{self._relation_counter:03d}"
        if prefix == "evt":
            self._event_counter += 1
            return f"evt_{self._event_counter:03d}"
        self._hint_counter += 1
        return f"hint_{self._hint_counter:03d}"
