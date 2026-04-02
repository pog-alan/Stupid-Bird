from __future__ import annotations

from typing import List

from .v01_types import AttributeEdge, EventNode, ParsedScene, RelationEdge, StateInference


def apply_state_updates(parsed: ParsedScene) -> tuple[List[StateInference], List[AttributeEdge], List[RelationEdge]]:
    inferred_states: List[StateInference] = []
    derived_attributes: List[AttributeEdge] = []
    derived_relations: List[RelationEdge] = []

    attr_index = {(item.normalized, item.target_id) for item in parsed.attributes}
    relation_index = {(item.normalized, item.source_id, item.target_id) for item in parsed.relations}

    for event in parsed.events:
        if event.event_type == "撞击" and event.target_id is not None:
            inferred_states.append(
                StateInference(
                    target_id=event.target_id,
                    target_label=event.target_label or "",
                    state="不稳定风险升高",
                    source_event=event.event_id,
                )
            )

        elif event.event_type == "翻倒" and event.target_id is not None:
            inferred_states.append(
                StateInference(
                    target_id=event.target_id,
                    target_label=event.target_label or "",
                    state="翻倒",
                    source_event=event.event_id,
                )
            )
            key = ("翻倒", event.target_id)
            if key not in attr_index:
                attr_index.add(key)
                derived_attributes.append(
                    AttributeEdge(
                        attribute_id=f"{event.event_id}_attr",
                        label="翻倒",
                        normalized="翻倒",
                        target_id=event.target_id,
                        target_label=event.target_label,
                        confidence=min(0.98, event.confidence),
                        segment_id=event.segment_id,
                        span=event.span,
                        derived=True,
                    )
                )

        elif event.event_type == "泄漏" and event.actor_id is not None and event.target_id is not None:
            inferred_states.append(
                StateInference(
                    target_id=event.actor_id,
                    target_label=event.actor_label or "",
                    state=f"覆盖于{event.target_label}",
                    source_event=event.event_id,
                )
            )
            rel_key = ("覆盖", event.actor_id, event.target_id)
            if rel_key not in relation_index:
                relation_index.add(rel_key)
                derived_relations.append(
                    RelationEdge(
                        relation_id=f"{event.event_id}_rel",
                        label="覆盖",
                        normalized="覆盖",
                        source_id=event.actor_id,
                        source_label=event.actor_label or "",
                        target_id=event.target_id,
                        target_label=event.target_label or "",
                        confidence=min(0.95, event.confidence),
                        segment_id=event.segment_id,
                        span=event.span,
                        derived=True,
                    )
                )

        elif event.event_type == "堆放":
            for object_id, object_label in zip(event.target_ids, event.target_labels):
                inferred_states.append(
                    StateInference(
                        target_id=object_id,
                        target_label=object_label,
                        state="堆积",
                        source_event=event.event_id,
                    )
                )
                attr_key = ("堆积", object_id)
                if attr_key not in attr_index:
                    attr_index.add(attr_key)
                    derived_attributes.append(
                        AttributeEdge(
                            attribute_id=f"{event.event_id}_{object_id}_attr",
                            label="堆积",
                            normalized="堆积",
                            target_id=object_id,
                            target_label=object_label,
                            confidence=min(0.9, event.confidence),
                            segment_id=event.segment_id,
                            span=event.span,
                            derived=True,
                        )
                    )
                if event.location_id is not None:
                    rel_key = ("堆放于", object_id, event.location_id)
                    if rel_key not in relation_index:
                        relation_index.add(rel_key)
                        derived_relations.append(
                            RelationEdge(
                                relation_id=f"{event.event_id}_{object_id}_rel",
                                label="堆放于",
                                normalized="堆放于",
                                source_id=object_id,
                                source_label=object_label,
                                target_id=event.location_id,
                                target_label=event.location_label or "",
                                confidence=min(0.9, event.confidence),
                                segment_id=event.segment_id,
                                span=event.span,
                                derived=True,
                            )
                        )

        elif event.event_type == "散落" and event.target_id is not None:
            inferred_states.append(
                StateInference(
                    target_id=event.target_id,
                    target_label=event.target_label or "",
                    state="散落",
                    source_event=event.event_id,
                )
            )
            attr_key = ("散乱", event.target_id)
            if attr_key not in attr_index:
                attr_index.add(attr_key)
                derived_attributes.append(
                    AttributeEdge(
                        attribute_id=f"{event.event_id}_attr",
                        label="散乱",
                        normalized="散乱",
                        target_id=event.target_id,
                        target_label=event.target_label,
                        confidence=min(0.9, event.confidence),
                        segment_id=event.segment_id,
                        span=event.span,
                        derived=True,
                    )
                )
            if event.location_id is not None:
                rel_key = ("散落于", event.target_id, event.location_id)
                if rel_key not in relation_index:
                    relation_index.add(rel_key)
                    derived_relations.append(
                        RelationEdge(
                            relation_id=f"{event.event_id}_rel",
                            label="散落于",
                            normalized="散落于",
                            source_id=event.target_id,
                            source_label=event.target_label or "",
                            target_id=event.location_id,
                            target_label=event.location_label or "",
                            confidence=min(0.9, event.confidence),
                            segment_id=event.segment_id,
                            span=event.span,
                            derived=True,
                        )
                    )

    return inferred_states, derived_attributes, derived_relations
