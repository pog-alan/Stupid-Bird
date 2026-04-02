from __future__ import annotations

from typing import Dict

from .v01_types import AnalysisResult


def build_output(result: AnalysisResult) -> Dict[str, object]:
    scene_hypotheses = [{"label": item.label, "score": round(item.score, 3)} for item in result.scene_hypotheses]
    best = None
    if result.best_hypothesis is not None:
        best = {"label": result.best_hypothesis.label, "score": round(result.best_hypothesis.score, 3)}

    competitive = False
    if len(result.scene_hypotheses) >= 2:
        competitive = abs(result.scene_hypotheses[0].score - result.scene_hypotheses[1].score) < 0.08

    return {
        "input_text": result.input_text,
        "objects": [
            {
                "id": item.object_id,
                "label": item.normalized,
                "surface": item.label,
                "category": item.category,
                "confidence": round(item.confidence, 3),
            }
            for item in result.objects
        ],
        "attributes": [
            {
                "id": item.attribute_id,
                "target": item.target_id,
                "target_label": item.target_label,
                "label": item.normalized,
                "confidence": round(item.confidence, 3),
                "derived": item.derived,
            }
            for item in result.attributes
        ],
        "relations": [
            {
                "id": item.relation_id,
                "source": item.source_id,
                "source_label": item.source_label,
                "type": item.label,
                "normalized_type": item.normalized,
                "target": item.target_id,
                "target_label": item.target_label,
                "confidence": round(item.confidence, 3),
                "derived": item.derived,
            }
            for item in result.relations
        ],
        "events": [
            {
                "id": item.event_id,
                "type": item.event_type,
                "actor": item.actor_id,
                "actor_label": item.actor_label,
                "target": item.target_id,
                "target_label": item.target_label,
                "location": item.location_id,
                "location_label": item.location_label,
                "targets": list(item.target_ids),
                "target_labels": list(item.target_labels),
                "confidence": round(item.confidence, 3),
            }
            for item in result.events
        ],
        "scene_hypotheses": scene_hypotheses,
        "best_hypothesis": best,
        "reasoning_path": [
            {
                "step": item.step,
                "signal": item.signal,
                "activated_spaces": list(item.activated_spaces),
                "contribution": item.contribution,
            }
            for item in result.reasoning_path
        ],
        "state_inference": [
            {
                "target": item.target_id,
                "target_label": item.target_label,
                "state": item.state,
                "source_event": item.source_event,
            }
            for item in result.state_inference
        ],
        "competitive_explanation": competitive,
    }
