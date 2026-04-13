from __future__ import annotations

import json

from sb.hierarchical_context import (
    HierarchicalContextSpec,
    MemoryLevel,
    MemoryNode,
    ReplayQuery,
)


def main() -> None:
    spec = HierarchicalContextSpec()

    nodes = [
        MemoryNode(
            node_id="scene-001",
            level=MemoryLevel.SUMMARY_SCENE,
            salience=0.90,
            stability=0.85,
            age=0.10,
            replay_hits=4,
            summarized=True,
            task_labels=["long_context_recall"],
            entities=["cup", "table"],
            relations=["on_left_of"],
            events=["spill"],
        ),
        MemoryNode(
            node_id="episode-014",
            level=MemoryLevel.SUMMARY_EPISODE,
            salience=0.72,
            stability=0.68,
            age=0.25,
            replay_hits=2,
            summarized=True,
            task_labels=["passkey_retrieval"],
            entities=["key", "query"],
            relations=["linked_to"],
            events=["store"],
        ),
        MemoryNode(
            node_id="episodic-113",
            level=MemoryLevel.EPISODIC,
            salience=0.66,
            stability=0.58,
            age=0.18,
            replay_hits=1,
            summarized=False,
            task_labels=["passkey_retrieval"],
            entities=["key", "query"],
            relations=["near"],
            events=["retrieve"],
        ),
    ]

    replay = spec.build_replay_plan(
        ReplayQuery(
            task_label="passkey_retrieval",
            entities=["key", "query"],
            required_levels=[MemoryLevel.SUMMARY_EPISODE, MemoryLevel.EPISODIC],
            budget=4,
        ),
        nodes,
    )
    mapping = spec.build_mapping_relation(
        source_ids=["episodic-113", "episode-014"],
        target_ids=["scene-001"],
        coverage=0.78,
        confidence=0.84,
    )
    merge_plan = spec.build_merge_plan(
        source_ids=["episodic-113", "episode-014"],
        target_ids=["summary-merge-01"],
        similarity=0.83,
        temporal_affinity=0.72,
        task_overlap=1.0,
        structural_match=0.68,
        conflict=0.06,
        coverage=0.74,
    )

    payload = {
        "promotion_equation": spec.promotion_equation(),
        "merge_equation": spec.merge_equation(),
        "forgetting_equation": spec.forgetting_equation(),
        "replay_equation": spec.replay_equation(),
        "mapping_relation": {
            "source_ids": list(mapping.source_ids),
            "target_ids": list(mapping.target_ids),
            "cardinality": mapping.cardinality.value,
            "coverage": mapping.coverage,
            "confidence": mapping.confidence,
        },
        "merge_plan": {
            "mode": merge_plan.mode.value,
            "source_ids": list(merge_plan.source_ids),
            "target_ids": list(merge_plan.target_ids),
            "retained_source_ids": list(merge_plan.retained_source_ids),
            "coverage": merge_plan.coverage,
            "score": round(merge_plan.score, 4),
        },
        "summary_plan": spec.build_summary_plan(
            {
                MemoryLevel.WORKING: 12,
                MemoryLevel.EPISODIC: 7,
                MemoryLevel.SUMMARY_WINDOW: 3,
            }
        ),
        "replay_schedule": spec.default_replay_schedule(),
        "replay_plan": [
            {
                "level": item.level.value,
                "node_id": item.node_id,
                "score": round(item.score, 4),
                "reason": item.reason,
            }
            for item in replay
        ],
        "forgetting_actions": {
            node.node_id: spec.forgetting_action(node)
            for node in nodes
        },
        "forgetting_steps": {
            node.node_id: {
                "stage": spec.forgetting_step(node).stage.value,
                "strength_scale": spec.forgetting_step(node).strength_scale,
                "value_scale": spec.forgetting_step(node).value_scale,
                "age_boost": spec.forgetting_step(node).age_boost,
            }
            for node in nodes
        },
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
