from __future__ import annotations

import json

from sb.emotion_feedback import (
    build_emotion_feedback,
    build_emotion_supervision,
    emotion_supervision_loss_spec,
)


def main() -> None:
    result = {
        "objects": [{"name": "沟渠"}, {"name": "积水"}, {"name": "居民房"}],
        "attributes": [{"target": "积水", "name": "发黑"}],
        "relations": [{"source": "积水", "type": "靠近", "target": "居民房"}],
        "events": [{"name": "污染扩散", "confidence": 0.62}],
        "scene_hypotheses": [
            {"name": "污染/异常积水", "score": 0.79},
            {"name": "普通积水", "score": 0.56},
            {"name": "施工排水", "score": 0.31},
        ],
        "temporary_concepts": [{"name": "黑臭水体"}],
        "conflicts": [],
    }
    feedback = build_emotion_feedback(
        result,
        runtime={
            "memory_pressure": 0.22,
            "token_budget_ratio": 0.35,
            "propagation_budget_ratio": 0.28,
        },
    )
    supervision = build_emotion_supervision(feedback, label_source="heuristic_scene_rules")
    print(
        json.dumps(
            {
                "feedback": feedback.as_dict(),
                "supervision": supervision.as_dict(),
                "loss_spec": emotion_supervision_loss_spec(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
