from __future__ import annotations

import json

from sb.acmm import (
    ACMMConfig,
    AffectiveCausalMemoryModel,
    CausalGraph,
    CausalRule,
    acmm_model_spec,
)


def main() -> None:
    model = AffectiveCausalMemoryModel(
        causal_graph=CausalGraph(
            rules=[
                CausalRule("采掘扰动", "裸地", confidence=0.82, description="采掘扰动通常导致地表裸露"),
                CausalRule("治理工程", "植被恢复", confidence=0.68, description="治理工程后应出现恢复趋势"),
                CausalRule("发黑积水", "污染风险", confidence=0.76, description="发黑积水增加污染风险"),
            ]
        ),
        config=ACMMConfig(embedding_dim=32),
    )

    expected_normal = model.objectify(
        {
            "objects": [
                {"id": "plot-a", "type": "恢复区", "state": "植被恢复", "confidence": 0.82},
                {"id": "road-b", "type": "道路", "state": "稳定", "confidence": 0.77},
            ]
        }
    ).embedding(32)

    result = model.cognitive_step(
        {
            "timestamp": "2026-04-30",
            "objects": [
                {"id": "plot-a", "type": "采掘扰动", "state": "裸地", "confidence": 0.91},
                {"id": "water-c", "type": "发黑积水", "state": "异常", "confidence": 0.84},
                {"id": "house-d", "type": "居民房", "state": "邻近", "confidence": 0.73},
            ],
            "relations": [
                {"source": "water-c", "relation": "靠近", "target": "house-d", "confidence": 0.78},
                {"source": "plot-a", "relation": "邻接", "target": "water-c", "confidence": 0.66},
            ],
            "predicted_embedding": list(expected_normal),
            "label_probabilities": {"恢复区": 0.18, "裸地": 0.34, "污染风险": 0.41, "其他": 0.07},
            "risk_score": 0.88,
            "task_value": 0.92,
            "rule_violation_score": 0.58,
        }
    )
    print(json.dumps({"spec": acmm_model_spec(), "step": result.as_dict()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
