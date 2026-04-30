from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Dict, List, Mapping, Sequence

from sb.acmm import ACMMConfig, AffectiveCausalMemoryModel, CausalGraph, CausalRule


@dataclass(frozen=True)
class ValidationCase:
    case_id: str
    true_label: str
    observation: Dict[str, object]
    high_risk: bool


def build_validation_model() -> AffectiveCausalMemoryModel:
    return AffectiveCausalMemoryModel(
        causal_graph=CausalGraph(
            rules=[
                CausalRule("采掘扰动", "裸地", confidence=0.82, description="采掘扰动通常导致地表裸露"),
                CausalRule("治理工程", "植被恢复", confidence=0.68, description="治理工程后应出现恢复趋势"),
                CausalRule("发黑积水", "污染风险", confidence=0.76, description="发黑积水增加污染风险"),
                CausalRule("植被下降", "恢复异常", confidence=0.72, description="恢复区连续退化应进入复核"),
            ]
        ),
        config=ACMMConfig(embedding_dim=32, review_threshold=0.76, alert_threshold=0.70),
    )


def reference_embedding(model: AffectiveCausalMemoryModel, kind: str) -> List[float]:
    templates = {
        "restored": {
            "objects": [
                {"id": "ref-plot", "type": "恢复区", "state": "植被恢复", "confidence": 0.88},
                {"id": "ref-road", "type": "道路", "state": "稳定", "confidence": 0.80},
            ]
        },
        "water": {
            "objects": [
                {"id": "ref-water", "type": "季节性积水", "state": "临时水体", "confidence": 0.82},
                {"id": "ref-grass", "type": "草地", "state": "稳定", "confidence": 0.74},
            ]
        },
    }
    return list(model.objectify(templates[kind]).embedding(model.config.embedding_dim))


def mechanism_checks() -> Dict[str, object]:
    model = build_validation_model()
    normal_reference = reference_embedding(model, "restored")
    normal_case = {
        "objects": [
            {"id": "plot-n", "type": "恢复区", "state": "植被恢复", "confidence": 0.88},
            {"id": "road-n", "type": "道路", "state": "稳定", "confidence": 0.80},
        ],
        "predicted_embedding": normal_reference,
        "label_probabilities": {"恢复区": 0.86, "污染风险": 0.06, "违法采掘": 0.04, "其他": 0.04},
        "risk_score": 0.05,
        "task_value": 0.35,
        "rule_violation_score": 0.0,
    }
    risk_case = {
        "objects": [
            {"id": "plot-r", "type": "采掘扰动", "state": "裸地", "confidence": 0.93},
            {"id": "water-r", "type": "发黑积水", "state": "异常", "confidence": 0.86},
            {"id": "house-r", "type": "居民房", "state": "邻近", "confidence": 0.78},
        ],
        "relations": [{"source": "water-r", "relation": "靠近", "target": "house-r", "confidence": 0.80}],
        "predicted_embedding": normal_reference,
        "label_probabilities": {"恢复区": 0.36, "污染风险": 0.34, "违法采掘": 0.24, "其他": 0.06},
        "risk_score": 0.91,
        "task_value": 0.94,
        "rule_violation_score": 0.62,
    }

    normal = model.cognitive_step(normal_case)
    first_risk = model.cognitive_step(risk_case)
    repeated_risk = model.cognitive_step(risk_case)

    checks = {
        "surprise_increases_on_prediction_error": first_risk.emotion.surprise > normal.emotion.surprise + 0.20,
        "review_gate_increases_on_risk_conflict": first_risk.gates.request_review > normal.gates.request_review + 0.20,
        "write_gate_increases_on_novelty": first_risk.gates.write_memory > normal.gates.write_memory + 0.15,
        "novelty_drops_after_memory_write": repeated_risk.emotion.novelty < first_risk.emotion.novelty - 0.30,
        "risk_case_writes_counterexample_or_rule": any(
            item.memory_type in {"counterexample", "rule", "causal"} for item in first_risk.memory_writes
        ),
    }
    if not all(checks.values()):
        raise AssertionError(f"ACMM 机制验证失败：{checks}")
    return {
        "checks": checks,
        "normal": {"emotion": normal.emotion.as_dict(), "gates": normal.gates.as_dict()},
        "first_risk": {"emotion": first_risk.emotion.as_dict(), "gates": first_risk.gates.as_dict()},
        "repeated_risk": {"emotion": repeated_risk.emotion.as_dict(), "gates": repeated_risk.gates.as_dict()},
        "memory_after_checks": model.memory.as_dict(),
    }


def build_synthetic_cases(model: AffectiveCausalMemoryModel) -> List[ValidationCase]:
    restored = reference_embedding(model, "restored")
    water = reference_embedding(model, "water")
    cases: List[ValidationCase] = []

    for index in range(8):
        cases.append(
            ValidationCase(
                case_id=f"normal-restored-{index}",
                true_label="恢复区",
                high_risk=False,
                observation={
                    "objects": [
                        {"id": f"plot-n-{index}", "type": "恢复区", "state": "植被恢复", "confidence": 0.86},
                        {"id": f"road-n-{index}", "type": "道路", "state": "稳定", "confidence": 0.75},
                    ],
                    "predicted_embedding": restored,
                    "label_probabilities": {"恢复区": 0.84, "违法采掘": 0.05, "污染风险": 0.05, "普通积水": 0.06},
                    "risk_score": 0.06,
                    "task_value": 0.35,
                    "rule_violation_score": 0.0,
                },
            )
        )

    for index in range(8):
        probabilities = (
            {"恢复区": 0.40, "违法采掘": 0.35, "污染风险": 0.20, "普通积水": 0.05}
            if index % 2 == 0
            else {"违法采掘": 0.42, "恢复区": 0.30, "污染风险": 0.22, "普通积水": 0.06}
        )
        cases.append(
            ValidationCase(
                case_id=f"illegal-mining-{index}",
                true_label="违法采掘",
                high_risk=True,
                observation={
                    "objects": [
                        {"id": f"mine-{index}", "type": "采掘扰动", "state": "裸地", "confidence": 0.91},
                        {"id": f"truck-{index}", "type": "车辆", "state": "活动", "confidence": 0.72},
                    ],
                    "relations": [{"source": f"mine-{index}", "relation": "邻接", "target": f"truck-{index}"}],
                    "predicted_embedding": restored,
                    "label_probabilities": probabilities,
                    "risk_score": 0.92,
                    "task_value": 0.95,
                    "rule_violation_score": 0.64,
                },
            )
        )

    for index in range(6):
        cases.append(
            ValidationCase(
                case_id=f"polluted-water-{index}",
                true_label="污染风险",
                high_risk=True,
                observation={
                    "objects": [
                        {"id": f"water-p-{index}", "type": "发黑积水", "state": "异常", "confidence": 0.88},
                        {"id": f"home-p-{index}", "type": "居民房", "state": "邻近", "confidence": 0.70},
                    ],
                    "relations": [{"source": f"water-p-{index}", "relation": "靠近", "target": f"home-p-{index}"}],
                    "predicted_embedding": water,
                    "label_probabilities": {"普通积水": 0.37, "污染风险": 0.34, "恢复区": 0.19, "违法采掘": 0.10},
                    "risk_score": 0.86,
                    "task_value": 0.88,
                    "rule_violation_score": 0.52,
                },
            )
        )

    for index in range(4):
        cases.append(
            ValidationCase(
                case_id=f"seasonal-water-{index}",
                true_label="普通积水",
                high_risk=False,
                observation={
                    "objects": [
                        {"id": f"water-s-{index}", "type": "季节性积水", "state": "临时水体", "confidence": 0.82},
                        {"id": f"grass-s-{index}", "type": "草地", "state": "稳定", "confidence": 0.76},
                    ],
                    "predicted_embedding": water,
                    "label_probabilities": {"普通积水": 0.44, "污染风险": 0.28, "恢复区": 0.20, "违法采掘": 0.08},
                    "risk_score": 0.30,
                    "task_value": 0.48,
                    "rule_violation_score": 0.08,
                },
            )
        )

    for index in range(4):
        cases.append(
            ValidationCase(
                case_id=f"restoration-conflict-{index}",
                true_label="需复核",
                high_risk=True,
                observation={
                    "objects": [
                        {"id": f"plot-c-{index}", "type": "恢复区", "state": "植被下降", "confidence": 0.84},
                        {"id": f"bare-c-{index}", "type": "裸地", "state": "扩张", "confidence": 0.74},
                    ],
                    "predicted_embedding": restored,
                    "label_probabilities": {"恢复区": 0.35, "裸地": 0.32, "污染风险": 0.24, "其他": 0.09},
                    "risk_score": 0.66,
                    "task_value": 0.82,
                    "rule_violation_score": 0.74,
                },
            )
        )

    return cases


def run_synthetic_benchmark(seed: int, random_trials: int) -> Dict[str, object]:
    model = build_validation_model()
    cases = build_synthetic_cases(model)
    rows: List[Dict[str, object]] = []
    for case in cases:
        result = model.cognitive_step(case.observation)
        pred = _argmax(case.observation["label_probabilities"])
        error = pred != case.true_label
        review_score = max(result.gates.request_review, result.gates.trigger_alert)
        rows.append(
            {
                "case_id": case.case_id,
                "true_label": case.true_label,
                "predicted_label": pred,
                "error": error,
                "high_risk": case.high_risk,
                "review_score": review_score,
                "emotion": result.emotion.as_dict(),
                "gates": result.gates.as_dict(),
                "memory_writes": [item.memory_type for item in result.memory_writes],
                "action_plan": list(result.action_plan),
            }
        )

    budget = max(1, round(len(rows) * 0.30))
    ranked = sorted(range(len(rows)), key=lambda idx: float(rows[idx]["review_score"]), reverse=True)
    acmm_selected = set(ranked[:budget])
    random_metrics = _random_selection_metrics(rows, budget=budget, seed=seed, trials=random_trials)
    acmm_metrics = _selection_metrics(rows, acmm_selected)

    checks = {
        "acmm_error_capture_beats_random": acmm_metrics["error_capture_rate"] > random_metrics["error_capture_rate_mean"],
        "acmm_high_risk_capture_beats_random": (
            acmm_metrics["high_risk_capture_rate"] > random_metrics["high_risk_capture_rate_mean"]
        ),
        "acmm_review_precision_beats_random": acmm_metrics["review_precision"] > random_metrics["review_precision_mean"],
    }
    if not all(checks.values()):
        raise AssertionError(f"ACMM 合成对照验证失败：{checks}")

    return {
        "case_count": len(rows),
        "review_budget": budget,
        "baseline_accuracy": round(mean(0.0 if row["error"] else 1.0 for row in rows), 4),
        "baseline_error_count": sum(1 for row in rows if row["error"]),
        "high_risk_count": sum(1 for row in rows if row["high_risk"]),
        "checks": checks,
        "acmm_top_budget": acmm_metrics,
        "random_top_budget": random_metrics,
        "memory_after_benchmark": model.memory.as_dict(),
        "top_reviewed_cases": [rows[idx] for idx in ranked[:budget]],
    }


def _selection_metrics(rows: Sequence[Mapping[str, object]], selected: set[int]) -> Dict[str, float]:
    total_errors = sum(1 for row in rows if row["error"])
    total_high_risk = sum(1 for row in rows if row["high_risk"])
    reviewed_errors = sum(1 for idx in selected if rows[idx]["error"])
    reviewed_high_risk = sum(1 for idx in selected if rows[idx]["high_risk"])
    budget = max(1, len(selected))
    return {
        "review_precision": round(reviewed_errors / budget, 4),
        "high_risk_precision": round(reviewed_high_risk / budget, 4),
        "error_capture_rate": round(reviewed_errors / max(1, total_errors), 4),
        "high_risk_capture_rate": round(reviewed_high_risk / max(1, total_high_risk), 4),
    }


def _random_selection_metrics(
    rows: Sequence[Mapping[str, object]],
    *,
    budget: int,
    seed: int,
    trials: int,
) -> Dict[str, float]:
    rng = random.Random(seed)
    metrics = []
    indices = list(range(len(rows)))
    for _ in range(trials):
        selected = set(rng.sample(indices, budget))
        metrics.append(_selection_metrics(rows, selected))
    return {
        "review_precision_mean": round(mean(item["review_precision"] for item in metrics), 4),
        "high_risk_precision_mean": round(mean(item["high_risk_precision"] for item in metrics), 4),
        "error_capture_rate_mean": round(mean(item["error_capture_rate"] for item in metrics), 4),
        "high_risk_capture_rate_mean": round(mean(item["high_risk_capture_rate"] for item in metrics), 4),
        "trials": trials,
    }


def _argmax(probabilities: object) -> str:
    if not isinstance(probabilities, Mapping) or not probabilities:
        return "unknown"
    return max(probabilities.items(), key=lambda item: float(item[1]))[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate ACMM mechanism and synthetic review selection.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--random-trials", type=int, default=200)
    parser.add_argument("--output-path", type=Path, default=None)
    args = parser.parse_args()

    report = {
        "mechanism": mechanism_checks(),
        "synthetic_benchmark": run_synthetic_benchmark(seed=args.seed, random_trials=args.random_trials),
    }
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output_path is not None:
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        args.output_path.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
