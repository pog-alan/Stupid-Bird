from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from statistics import mean
from typing import Dict, List, Mapping, Sequence

from sb.acmm import ACMMConfig, AffectiveCausalMemoryModel, CausalGraph, CausalRule
from sb.acmm_text import build_text_observation, find_manifest_dataset_path, iter_chinese_c4_texts


def build_model() -> AffectiveCausalMemoryModel:
    return AffectiveCausalMemoryModel(
        causal_graph=CausalGraph(
            rules=[
                CausalRule("污染线索", "污染风险", confidence=0.82),
                CausalRule("采掘扰动", "裸地", confidence=0.80),
                CausalRule("建筑堆放", "堆放异常", confidence=0.68),
                CausalRule("恢复治理", "恢复", confidence=0.62),
                CausalRule("居民区", "敏感目标", confidence=0.72),
            ]
        ),
        config=ACMMConfig(embedding_dim=32, write_threshold=0.60, review_threshold=0.74, alert_threshold=0.68),
    )


def run_eval(
    *,
    dataset_path: Path,
    limit: int,
    min_chars: int,
    review_ratio: float,
    random_trials: int,
    seed: int,
) -> Dict[str, object]:
    model = build_model()
    rows = list(iter_chinese_c4_texts(dataset_path, limit=limit, min_chars=min_chars))
    if not rows:
        raise RuntimeError(f"Chinese-C4 样本为空或均短于 min_chars：{dataset_path}")

    evaluated: List[Dict[str, object]] = []
    for row in rows:
        text_observation = build_text_observation(row)
        result = model.cognitive_step(text_observation.observation)
        review_score = max(result.gates.request_review, result.gates.trigger_alert, result.gates.update_rule)
        evaluated.append(
            {
                "row_id": text_observation.row_id,
                "weak_label": text_observation.weak_label,
                "weak_high_risk": text_observation.weak_high_risk,
                "matched_rules": list(text_observation.matched_rules),
                "review_score": review_score,
                "emotion": result.emotion.as_dict(),
                "gates": result.gates.as_dict(),
                "memory_writes": [item.memory_type for item in result.memory_writes],
                "text_preview": text_observation.text[:160],
            }
        )

    budget = max(1, round(len(evaluated) * review_ratio))
    ranked = sorted(range(len(evaluated)), key=lambda index: float(evaluated[index]["review_score"]), reverse=True)
    selected = set(ranked[:budget])
    acmm_metrics = _selection_metrics(evaluated, selected)
    random_metrics = _random_selection_metrics(evaluated, budget=budget, trials=random_trials, seed=seed)
    weak_high_risk_count = sum(1 for item in evaluated if item["weak_high_risk"])
    matched_count = sum(1 for item in evaluated if item["matched_rules"])

    return {
        "dataset_path": str(dataset_path),
        "sample_count": len(evaluated),
        "review_budget": budget,
        "weak_high_risk_count": weak_high_risk_count,
        "weak_high_risk_rate": round(weak_high_risk_count / len(evaluated), 4),
        "matched_rule_count": matched_count,
        "matched_rule_rate": round(matched_count / len(evaluated), 4),
        "acmm_top_budget": acmm_metrics,
        "random_top_budget": random_metrics,
        "risk_lift_over_base_rate": round(
            acmm_metrics["weak_high_risk_precision"] / max(1e-9, weak_high_risk_count / len(evaluated)),
            4,
        ),
        "risk_lift_over_random": round(
            acmm_metrics["weak_high_risk_precision"]
            / max(1e-9, random_metrics["weak_high_risk_precision_mean"]),
            4,
        ),
        "memory_after_eval": model.memory.as_dict(),
        "top_reviewed": [evaluated[index] for index in ranked[: min(budget, 12)]],
        "notes": [
            "该评测使用关键词/规则生成弱标签，不等价人工标注真值。",
            "指标衡量 ACMM 门控是否能在真实中文文本中富集弱高风险样本。",
            "若要证明真实任务性能收益，需要接人工标注或下游任务标签做 A/B。",
        ],
    }


def _selection_metrics(rows: Sequence[Mapping[str, object]], selected: set[int]) -> Dict[str, float]:
    total_high_risk = sum(1 for item in rows if item["weak_high_risk"])
    selected_high_risk = sum(1 for index in selected if rows[index]["weak_high_risk"])
    selected_rule_matched = sum(1 for index in selected if rows[index]["matched_rules"])
    budget = max(1, len(selected))
    return {
        "weak_high_risk_precision": round(selected_high_risk / budget, 4),
        "weak_high_risk_capture_rate": round(selected_high_risk / max(1, total_high_risk), 4),
        "rule_matched_precision": round(selected_rule_matched / budget, 4),
    }


def _random_selection_metrics(
    rows: Sequence[Mapping[str, object]],
    *,
    budget: int,
    trials: int,
    seed: int,
) -> Dict[str, float]:
    rng = random.Random(seed)
    indices = list(range(len(rows)))
    all_metrics = []
    for _ in range(trials):
        selected = set(rng.sample(indices, budget))
        all_metrics.append(_selection_metrics(rows, selected))
    return {
        "weak_high_risk_precision_mean": round(mean(item["weak_high_risk_precision"] for item in all_metrics), 4),
        "weak_high_risk_capture_rate_mean": round(mean(item["weak_high_risk_capture_rate"] for item in all_metrics), 4),
        "rule_matched_precision_mean": round(mean(item["rule_matched_precision"] for item in all_metrics), 4),
        "trials": trials,
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Run ACMM weak-supervision evaluation on Chinese-C4 text.")
    parser.add_argument("--manifest-path", default="data/manifest.json")
    parser.add_argument("--dataset-name", default="chinese_c4_sample")
    parser.add_argument("--dataset-path", default="")
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--min-chars", type=int, default=80)
    parser.add_argument("--review-ratio", type=float, default=0.20)
    parser.add_argument("--random-trials", type=int, default=200)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--output-path", type=Path, default=Path("data/processed/experiments/acmm_chinese_c4_eval.json"))
    args = parser.parse_args()

    if not 0.0 < args.review_ratio <= 1.0:
        raise ValueError("review-ratio 必须在 (0, 1] 内。")
    dataset_path = (
        Path(args.dataset_path)
        if args.dataset_path
        else find_manifest_dataset_path(args.manifest_path, dataset_name=args.dataset_name)
    )
    report = run_eval(
        dataset_path=dataset_path,
        limit=args.limit,
        min_chars=args.min_chars,
        review_ratio=args.review_ratio,
        random_trials=args.random_trials,
        seed=args.seed,
    )
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output_path:
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        args.output_path.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
