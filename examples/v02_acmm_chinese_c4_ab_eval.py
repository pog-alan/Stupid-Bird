from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from statistics import mean
from typing import Dict, List, Mapping, Sequence, Tuple

from examples.v02_acmm_chinese_c4_eval import build_model
from sb.acmm import ACMMConfig, AffectiveCausalMemoryModel, CausalGraph


DEFAULT_POSITIVE_LABELS = ("弱风险线索", "高风险/需复核")


def load_label_records(
    path: Path,
    *,
    allow_weak_labels: bool,
    positive_labels: Sequence[str],
) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            human_label = str(raw.get("human_label", "") or "")
            if human_label:
                label = human_label
                label_source = "human"
                is_positive = label in positive_labels
            elif allow_weak_labels:
                label = "高风险/需复核" if bool(raw.get("weak_high_risk")) else "普通文本"
                label_source = "weak"
                is_positive = bool(raw.get("weak_high_risk"))
            else:
                continue
            if label == "不确定":
                continue
            item = dict(raw)
            item["eval_label"] = label
            item["eval_label_source"] = label_source
            item["eval_positive"] = is_positive
            records.append(item)
    return records


def score_records(records: Sequence[Mapping[str, object]]) -> Dict[str, List[float]]:
    full_acmm = build_model()
    memory_only = AffectiveCausalMemoryModel(
        causal_graph=CausalGraph(),
        config=ACMMConfig(embedding_dim=32, write_threshold=0.60, review_threshold=0.74, alert_threshold=0.68),
    )
    scores = {"baseline": [], "memory": [], "causal": [], "acmm": []}
    for record in records:
        observation = dict(record.get("observation", {}) or {})
        scores["baseline"].append(_baseline_score(record, observation))
        memory_result = memory_only.cognitive_step(_without_causal_pressure(observation))
        scores["memory"].append(
            round(
                0.45 * memory_result.emotion.novelty
                + 0.25 * memory_result.emotion.uncertainty
                + 0.20 * memory_result.gates.write_memory
                + 0.10 * memory_result.emotion.value,
                4,
            )
        )
        causal_result = build_model().cognitive_step(observation)
        scores["causal"].append(max(causal_result.gates.request_review, causal_result.gates.trigger_alert))
        acmm_result = full_acmm.cognitive_step(observation)
        scores["acmm"].append(max(acmm_result.gates.request_review, acmm_result.gates.trigger_alert, acmm_result.gates.update_rule))
    return scores


def evaluate_methods(
    records: Sequence[Mapping[str, object]],
    scores: Mapping[str, Sequence[float]],
    *,
    review_ratio: float,
    random_trials: int,
    seed: int,
) -> Dict[str, object]:
    budget = max(1, round(len(records) * review_ratio))
    positives = [bool(item["eval_positive"]) for item in records]
    method_reports = {}
    for method, method_scores in scores.items():
        ranked = sorted(range(len(records)), key=lambda index: float(method_scores[index]), reverse=True)
        selected = set(ranked[:budget])
        report = _selection_metrics(positives, selected)
        report["mean_selected_score"] = round(mean(float(method_scores[index]) for index in selected), 4)
        report["top_cases"] = [_case_summary(records[index], float(method_scores[index])) for index in ranked[: min(budget, 10)]]
        method_reports[method] = report
    method_reports["random"] = _random_metrics(positives, budget=budget, random_trials=random_trials, seed=seed)
    base_rate = sum(positives) / max(1, len(positives))
    return {
        "record_count": len(records),
        "positive_count": sum(positives),
        "positive_rate": round(base_rate, 4),
        "review_budget": budget,
        "review_ratio": review_ratio,
        "methods": method_reports,
        "best_method_by_precision": max(
            (name for name in method_reports if name != "random"),
            key=lambda name: method_reports[name]["precision_at_budget"],
        ),
        "notes": [
            "baseline 为关键词/弱规则风险分数；memory 为无因果图的记忆新颖度排序；causal 为因果/风险门控但不累积长期记忆；acmm 为记忆+因果+情绪门控。",
            "如果 eval_label_source=weak，本报告只能作为流程 smoke，不能证明真实人工标签收益。",
        ],
    }


def _baseline_score(record: Mapping[str, object], observation: Mapping[str, object]) -> float:
    risk = float(observation.get("risk_score", 0.0) or 0.0)
    conflict = float(observation.get("rule_violation_score", 0.0) or 0.0)
    matched = len(record.get("matched_rules", []) or [])
    weak_high_risk = 1.0 if record.get("weak_high_risk") else 0.0
    return round(min(1.0, 0.62 * risk + 0.22 * conflict + 0.05 * matched + 0.11 * weak_high_risk), 4)


def _without_causal_pressure(observation: Mapping[str, object]) -> Dict[str, object]:
    clean = dict(observation)
    clean["risk_score"] = 0.05
    clean["rule_violation_score"] = 0.0
    return clean


def _selection_metrics(positives: Sequence[bool], selected: set[int]) -> Dict[str, float]:
    selected_positive = sum(1 for index in selected if positives[index])
    total_positive = sum(1 for item in positives if item)
    budget = max(1, len(selected))
    return {
        "precision_at_budget": round(selected_positive / budget, 4),
        "recall_at_budget": round(selected_positive / max(1, total_positive), 4),
        "selected_positive": selected_positive,
    }


def _random_metrics(
    positives: Sequence[bool],
    *,
    budget: int,
    random_trials: int,
    seed: int,
) -> Dict[str, float]:
    rng = random.Random(seed)
    indices = list(range(len(positives)))
    metrics = []
    for _ in range(random_trials):
        metrics.append(_selection_metrics(positives, set(rng.sample(indices, budget))))
    return {
        "precision_at_budget_mean": round(mean(item["precision_at_budget"] for item in metrics), 4),
        "recall_at_budget_mean": round(mean(item["recall_at_budget"] for item in metrics), 4),
        "trials": random_trials,
    }


def _case_summary(record: Mapping[str, object], score: float) -> Dict[str, object]:
    return {
        "row_id": record.get("row_id"),
        "score": round(score, 4),
        "eval_label": record.get("eval_label"),
        "eval_label_source": record.get("eval_label_source"),
        "weak_label": record.get("weak_label"),
        "weak_high_risk": record.get("weak_high_risk"),
        "matched_rules": record.get("matched_rules", []),
        "text_preview": str(record.get("text", ""))[:120],
    }


def parse_positive_labels(text: str) -> Tuple[str, ...]:
    labels = tuple(item.strip() for item in text.split(",") if item.strip())
    if not labels:
        raise ValueError("positive-labels 不能为空。")
    return labels


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="A/B evaluate ACMM review ranking on labeled Chinese-C4 samples.")
    parser.add_argument("--label-path", type=Path, default=Path("data/processed/labels/acmm_chinese_c4_labels.jsonl"))
    parser.add_argument("--output-path", type=Path, default=Path("data/processed/experiments/acmm_chinese_c4_ab_eval.json"))
    parser.add_argument("--review-ratio", type=float, default=0.20)
    parser.add_argument("--random-trials", type=int, default=200)
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--positive-labels", default=",".join(DEFAULT_POSITIVE_LABELS))
    parser.add_argument("--allow-weak-labels", action="store_true")
    return parser


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = build_parser().parse_args()
    if not 0.0 < args.review_ratio <= 1.0:
        raise ValueError("review-ratio 必须在 (0, 1] 内。")
    positives = parse_positive_labels(args.positive_labels)
    records = load_label_records(args.label_path, allow_weak_labels=args.allow_weak_labels, positive_labels=positives)
    if not records:
        raise RuntimeError(
            f"没有可评测标签：{args.label_path}。请先运行 label tool 标注，或显式添加 --allow-weak-labels 做 smoke。"
        )
    scores = score_records(records)
    report = evaluate_methods(
        records,
        scores,
        review_ratio=args.review_ratio,
        random_trials=args.random_trials,
        seed=args.seed,
    )
    report["label_path"] = str(args.label_path)
    report["positive_labels"] = list(positives)
    report["allow_weak_labels"] = bool(args.allow_weak_labels)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
