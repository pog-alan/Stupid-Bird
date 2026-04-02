from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from .v01_types import AnalysisResult


@dataclass(frozen=True)
class ProactiveQuestion:
    question: str
    reason: str
    priority: int


def propose_questions(
    result: AnalysisResult,
    template_path: str | Path | None = None,
) -> List[ProactiveQuestion]:
    templates = _load_templates(template_path)
    questions: List[ProactiveQuestion] = []
    if result.best_hypothesis is None:
        return [
            ProactiveQuestion(
                _choose(templates["missing_objects"]),
                "缺乏场景核心对象",
                1,
            )
        ]

    if len(result.scene_hypotheses) >= 2:
        gap = abs(result.scene_hypotheses[0].score - result.scene_hypotheses[1].score)
        if gap < 0.08:
            questions.append(
                ProactiveQuestion(
                    _choose(templates["scene_competitive"]),
                    "候选场景分数接近",
                    2,
                )
            )

    if not result.objects:
        questions.append(
            ProactiveQuestion(
                _choose(templates["missing_objects"]),
                "对象缺失",
                1,
            )
        )
    elif not result.attributes:
        questions.append(
            ProactiveQuestion(
                _choose(templates["missing_attributes"]),
                "属性缺失",
                3,
            )
        )

    if not result.relations:
        questions.append(
            ProactiveQuestion(
                _choose(templates["missing_relations"]),
                "关系缺失",
                4,
            )
        )

    if not result.events:
        questions.append(
            ProactiveQuestion(
                _choose(templates["missing_events"]),
                "事件缺失",
                5,
            )
        )

    questions.sort(key=lambda item: item.priority)
    return questions


def append_questions_to_payload(payload: Dict[str, object], result: AnalysisResult) -> Dict[str, object]:
    questions = propose_questions(result)
    payload["proactive_questions"] = [
        {"question": item.question, "reason": item.reason, "priority": item.priority}
        for item in questions[:2]
    ]
    return payload


def _choose(options: List[str]) -> str:
    return options[0] if options else "能补充一下更具体的细节吗？"


def _load_templates(template_path: str | Path | None) -> Dict[str, List[str]]:
    defaults = {
        "scene_competitive": ["这个场景更像是异常堆放还是正常摆放？"],
        "missing_objects": ["能补充一下场景里最关键的对象是什么吗？"],
        "missing_attributes": ["这些物体有没有明显的状态或特征（例如散乱、破碎、翻倒）？"],
        "missing_relations": ["这些物体之间有位置关系吗（比如在上面、旁边、里面）？"],
        "missing_events": ["这个场景里有没有发生动作或变化（比如碰倒、散落、泄漏）？"],
    }
    if template_path is None:
        template_path = Path(__file__).resolve().parents[1] / "文档" / "笨鸟v0.1主动提问模板.json"
    path = Path(template_path)
    if not path.exists():
        return defaults
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return defaults
    for key, fallback in defaults.items():
        if key not in raw or not isinstance(raw[key], list) or not raw[key]:
            raw[key] = fallback
    return raw
