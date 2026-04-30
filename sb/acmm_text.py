from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple


@dataclass(frozen=True)
class TextSignalRule:
    name: str
    object_type: str
    state: str
    keywords: Tuple[str, ...]
    risk: float
    task_value: float
    label: str

    def match_count(self, text: str) -> int:
        return sum(1 for keyword in self.keywords if keyword in text)


@dataclass(frozen=True)
class TextObservation:
    row_id: str
    text: str
    observation: Dict[str, object]
    weak_label: str
    weak_high_risk: bool
    matched_rules: Tuple[str, ...]

    def as_dict(self) -> Dict[str, object]:
        return {
            "row_id": self.row_id,
            "text": self.text,
            "observation": self.observation,
            "weak_label": self.weak_label,
            "weak_high_risk": self.weak_high_risk,
            "matched_rules": list(self.matched_rules),
        }


DEFAULT_TEXT_SIGNAL_RULES: Tuple[TextSignalRule, ...] = (
    TextSignalRule(
        name="pollution",
        object_type="污染线索",
        state="异常",
        keywords=("污染", "污水", "排污", "废水", "黑臭", "发黑", "泄漏", "有毒", "垃圾", "臭味"),
        risk=0.90,
        task_value=0.88,
        label="污染风险",
    ),
    TextSignalRule(
        name="mining",
        object_type="采掘扰动",
        state="裸地",
        keywords=("采矿", "采掘", "矿区", "矿山", "开采", "露天矿", "矿坑", "矿权"),
        risk=0.86,
        task_value=0.90,
        label="采掘扰动",
    ),
    TextSignalRule(
        name="construction_waste",
        object_type="建筑堆放",
        state="堆积",
        keywords=("施工", "工地", "建筑垃圾", "废弃物", "碎砖", "砖块", "木板", "渣土", "堆放"),
        risk=0.62,
        task_value=0.70,
        label="堆放异常",
    ),
    TextSignalRule(
        name="water",
        object_type="水体",
        state="待判断",
        keywords=("积水", "沟渠", "河道", "水体", "排水", "池塘", "雨水"),
        risk=0.45,
        task_value=0.55,
        label="水体线索",
    ),
    TextSignalRule(
        name="restoration",
        object_type="恢复治理",
        state="恢复",
        keywords=("恢复", "治理", "复绿", "植被", "林地", "耕地", "绿化", "修复"),
        risk=0.25,
        task_value=0.62,
        label="恢复治理",
    ),
    TextSignalRule(
        name="residential",
        object_type="居民区",
        state="敏感目标",
        keywords=("居民", "小区", "村庄", "房屋", "学校", "医院", "饮用水"),
        risk=0.55,
        task_value=0.78,
        label="敏感目标",
    ),
)


def find_manifest_dataset_path(
    manifest_path: str | Path,
    dataset_name: str = "chinese_c4_sample",
) -> Path:
    path = Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(f"未找到数据 manifest：{path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    for item in manifest.get("datasets", []):
        if item.get("name") == dataset_name and item.get("path"):
            dataset_path = Path(str(item["path"]))
            if dataset_path.exists():
                return dataset_path
            raise FileNotFoundError(f"manifest 中存在 {dataset_name}，但文件不存在：{dataset_path}")
    raise FileNotFoundError(f"manifest 中未找到数据集：{dataset_name}")


def iter_chinese_c4_texts(
    dataset_path: str | Path,
    *,
    limit: int | None = None,
    min_chars: int = 60,
) -> Iterable[Dict[str, object]]:
    count = 0
    with Path(dataset_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            text = str(row.get("text", "")).strip()
            if len(text) < min_chars:
                continue
            yield row
            count += 1
            if limit is not None and count >= limit:
                break


def build_text_observation(
    row: Mapping[str, object],
    *,
    rules: Sequence[TextSignalRule] = DEFAULT_TEXT_SIGNAL_RULES,
) -> TextObservation:
    text = " ".join(str(row.get("text", "")).split())
    row_id = str(row.get("id", ""))
    matches = [(rule, rule.match_count(text)) for rule in rules]
    active = [(rule, count) for rule, count in matches if count > 0]

    objects: List[Dict[str, object]] = []
    relations: List[Dict[str, object]] = []
    scores: Dict[str, float] = {"普通文本": 0.35}
    risk_score = 0.05
    task_value = 0.30
    conflict_score = 0.0

    for index, (rule, count) in enumerate(active, start=1):
        strength = min(1.0, 0.58 + 0.12 * count)
        objects.append(
            {
                "id": f"{row_id or 'row'}-{rule.name}",
                "type": rule.object_type,
                "state": rule.state,
                "confidence": round(strength, 4),
                "attributes": {"keyword_hits": float(count)},
            }
        )
        scores[rule.label] = scores.get(rule.label, 0.05) + rule.risk * strength
        risk_score = max(risk_score, rule.risk * strength)
        task_value = max(task_value, rule.task_value)
        if index > 1:
            relations.append(
                {
                    "source": f"{row_id or 'row'}-{active[index - 2][0].name}",
                    "relation": "文本共现",
                    "target": f"{row_id or 'row'}-{rule.name}",
                    "confidence": 0.55,
                }
            )

    active_names = {rule.name for rule, _ in active}
    if {"pollution", "residential"} <= active_names:
        risk_score = max(risk_score, 0.92)
        task_value = max(task_value, 0.92)
        conflict_score = max(conflict_score, 0.45)
    if {"mining", "restoration"} <= active_names:
        conflict_score = max(conflict_score, 0.55)
    if {"water", "pollution"} <= active_names:
        risk_score = max(risk_score, 0.88)

    if not objects:
        objects.append(
            {
                "id": f"{row_id or 'row'}-background",
                "type": "普通文本",
                "state": "背景",
                "confidence": 0.72,
            }
        )
        scores["普通文本"] = 0.82

    label_probabilities = _normalize_scores(scores)
    weak_label = max(label_probabilities.items(), key=lambda item: item[1])[0]
    weak_high_risk = risk_score >= 0.55 or conflict_score >= 0.45
    observation = {
        "objects": objects,
        "relations": relations,
        "label_probabilities": label_probabilities,
        "risk_score": round(risk_score, 4),
        "task_value": round(task_value, 4),
        "rule_violation_score": round(conflict_score, 4),
        "timestamp": str(row.get("timestamp", "")),
        "source_text_id": row_id,
    }
    return TextObservation(
        row_id=row_id,
        text=text,
        observation=observation,
        weak_label=weak_label,
        weak_high_risk=weak_high_risk,
        matched_rules=tuple(rule.name for rule, _ in active),
    )


def _normalize_scores(scores: Mapping[str, float]) -> Dict[str, float]:
    clean = {str(key): max(0.0, float(value)) for key, value in scores.items()}
    total = sum(clean.values())
    if total <= 1e-12:
        return {"普通文本": 1.0}
    return {key: round(value / total, 6) for key, value in clean.items()}
