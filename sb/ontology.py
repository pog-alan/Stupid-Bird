from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple


def _slug(label: str) -> str:
    ascii_only = []
    for char in label.lower():
        if char.isalnum():
            ascii_only.append(char)
        elif "\u4e00" <= char <= "\u9fff":
            ascii_only.append(f"u{ord(char):x}")
    if not ascii_only:
        return "item"
    return "_".join(ascii_only)


@dataclass(frozen=True)
class SceneSpace:
    label: str
    support_objects: Tuple[str, ...]
    support_attributes: Tuple[str, ...]
    support_relations: Tuple[str, ...]
    support_events: Tuple[str, ...]
    reject_signals: Tuple[str, ...]
    base_score: float


@dataclass
class SBV01Ontology:
    config_path: Path
    raw_config: Mapping[str, object]
    limits: Dict[str, int]
    thresholds: Dict[str, float]
    depth_decay: Dict[int, float]
    local_activation_weights: Dict[str, float]
    global_hypothesis_weights: Dict[str, float]
    object_lexicon: Dict[str, Tuple[str, ...]]
    attribute_lexicon: Dict[str, Tuple[str, ...]]
    relation_patterns: List[Dict[str, object]]
    event_patterns: List[Dict[str, object]]
    scene_spaces: Dict[str, SceneSpace]
    mutual_exclusion: set[Tuple[str, str]]
    state_update_rules: List[Dict[str, object]]
    scene_hint_terms: Dict[str, Tuple[str, ...]] = field(default_factory=dict)

    object_alias_to_label: Dict[str, str] = field(default_factory=dict)
    attribute_alias_to_label: Dict[str, str] = field(default_factory=dict)
    object_categories: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.object_alias_to_label = self._build_alias_map(self.object_lexicon)
        self.attribute_alias_to_label = self._build_alias_map(self.attribute_lexicon)
        self.object_categories = self._build_object_categories(self.object_lexicon.keys())
        if not self.scene_hint_terms:
            self.scene_hint_terms = {
                "屋后空地": ("建筑废弃物堆放", "临时施工场景"),
                "院子角落": ("生活杂物堆放",),
                "院角": ("生活杂物堆放",),
                "路边": ("污染/异常积水", "建筑废弃物堆放"),
                "沟渠里": ("污染/异常积水",),
                "靠近居民房": ("污染/异常积水",),
                "门口": ("生活杂物堆放", "普通物体摆放"),
                "不整齐堆放": ("建筑废弃物堆放", "生活杂物堆放"),
                "摆放散乱": ("生活杂物堆放", "建筑废弃物堆放"),
            }

    @staticmethod
    def _build_alias_map(lexicon: Mapping[str, Sequence[str]]) -> Dict[str, str]:
        alias_map: Dict[str, str] = {}
        for label, aliases in lexicon.items():
            alias_map[label] = label
            for alias in aliases:
                alias_map[alias] = label
        return alias_map

    @staticmethod
    def _build_object_categories(labels: Iterable[str]) -> Dict[str, str]:
        categories: Dict[str, str] = {}
        container = {"杯子", "桶", "箱子", "纸箱", "袋子"}
        piled = {"砖块", "木板", "垃圾", "杂物", "生活杂物", "塑料袋", "纸箱"}
        environment = {"空地", "地面", "桌子", "桌面", "道路", "沟渠", "院子", "门口", "路边", "杂草"}
        attached = {"小卡车", "卡车", "货车", "房屋", "居民房", "树木", "垃圾桶"}
        living = {"猫"}
        liquid = {"水"}

        for label in labels:
            if label in container:
                categories[label] = "容器类"
            elif label in piled:
                categories[label] = "堆放物类"
            elif label in environment:
                categories[label] = "环境类"
            elif label in attached:
                categories[label] = "附属物类"
            elif label in living:
                categories[label] = "生物类"
            elif label in liquid:
                categories[label] = "液体类"
            else:
                categories[label] = "一般对象"
        return categories

    @property
    def object_terms(self) -> List[str]:
        return sorted(self.object_alias_to_label, key=len, reverse=True)

    @property
    def attribute_terms(self) -> List[str]:
        return sorted(self.attribute_alias_to_label, key=len, reverse=True)

    def concept_space_id(self, label: str) -> str:
        return f"concept_{_slug(label)}"

    def attribute_space_id(self, label: str) -> str:
        return f"attribute_{_slug(label)}"

    def relation_space_id(self, label: str) -> str:
        return f"relation_{_slug(label)}"

    def scene_space_id(self, label: str) -> str:
        return f"scene_{_slug(label)}"

    def normalize_object(self, surface: str) -> str | None:
        return self.object_alias_to_label.get(surface)

    def normalize_attribute(self, surface: str) -> str | None:
        return self.attribute_alias_to_label.get(surface)

    def get_scene_hint_targets(self, label: str) -> Tuple[str, ...]:
        return self.scene_hint_terms.get(label, ())

    def is_mutually_exclusive(self, left: str, right: str) -> bool:
        return (left, right) in self.mutual_exclusion or (right, left) in self.mutual_exclusion


def _tuple_map(data: Mapping[str, Sequence[str]]) -> Dict[str, Tuple[str, ...]]:
    return {key: tuple(value) for key, value in data.items()}


def load_default_ontology(config_path: str | Path | None = None) -> SBV01Ontology:
    if config_path is None:
        config_path = Path(__file__).resolve().parents[1] / "文档" / "笨鸟v0.1配置样例.json"
    path = Path(config_path)
    raw = json.loads(path.read_text(encoding="utf-8"))

    scene_spaces = {
        item["label"]: SceneSpace(
            label=item["label"],
            support_objects=tuple(item["support_objects"]),
            support_attributes=tuple(item["support_attributes"]),
            support_relations=tuple(item["support_relations"]),
            support_events=tuple(item["support_events"]),
            reject_signals=tuple(item["reject_signals"]),
            base_score=float(item["base_score"]),
        )
        for item in raw["scene_spaces"]
    }

    return SBV01Ontology(
        config_path=path,
        raw_config=raw,
        limits={key: int(value) for key, value in raw["limits"].items()},
        thresholds={key: float(value) for key, value in raw["thresholds"].items()},
        depth_decay={int(key): float(value) for key, value in raw["depth_decay"].items()},
        local_activation_weights={key: float(value) for key, value in raw["weights"]["local_activation"].items()},
        global_hypothesis_weights={
            key: float(value) for key, value in raw["weights"]["global_hypothesis"].items()
        },
        object_lexicon=_tuple_map(raw["object_lexicon"]),
        attribute_lexicon=_tuple_map(raw["attribute_lexicon"]),
        relation_patterns=list(raw["relation_patterns"]),
        event_patterns=list(raw["event_patterns"]),
        scene_spaces=scene_spaces,
        mutual_exclusion={tuple(pair) for pair in raw["mutual_exclusion"]},
        state_update_rules=list(raw["state_update_rules"]),
        scene_hint_terms={
            key: tuple(value) for key, value in raw.get("scene_hint_terms", {}).items()
        },
    )
