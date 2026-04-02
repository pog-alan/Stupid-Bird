from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class Segment:
    segment_id: str
    text: str
    start: int
    end: int


@dataclass
class ExtractedSignal:
    signal_id: str
    signal_type: str
    surface: str
    normalized: str
    category: str
    segment_id: str
    span: Tuple[int, int]
    confidence: float
    anchors: Tuple[str, ...] = ()
    attributes: Dict[str, Any] = field(default_factory=dict)
    provenance: Tuple[str, ...] = ()


@dataclass
class ObjectNode:
    object_id: str
    label: str
    normalized: str
    category: str
    segment_id: str
    span: Tuple[int, int]
    confidence: float
    source: str = "lexicon"


@dataclass
class AttributeEdge:
    attribute_id: str
    label: str
    normalized: str
    target_id: Optional[str]
    target_label: Optional[str]
    confidence: float
    segment_id: str
    span: Tuple[int, int]
    derived: bool = False


@dataclass
class RelationEdge:
    relation_id: str
    label: str
    normalized: str
    source_id: str
    source_label: str
    target_id: str
    target_label: str
    confidence: float
    segment_id: str
    span: Tuple[int, int]
    derived: bool = False


@dataclass
class EventNode:
    event_id: str
    event_type: str
    actor_id: Optional[str]
    actor_label: Optional[str]
    target_id: Optional[str]
    target_label: Optional[str]
    location_id: Optional[str]
    location_label: Optional[str]
    target_ids: Tuple[str, ...] = ()
    target_labels: Tuple[str, ...] = ()
    confidence: float = 0.0
    segment_id: str = ""
    span: Tuple[int, int] = (0, 0)
    provenance: Tuple[str, ...] = ()


@dataclass
class SceneHint:
    hint_id: str
    label: str
    confidence: float
    segment_id: str
    span: Tuple[int, int]


@dataclass
class ParsedScene:
    input_text: str
    normalized_text: str
    segments: List[Segment]
    signals: List[ExtractedSignal]
    objects: List[ObjectNode]
    attributes: List[AttributeEdge]
    relations: List[RelationEdge]
    events: List[EventNode]
    scene_hints: List[SceneHint]


@dataclass
class StateInference:
    target_id: str
    target_label: str
    state: str
    source_event: Optional[str] = None


@dataclass
class ActivationHit:
    signal_id: str
    signal_label: str
    signal_type: str
    space_id: str
    space_type: str
    space_label: str
    score: float
    supports: Tuple[str, ...] = ()


@dataclass
class ReasoningStep:
    step: int
    signal: str
    activated_spaces: Tuple[str, ...]
    contribution: str = ""


@dataclass
class SceneHypothesis:
    label: str
    score: float
    object_support: float
    attribute_consistency: float
    relation_consistency: float
    scene_compatibility: float
    event_plausibility: float
    conflict_penalty: float
    mutual_exclusion_penalty: float
    complexity_penalty: float
    matched_objects: Tuple[str, ...] = ()
    matched_attributes: Tuple[str, ...] = ()
    matched_relations: Tuple[str, ...] = ()
    matched_events: Tuple[str, ...] = ()


@dataclass
class AnalysisResult:
    input_text: str
    objects: List[ObjectNode]
    attributes: List[AttributeEdge]
    relations: List[RelationEdge]
    events: List[EventNode]
    scene_hypotheses: List[SceneHypothesis]
    best_hypothesis: Optional[SceneHypothesis]
    reasoning_path: List[ReasoningStep]
    state_inference: List[StateInference]
