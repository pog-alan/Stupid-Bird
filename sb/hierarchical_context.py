from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Iterable, List, Sequence


class MemoryLevel(str, Enum):
    WORKING = "working"
    EPISODIC = "episodic"
    SUMMARY_WINDOW = "summary_window"
    SUMMARY_EPISODE = "summary_episode"
    SUMMARY_SCENE = "summary_scene"
    SEMANTIC = "semantic"


class MappingCardinality(str, Enum):
    ONE_TO_ONE = "1-1"
    ONE_TO_MANY = "1-n"
    MANY_TO_ONE = "n-1"


class MergeMode(str, Enum):
    FULL = "full_merge"
    PARTIAL = "partial_merge"
    MULTI = "multi_merge"
    SEPARATE = "separate"


class ForgetStage(str, Enum):
    KEEP = "keep"
    COOL = "cool"
    FADE = "fade"
    ARCHIVE = "archive"
    PRUNE = "prune"


@dataclass(frozen=True)
class SummaryLevelConfig:
    level: MemoryLevel
    source_levels: Sequence[MemoryLevel]
    min_support: int
    compression_ratio: float
    promote_threshold: float
    salience_threshold: float
    replay_priority: float


@dataclass(frozen=True)
class MappingRelation:
    source_ids: Sequence[str]
    target_ids: Sequence[str]
    cardinality: MappingCardinality
    coverage: float
    confidence: float


@dataclass(frozen=True)
class MergePolicy:
    similarity_threshold: float = 0.82
    temporal_bonus: float = 0.10
    task_overlap_bonus: float = 0.08
    structural_bonus: float = 0.06
    conflict_penalty: float = 0.22
    split_band: float = 0.08

    def merge_score(
        self,
        *,
        similarity: float,
        temporal_affinity: float,
        task_overlap: float,
        structural_match: float,
        conflict: float,
    ) -> float:
        return (
            similarity
            + self.temporal_bonus * temporal_affinity
            + self.task_overlap_bonus * task_overlap
            + self.structural_bonus * structural_match
            - self.conflict_penalty * conflict
        )


@dataclass(frozen=True)
class MergePlan:
    mode: MergeMode
    source_ids: Sequence[str]
    target_ids: Sequence[str]
    retained_source_ids: Sequence[str]
    coverage: float
    score: float


@dataclass(frozen=True)
class ForgettingPolicy:
    base_decay: float = 0.28
    replay_protect_gain: float = 0.10
    summary_protect_gain: float = 0.08
    semantic_floor: float = 0.15
    age_penalty: float = 0.06
    cold_threshold: float = 0.34
    fade_threshold: float = 0.22
    archive_threshold: float = 0.12
    prune_after_steps: int = 8
    cool_strength_scale: float = 0.99
    fade_strength_scale: float = 0.94
    archive_strength_scale: float = 0.82
    cool_value_scale: float = 0.998
    fade_value_scale: float = 0.985
    archive_value_scale: float = 0.94
    cool_age_boost: float = 0.01
    fade_age_boost: float = 0.03
    archive_age_boost: float = 0.06

    def retention_score(
        self,
        *,
        salience: float,
        stability: float,
        replay_hits: int,
        summarized: bool,
        age: float,
        semantic_level: bool,
    ) -> float:
        replay_boost = min(replay_hits, 6) * self.replay_protect_gain
        summary_boost = self.summary_protect_gain if summarized else 0.0
        floor = self.semantic_floor if semantic_level else 0.0
        retained = (
            self.base_decay
            + 0.45 * salience
            + 0.30 * stability
            + replay_boost
            + summary_boost
            - self.age_penalty * age
        )
        return max(floor, min(1.0, retained))


@dataclass(frozen=True)
class ForgettingStep:
    stage: ForgetStage
    strength_scale: float
    value_scale: float
    age_boost: float
    cold_step_delta: int
    route_scale: float


@dataclass(frozen=True)
class ReplayQuery:
    task_label: str
    entities: Sequence[str] = field(default_factory=list)
    relations: Sequence[str] = field(default_factory=list)
    events: Sequence[str] = field(default_factory=list)
    required_levels: Sequence[MemoryLevel] = field(default_factory=list)
    budget: int = 12


@dataclass
class MemoryNode:
    node_id: str
    level: MemoryLevel
    salience: float
    stability: float
    age: float
    replay_hits: int = 0
    summarized: bool = False
    task_labels: List[str] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)
    relations: List[str] = field(default_factory=list)
    events: List[str] = field(default_factory=list)
    summary_of: List[str] = field(default_factory=list)
    cold_steps: int = 0


@dataclass(frozen=True)
class ReplaySegment:
    level: MemoryLevel
    node_id: str
    score: float
    reason: str


@dataclass(frozen=True)
class HierarchicalContextConfig:
    summary_levels: Sequence[SummaryLevelConfig]
    merge_policy: MergePolicy = field(default_factory=MergePolicy)
    forgetting_policy: ForgettingPolicy = field(default_factory=ForgettingPolicy)
    replay_decay: float = 0.92
    coarse_budget_ratio: float = 0.35
    fine_budget_ratio: float = 0.65


class HierarchicalContextSpec:
    """Design-time spec for approximate-infinite-context memory."""

    def __init__(self, config: HierarchicalContextConfig | None = None) -> None:
        self.config = config or self.default_config()

    @staticmethod
    def default_config() -> HierarchicalContextConfig:
        return HierarchicalContextConfig(
            summary_levels=[
                SummaryLevelConfig(
                    level=MemoryLevel.SUMMARY_WINDOW,
                    source_levels=(MemoryLevel.WORKING, MemoryLevel.EPISODIC),
                    min_support=4,
                    compression_ratio=0.50,
                    promote_threshold=0.62,
                    salience_threshold=0.45,
                    replay_priority=0.70,
                ),
                SummaryLevelConfig(
                    level=MemoryLevel.SUMMARY_EPISODE,
                    source_levels=(MemoryLevel.SUMMARY_WINDOW, MemoryLevel.EPISODIC),
                    min_support=3,
                    compression_ratio=0.35,
                    promote_threshold=0.68,
                    salience_threshold=0.52,
                    replay_priority=0.82,
                ),
                SummaryLevelConfig(
                    level=MemoryLevel.SUMMARY_SCENE,
                    source_levels=(MemoryLevel.SUMMARY_EPISODE, MemoryLevel.SEMANTIC),
                    min_support=2,
                    compression_ratio=0.20,
                    promote_threshold=0.75,
                    salience_threshold=0.60,
                    replay_priority=0.95,
                ),
            ]
        )

    def promotion_equation(self) -> str:
        return (
            "promote_score = 0.40 * salience + 0.25 * closure + 0.20 * replay_need "
            "+ 0.15 * consistency - compression_cost"
        )

    def merge_equation(self) -> str:
        return (
            "merge_score = similarity + temporal_bonus * temporal_affinity + "
            "task_overlap_bonus * task_overlap + structural_bonus * structural_match "
            "- conflict_penalty * conflict"
        )

    def forgetting_equation(self) -> str:
        return (
            "retention = base_decay + 0.45 * salience + 0.30 * stability + replay_protect "
            "+ summary_protect - age_penalty * age"
        )

    def replay_equation(self) -> str:
        return (
            "replay_score = 0.35 * task_match + 0.25 * entity_match + 0.20 * relation_event_match "
            "+ 0.20 * salience + level_priority"
        )

    def build_summary_plan(self, node_counts: Dict[MemoryLevel, int]) -> List[str]:
        actions: List[str] = []
        for level_config in self.config.summary_levels:
            source_count = sum(node_counts.get(level, 0) for level in level_config.source_levels)
            if source_count >= level_config.min_support:
                actions.append(
                    f"{level_config.level.value}: promote from "
                    f"{','.join(level.value for level in level_config.source_levels)}"
                )
        return actions

    @staticmethod
    def mapping_cardinality(source_count: int, target_count: int) -> MappingCardinality:
        if source_count <= 1 and target_count <= 1:
            return MappingCardinality.ONE_TO_ONE
        if source_count <= 1:
            return MappingCardinality.ONE_TO_MANY
        return MappingCardinality.MANY_TO_ONE

    def build_mapping_relation(
        self,
        *,
        source_ids: Sequence[str],
        target_ids: Sequence[str],
        coverage: float,
        confidence: float,
    ) -> MappingRelation:
        return MappingRelation(
            source_ids=tuple(source_ids),
            target_ids=tuple(target_ids),
            cardinality=self.mapping_cardinality(len(source_ids), len(target_ids)),
            coverage=coverage,
            confidence=confidence,
        )

    def should_merge(
        self,
        *,
        similarity: float,
        temporal_affinity: float,
        task_overlap: float,
        structural_match: float,
        conflict: float,
    ) -> str:
        score = self.config.merge_policy.merge_score(
            similarity=similarity,
            temporal_affinity=temporal_affinity,
            task_overlap=task_overlap,
            structural_match=structural_match,
            conflict=conflict,
        )
        threshold = self.config.merge_policy.similarity_threshold
        if score >= threshold:
            return "merge"
        if score >= threshold - self.config.merge_policy.split_band:
            return "align"
        return "separate"

    def build_merge_plan(
        self,
        *,
        source_ids: Sequence[str],
        target_ids: Sequence[str],
        similarity: float,
        temporal_affinity: float,
        task_overlap: float,
        structural_match: float,
        conflict: float,
        coverage: float,
    ) -> MergePlan:
        score = self.config.merge_policy.merge_score(
            similarity=similarity,
            temporal_affinity=temporal_affinity,
            task_overlap=task_overlap,
            structural_match=structural_match,
            conflict=conflict,
        )
        threshold = self.config.merge_policy.similarity_threshold
        if score >= threshold + 0.06 and len(source_ids) > 1 and len(target_ids) > 1:
            mode = MergeMode.MULTI
            retained = ()
        elif score >= threshold:
            mode = MergeMode.FULL
            retained = ()
        elif score >= threshold - self.config.merge_policy.split_band:
            mode = MergeMode.PARTIAL
            retained = tuple(source_ids[: max(1, len(source_ids) // 2)])
        else:
            mode = MergeMode.SEPARATE
            retained = tuple(source_ids)
        return MergePlan(
            mode=mode,
            source_ids=tuple(source_ids),
            target_ids=tuple(target_ids),
            retained_source_ids=retained,
            coverage=coverage,
            score=score,
        )

    def forgetting_action(self, node: MemoryNode) -> str:
        return self.forgetting_step(node).stage.value

    def forgetting_step(self, node: MemoryNode) -> ForgettingStep:
        policy = self.config.forgetting_policy
        retention = policy.retention_score(
            salience=node.salience,
            stability=node.stability,
            replay_hits=node.replay_hits,
            summarized=node.summarized,
            age=node.age,
            semantic_level=node.level == MemoryLevel.SEMANTIC,
        )
        if retention > policy.cold_threshold:
            return ForgettingStep(ForgetStage.KEEP, 1.0, 1.0, 0.0, -1, 1.0)
        if retention > policy.fade_threshold:
            return ForgettingStep(
                ForgetStage.COOL,
                policy.cool_strength_scale,
                policy.cool_value_scale,
                policy.cool_age_boost,
                1,
                0.96,
            )
        if retention > policy.archive_threshold:
            return ForgettingStep(
                ForgetStage.FADE,
                policy.fade_strength_scale,
                policy.fade_value_scale,
                policy.fade_age_boost,
                1,
                0.86,
            )
        if node.cold_steps + 1 < policy.prune_after_steps:
            return ForgettingStep(
                ForgetStage.ARCHIVE,
                policy.archive_strength_scale,
                policy.archive_value_scale,
                policy.archive_age_boost,
                1,
                0.72,
            )
        return ForgettingStep(ForgetStage.PRUNE, 0.0, 0.0, 0.0, 1, 0.0)

    def build_replay_plan(
        self,
        query: ReplayQuery,
        nodes: Iterable[MemoryNode],
    ) -> List[ReplaySegment]:
        required = set(query.required_levels)
        ranked: List[ReplaySegment] = []
        for node in nodes:
            task_match = 1.0 if query.task_label in node.task_labels else 0.0
            entity_match = self._overlap(query.entities, node.entities)
            relation_event_match = max(
                self._overlap(query.relations, node.relations),
                self._overlap(query.events, node.events),
            )
            level_bonus = self._level_priority(node.level)
            if required and node.level not in required:
                level_bonus -= 0.08
            score = (
                0.35 * task_match
                + 0.25 * entity_match
                + 0.20 * relation_event_match
                + 0.20 * node.salience
                + level_bonus
            )
            if score <= 0.0:
                continue
            ranked.append(
                ReplaySegment(
                    level=node.level,
                    node_id=node.node_id,
                    score=score,
                    reason=self._replay_reason(task_match, entity_match, relation_event_match, node.level),
                )
            )
        ranked.sort(key=lambda item: item.score, reverse=True)
        return ranked[: max(query.budget, 1)]

    def default_replay_schedule(self) -> List[str]:
        return [
            "step 1: coarse replay from summary_scene and summary_episode",
            "step 2: expand matched episode summaries into summary_window",
            "step 3: reopen episodic nodes linked to top summaries",
            "step 4: inject only top-k fine nodes into the recurrent core",
        ]

    def replay_component_weights(self) -> Dict[str, float]:
        return {
            "task": 0.35,
            "entity": 0.25,
            "relation_event": 0.20,
            "salience": 0.20,
        }

    def replay_level_priority(self, level: MemoryLevel) -> float:
        return self._level_priority(level)

    @staticmethod
    def _overlap(left: Sequence[str], right: Sequence[str]) -> float:
        if not left or not right:
            return 0.0
        left_set = set(left)
        right_set = set(right)
        return len(left_set & right_set) / max(len(left_set | right_set), 1)

    def _level_priority(self, level: MemoryLevel) -> float:
        mapping = {
            MemoryLevel.WORKING: 0.02,
            MemoryLevel.EPISODIC: 0.08,
            MemoryLevel.SUMMARY_WINDOW: 0.12,
            MemoryLevel.SUMMARY_EPISODE: 0.18,
            MemoryLevel.SUMMARY_SCENE: 0.24,
            MemoryLevel.SEMANTIC: 0.10,
        }
        return mapping[level]

    @staticmethod
    def _replay_reason(
        task_match: float,
        entity_match: float,
        relation_event_match: float,
        level: MemoryLevel,
    ) -> str:
        parts: List[str] = []
        if task_match > 0.0:
            parts.append("task")
        if entity_match > 0.0:
            parts.append("entity")
        if relation_event_match > 0.0:
            parts.append("relation/event")
        parts.append(level.value)
        return "+".join(parts)
