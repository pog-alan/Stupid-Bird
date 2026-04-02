from __future__ import annotations

from dataclasses import dataclass, field, replace
from math import sqrt
from typing import Dict, Iterable, List, Sequence, Tuple


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def _cosine_similarity(lhs: Sequence[float], rhs: Sequence[float]) -> float:
    if not lhs or not rhs or len(lhs) != len(rhs):
        return 0.0
    numerator = sum(a * b for a, b in zip(lhs, rhs))
    lhs_norm = sqrt(sum(a * a for a in lhs))
    rhs_norm = sqrt(sum(b * b for b in rhs))
    if lhs_norm == 0.0 or rhs_norm == 0.0:
        return 0.0
    return numerator / (lhs_norm * rhs_norm)


def _overlap_score(lhs: Iterable[str], rhs: Iterable[str]) -> float:
    left = set(lhs)
    right = set(rhs)
    if not left or not right:
        return 0.0
    shared = len(left & right)
    return shared / max(len(left), len(right))


def _sorted_union(*groups: Iterable[str]) -> Tuple[str, ...]:
    union = set()
    for group in groups:
        union.update(group)
    return tuple(sorted(union))


@dataclass(frozen=True)
class Signal:
    signal_id: str
    kind: str
    value: object
    tags: Tuple[str, ...] = ()
    embedding: Tuple[float, ...] = ()
    energy: float = 1.0
    confidence: float = 1.0
    ttl: int = 3
    provenance: Tuple[str, ...] = ()

    def signature(self) -> Tuple[object, ...]:
        return (self.kind, self.value, self.tags, self.embedding)


@dataclass(frozen=True)
class SignalTemplate:
    kind: str
    value: object
    tags: Tuple[str, ...] = ()
    embedding: Tuple[float, ...] = ()
    confidence_gain: float = 0.0
    energy_scale: float = 0.85


@dataclass
class TransformRule:
    required_tags: frozenset[str]
    emitted: Tuple[SignalTemplate, ...]
    weight: float = 1.0

    def matches(self, signal: Signal, workspace_tags: frozenset[str]) -> bool:
        available = set(signal.tags) | set(workspace_tags)
        return self.required_tags.issubset(available)


@dataclass
class Space:
    space_id: str
    space_type: str
    sensitive_tags: frozenset[str]
    preferred_kinds: frozenset[str] = frozenset()
    activation_bias: float = 0.0
    transform_rules: List[TransformRule] = field(default_factory=list)
    connections: Dict[str, float] = field(default_factory=dict)
    usage_count: int = 0
    accumulated_reward: float = 0.0

    def match_score(self, signal: Signal, workspace_tags: frozenset[str]) -> float:
        tag_score = _overlap_score(signal.tags, self.sensitive_tags)
        context_score = _overlap_score(workspace_tags, self.sensitive_tags)
        kind_score = 1.0 if not self.preferred_kinds or signal.kind in self.preferred_kinds else 0.0
        raw_score = (
            0.55 * tag_score
            + 0.20 * context_score
            + 0.15 * kind_score
            + 0.10 * self.activation_bias
        )
        return _clamp(raw_score)

    def emit(self, signal: Signal, workspace_tags: frozenset[str], seed: int) -> List[Signal]:
        emitted: List[Signal] = []
        for rule in self.transform_rules:
            if not rule.matches(signal, workspace_tags):
                continue
            for index, template in enumerate(rule.emitted):
                emitted.append(
                    Signal(
                        signal_id=f"{self.space_id}:{seed}:{index}",
                        kind=template.kind,
                        value=template.value,
                        tags=_sorted_union(signal.tags, template.tags),
                        embedding=template.embedding,
                        energy=signal.energy * template.energy_scale * rule.weight,
                        confidence=_clamp(signal.confidence + template.confidence_gain),
                        ttl=max(signal.ttl - 1, 0),
                        provenance=signal.provenance + (self.space_id,),
                    )
                )
        self.usage_count += 1
        return emitted


@dataclass
class Hypothesis:
    score: float
    signals: Tuple[Signal, ...]
    trace: Tuple[str, ...] = ()
    space_history: Tuple[str, ...] = ()


class SBNetwork:
    def __init__(self, spaces: Iterable[Space] | None = None) -> None:
        self.spaces: Dict[str, Space] = {}
        self.tag_index: Dict[str, set[str]] = {}
        self._seed = 0
        for space in spaces or ():
            self.add_space(space)

    def add_space(self, space: Space) -> None:
        self.spaces[space.space_id] = space
        self._rebuild_index()

    def _rebuild_index(self) -> None:
        self.tag_index = {}
        for space in self.spaces.values():
            for tag in space.sensitive_tags:
                self.tag_index.setdefault(tag, set()).add(space.space_id)

    def route(
        self,
        signal: Signal,
        workspace_tags: frozenset[str],
        history: Sequence[str] = (),
        top_k: int = 4,
        threshold: float = 0.35,
    ) -> List[Tuple[Space, float]]:
        candidate_ids = set()
        for tag in signal.tags:
            candidate_ids.update(self.tag_index.get(tag, set()))
        if not candidate_ids:
            candidate_ids = set(self.spaces.keys())

        ranked: List[Tuple[Space, float]] = []
        for space_id in candidate_ids:
            space = self.spaces[space_id]
            score = _clamp(space.match_score(signal, workspace_tags) + self._transition_bonus(space, history))
            if score >= threshold:
                ranked.append((space, score))

        ranked.sort(key=lambda item: item[1], reverse=True)
        return ranked[:top_k]

    def correlation(self, signal: Signal, workspace: Sequence[Signal]) -> float:
        if not workspace:
            return signal.confidence
        best = 0.0
        for other in workspace:
            tag_score = _overlap_score(signal.tags, other.tags)
            embedding_score = _cosine_similarity(signal.embedding, other.embedding)
            kind_bonus = 0.1 if signal.kind == other.kind else 0.0
            best = max(best, _clamp(0.6 * tag_score + 0.3 * embedding_score + kind_bonus))
        return best

    def _transition_bonus(self, candidate_space: Space, history: Sequence[str]) -> float:
        if not history:
            return 0.0

        bonus = 0.0
        for offset, source_id in enumerate(reversed(history[-2:]), start=1):
            source_space = self.spaces.get(source_id)
            if source_space is None:
                continue
            weight = source_space.connections.get(candidate_space.space_id, 0.0)
            bonus = max(bonus, weight / offset)
        return _clamp(bonus)

    def infer(
        self,
        inputs: Sequence[Signal],
        steps: int = 3,
        beam_width: int = 8,
        top_k: int = 4,
        activation_threshold: float = 0.35,
        correlation_threshold: float = 0.30,
        max_expansions: int | None = 128,
        max_signals_per_beam: int = 12,
    ) -> Hypothesis:
        base_score = sum(signal.confidence for signal in inputs) / max(len(inputs), 1)
        beams = [Hypothesis(score=base_score, signals=tuple(inputs), trace=(), space_history=())]
        cache: Dict[Tuple[object, ...], Tuple[Signal, ...]] = {}
        expansions = 0

        for _ in range(steps):
            next_beams: List[Hypothesis] = []
            budget_exhausted = False
            for beam in beams:
                workspace_tags = frozenset(tag for signal in beam.signals for tag in signal.tags)
                for signal in beam.signals:
                    if signal.ttl <= 0 or signal.energy <= 0.0:
                        continue
                    for space, route_score in self.route(
                        signal,
                        workspace_tags,
                        history=beam.space_history,
                        top_k=top_k,
                        threshold=activation_threshold,
                    ):
                        cache_key = (space.space_id, signal.signature(), tuple(sorted(workspace_tags)))
                        emitted = cache.get(cache_key)
                        if emitted is None:
                            self._seed += 1
                            emitted = tuple(space.emit(signal, workspace_tags, self._seed))
                            cache[cache_key] = emitted

                        for child in emitted:
                            expansions += 1
                            if max_expansions is not None and expansions > max_expansions:
                                budget_exhausted = True
                                break
                            correlation_score = self.correlation(child, beam.signals)
                            if correlation_score < correlation_threshold:
                                continue
                            candidate = replace(
                                child,
                                confidence=_clamp(child.confidence * (0.7 + 0.3 * correlation_score)),
                            )
                            merged, changed = self._merge_signals(beam.signals, candidate)
                            if not changed:
                                continue
                            next_beams.append(
                                Hypothesis(
                                    score=beam.score + route_score + correlation_score + candidate.confidence,
                                    signals=merged,
                                    trace=beam.trace + (f"{signal.signal_id}->{space.space_id}",),
                                    space_history=beam.space_history + (space.space_id,),
                                )
                            )
                        if budget_exhausted:
                            break
                    if budget_exhausted:
                        break
                if budget_exhausted:
                    break
            if not next_beams:
                break
            for index, hypothesis in enumerate(next_beams):
                next_beams[index] = replace(
                    hypothesis,
                    signals=self._trim_signals(hypothesis.signals, max_signals_per_beam),
                )
            next_beams.sort(key=lambda item: item.score, reverse=True)
            beams = next_beams[:beam_width]
            if budget_exhausted:
                break

        beams.sort(key=lambda item: item.score, reverse=True)
        return beams[0]

    def learn_concept(
        self,
        episode: Sequence[Signal],
        concept_name: str,
        output_tags: Iterable[str] = (),
        novelty_threshold: float = 0.65,
    ) -> Space:
        workspace_tags = frozenset(tag for signal in episode for tag in signal.tags)
        preferred_kinds = frozenset(signal.kind for signal in episode)

        best_space = None
        best_score = 0.0
        for space in self.spaces.values():
            representative = max(
                (space.match_score(signal, workspace_tags) for signal in episode),
                default=0.0,
            )
            if representative > best_score:
                best_space = space
                best_score = representative

        if best_space is not None and best_score >= novelty_threshold:
            best_space.sensitive_tags = frozenset(set(best_space.sensitive_tags) | set(workspace_tags))
            best_space.accumulated_reward += 1.0
            self._rebuild_index()
            return best_space

        new_space = Space(
            space_id=f"space_{len(self.spaces) + 1}_{concept_name}",
            space_type="concept",
            sensitive_tags=workspace_tags,
            preferred_kinds=preferred_kinds,
            transform_rules=[
                TransformRule(
                    required_tags=frozenset(),
                    emitted=(
                        SignalTemplate(
                            kind="concept",
                            value=concept_name,
                            tags=tuple(sorted(set(output_tags) | set(workspace_tags))),
                            confidence_gain=0.1,
                        ),
                    ),
                )
            ],
        )
        self.add_space(new_space)
        return new_space

    def reinforce_transition(self, source_space: str, target_space: str, reward: float = 0.1) -> None:
        if source_space not in self.spaces or target_space not in self.spaces:
            return
        space = self.spaces[source_space]
        space.connections[target_space] = space.connections.get(target_space, 0.0) + reward
        space.accumulated_reward += reward

    def merge_similar_spaces(self, threshold: float = 0.8) -> List[Tuple[str, str]]:
        merged: List[Tuple[str, str]] = []
        ids = list(self.spaces.keys())
        retired: set[str] = set()

        for index, left_id in enumerate(ids):
            if left_id in retired:
                continue
            for right_id in ids[index + 1 :]:
                if right_id in retired:
                    continue
                left = self.spaces[left_id]
                right = self.spaces[right_id]
                similarity = _overlap_score(left.sensitive_tags, right.sensitive_tags)
                if similarity < threshold:
                    continue
                left.sensitive_tags = frozenset(set(left.sensitive_tags) | set(right.sensitive_tags))
                left.preferred_kinds = frozenset(set(left.preferred_kinds) | set(right.preferred_kinds))
                left.transform_rules.extend(right.transform_rules)
                left.connections.update(right.connections)
                retired.add(right_id)
                merged.append((left_id, right_id))

        for retired_id in retired:
            self.spaces.pop(retired_id, None)
        if retired:
            self._rebuild_index()
        return merged

    def prune_spaces(
        self,
        min_usage: int = 1,
        min_reward: float = 0.0,
        preserve: Iterable[str] = (),
    ) -> List[str]:
        preserve_ids = set(preserve)
        removed: List[str] = []

        for space_id, space in list(self.spaces.items()):
            if space_id in preserve_ids:
                continue
            if space.usage_count >= min_usage:
                continue
            if space.accumulated_reward > min_reward:
                continue
            removed.append(space_id)
            self.spaces.pop(space_id, None)

        if removed:
            for space in self.spaces.values():
                for removed_id in removed:
                    space.connections.pop(removed_id, None)
            self._rebuild_index()
        return removed

    @staticmethod
    def _merge_signals(existing: Sequence[Signal], candidate: Signal) -> Tuple[Tuple[Signal, ...], bool]:
        unique: Dict[Tuple[object, ...], Signal] = {signal.signature(): signal for signal in existing}
        if candidate.signature() not in unique:
            unique[candidate.signature()] = candidate
            return tuple(unique.values()), True
        return tuple(unique.values()), False

    @staticmethod
    def _trim_signals(signals: Sequence[Signal], max_signals: int) -> Tuple[Signal, ...]:
        if len(signals) <= max_signals:
            return tuple(signals)
        ranked = sorted(signals, key=lambda signal: (signal.confidence, signal.energy, signal.ttl), reverse=True)
        return tuple(ranked[:max_signals])
