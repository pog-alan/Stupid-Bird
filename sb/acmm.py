from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


ACMM_MEMORY_TYPES = ("episodic", "semantic", "causal", "rule", "counterexample")


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def _stable_index(text: str, dim: int) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % dim


def _norm(vec: Sequence[float]) -> float:
    return math.sqrt(sum(item * item for item in vec))


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    denom = _norm(a) * _norm(b)
    if denom <= 1e-12:
        return 0.0
    return sum(left * right for left, right in zip(a, b)) / denom


def cosine_distance(a: Sequence[float], b: Sequence[float]) -> float:
    return _clamp(1.0 - cosine_similarity(a, b))


def entropy(probabilities: Sequence[float]) -> float:
    total = sum(max(0.0, item) for item in probabilities)
    if total <= 1e-12:
        return 0.0
    normalized = [max(0.0, item) / total for item in probabilities]
    raw = -sum(item * math.log(item + 1e-12) for item in normalized)
    max_entropy = math.log(max(2, len(normalized)))
    return _clamp(raw / max_entropy)


@dataclass(frozen=True)
class ObjectState:
    object_id: str
    object_type: str
    attributes: Dict[str, float] = field(default_factory=dict)
    location: Optional[Dict[str, float]] = None
    state: str = "unknown"
    confidence: float = 1.0
    timestamp: str = ""

    def tokens(self) -> List[str]:
        items = [f"type:{self.object_type}", f"state:{self.state}"]
        items.extend(f"attr:{name}:{round(value, 3)}" for name, value in sorted(self.attributes.items()))
        return items

    def as_dict(self) -> Dict[str, object]:
        return {
            "object_id": self.object_id,
            "object_type": self.object_type,
            "attributes": dict(self.attributes),
            "location": self.location,
            "state": self.state,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
        }


@dataclass(frozen=True)
class RelationState:
    source: str
    relation: str
    target: str
    confidence: float = 1.0

    def token(self) -> str:
        return f"rel:{self.source}:{self.relation}:{self.target}"

    def as_dict(self) -> Dict[str, object]:
        return {
            "source": self.source,
            "relation": self.relation,
            "target": self.target,
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class WorldState:
    objects: Tuple[ObjectState, ...]
    relations: Tuple[RelationState, ...] = ()
    timestamp: str = ""
    label: Optional[str] = None

    def embedding(self, dim: int = 32) -> Tuple[float, ...]:
        vec = [0.0] * dim
        for obj in self.objects:
            for token in obj.tokens():
                vec[_stable_index(token, dim)] += obj.confidence
        for relation in self.relations:
            vec[_stable_index(relation.token(), dim)] += relation.confidence
        length = _norm(vec)
        if length <= 1e-12:
            return tuple(vec)
        return tuple(item / length for item in vec)

    def contains_type(self, object_type: str) -> bool:
        return any(obj.object_type == object_type for obj in self.objects)

    def contains_state(self, state: str) -> bool:
        return any(obj.state == state for obj in self.objects)

    def as_dict(self) -> Dict[str, object]:
        return {
            "objects": [item.as_dict() for item in self.objects],
            "relations": [item.as_dict() for item in self.relations],
            "timestamp": self.timestamp,
            "label": self.label,
        }


@dataclass(frozen=True)
class CausalRule:
    cause: str
    effect: str
    relation: str = "causes"
    confidence: float = 0.7
    description: str = ""

    def as_dict(self) -> Dict[str, object]:
        return {
            "cause": self.cause,
            "effect": self.effect,
            "relation": self.relation,
            "confidence": self.confidence,
            "description": self.description,
        }


@dataclass
class CausalGraph:
    rules: List[CausalRule] = field(default_factory=list)

    def predict_embedding(self, state: WorldState, dim: int = 32) -> Tuple[float, ...]:
        vec = list(state.embedding(dim))
        present = {obj.object_type for obj in state.objects} | {obj.state for obj in state.objects}
        for rule in self.rules:
            if rule.cause in present:
                vec[_stable_index(f"causal_effect:{rule.effect}", dim)] += rule.confidence
        length = _norm(vec)
        if length <= 1e-12:
            return tuple(vec)
        return tuple(item / length for item in vec)

    def violation_score(self, state: WorldState) -> float:
        present = {obj.object_type for obj in state.objects} | {obj.state for obj in state.objects}
        active = [rule for rule in self.rules if rule.cause in present]
        if not active:
            return 0.0
        missing_effect = [rule.confidence for rule in active if rule.effect not in present]
        if not missing_effect:
            return 0.0
        return _clamp(sum(missing_effect) / max(1, len(active)))

    def update_from_counterexample(self, cause: str, effect: str, penalty: float = 0.08) -> None:
        for index, rule in enumerate(self.rules):
            if rule.cause == cause and rule.effect == effect:
                self.rules[index] = CausalRule(
                    cause=rule.cause,
                    effect=rule.effect,
                    relation=rule.relation,
                    confidence=_clamp(rule.confidence - penalty),
                    description=rule.description,
                )
                return
        self.rules.append(CausalRule(cause=cause, effect=effect, confidence=_clamp(0.35 - penalty)))

    def as_dict(self) -> Dict[str, object]:
        return {"rules": [rule.as_dict() for rule in self.rules]}


@dataclass(frozen=True)
class ACMMEmotionVector:
    surprise: float
    uncertainty: float
    novelty: float
    risk: float
    value: float
    conflict: float
    curiosity: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "surprise": self.surprise,
            "uncertainty": self.uncertainty,
            "novelty": self.novelty,
            "risk": self.risk,
            "value": self.value,
            "conflict": self.conflict,
            "curiosity": self.curiosity,
        }


@dataclass(frozen=True)
class GateDecision:
    write_memory: float
    update_model: float
    request_review: float
    update_rule: float
    trigger_alert: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "write_memory": self.write_memory,
            "update_model": self.update_model,
            "request_review": self.request_review,
            "update_rule": self.update_rule,
            "trigger_alert": self.trigger_alert,
        }


@dataclass(frozen=True)
class MemoryItem:
    memory_id: str
    state_embedding: Tuple[float, ...]
    object_state: WorldState
    label: Optional[str]
    emotion: ACMMEmotionVector
    memory_type: str
    timestamp: str = ""

    def as_dict(self) -> Dict[str, object]:
        return {
            "memory_id": self.memory_id,
            "state_embedding": list(self.state_embedding),
            "object_state": self.object_state.as_dict(),
            "label": self.label,
            "emotion": self.emotion.as_dict(),
            "memory_type": self.memory_type,
            "timestamp": self.timestamp,
        }


@dataclass(frozen=True)
class RetrievedMemory:
    item: MemoryItem
    similarity: float

    def as_dict(self) -> Dict[str, object]:
        payload = self.item.as_dict()
        payload["similarity"] = self.similarity
        return payload


@dataclass
class LayeredMemoryStore:
    items: List[MemoryItem] = field(default_factory=list)

    def retrieve(self, state: WorldState, *, top_k: int = 3, dim: int = 32) -> List[RetrievedMemory]:
        embedding = state.embedding(dim)
        matches = [
            RetrievedMemory(item=item, similarity=round(cosine_similarity(embedding, item.state_embedding), 4))
            for item in self.items
        ]
        matches.sort(key=lambda item: item.similarity, reverse=True)
        return matches[:top_k]

    def write(self, state: WorldState, label: Optional[str], emotion: ACMMEmotionVector, memory_type: str) -> MemoryItem:
        if memory_type not in ACMM_MEMORY_TYPES:
            raise ValueError(f"未知记忆类型：{memory_type}")
        memory_id = f"{memory_type}:{len(self.items) + 1}"
        item = MemoryItem(
            memory_id=memory_id,
            state_embedding=state.embedding(),
            object_state=state,
            label=label,
            emotion=emotion,
            memory_type=memory_type,
            timestamp=state.timestamp,
        )
        self.items.append(item)
        return item

    def as_dict(self) -> Dict[str, object]:
        counts = {name: 0 for name in ACMM_MEMORY_TYPES}
        for item in self.items:
            counts[item.memory_type] = counts.get(item.memory_type, 0) + 1
        return {"count": len(self.items), "by_type": counts}


@dataclass(frozen=True)
class ACMMConfig:
    embedding_dim: int = 32
    retrieve_top_k: int = 3
    write_threshold: float = 0.70
    review_threshold: float = 0.80
    rule_update_threshold: float = 0.75
    alert_threshold: float = 0.78
    risk_keywords: Tuple[str, ...] = (
        "污染",
        "违法",
        "采掘",
        "裸地",
        "发黑",
        "积水",
        "泄漏",
        "异常",
        "风险",
    )


@dataclass(frozen=True)
class ACMMStepResult:
    state: WorldState
    predicted_embedding: Tuple[float, ...]
    label_probabilities: Dict[str, float]
    recalled: Tuple[RetrievedMemory, ...]
    emotion: ACMMEmotionVector
    gates: GateDecision
    memory_writes: Tuple[MemoryItem, ...]
    action_plan: Tuple[str, ...]

    def as_dict(self) -> Dict[str, object]:
        return {
            "state": self.state.as_dict(),
            "predicted_embedding": list(self.predicted_embedding),
            "label_probabilities": dict(self.label_probabilities),
            "recalled": [item.as_dict() for item in self.recalled],
            "emotion": self.emotion.as_dict(),
            "gates": self.gates.as_dict(),
            "memory_writes": [item.as_dict() for item in self.memory_writes],
            "action_plan": list(self.action_plan),
        }


class AffectiveCausalMemoryModel:
    """情绪门控因果记忆模型。

    ACMM 把“情绪”定义为可计算监督量，而不是拟人化状态。它先把输入对象化，
    再用因果图预测合理状态，用情绪向量评价学习价值，最后由门控决定记忆写入、
    模型更新、规则修正和人工复核。
    """

    def __init__(
        self,
        *,
        memory: LayeredMemoryStore | None = None,
        causal_graph: CausalGraph | None = None,
        config: ACMMConfig | None = None,
    ) -> None:
        self.memory = memory or LayeredMemoryStore()
        self.causal_graph = causal_graph or CausalGraph()
        self.config = config or ACMMConfig()

    def cognitive_step(self, observation: Mapping[str, object]) -> ACMMStepResult:
        state = self.objectify(observation)
        recalled = tuple(self.memory.retrieve(state, top_k=self.config.retrieve_top_k, dim=self.config.embedding_dim))
        predicted = self._predicted_embedding(observation, state)
        label_probabilities = self._label_probabilities(observation)
        emotion = self.compute_emotion(
            state=state,
            predicted_embedding=predicted,
            label_probabilities=label_probabilities,
            recalled=recalled,
            risk_score=_optional_float(observation.get("risk_score")),
            task_value=_optional_float(observation.get("task_value")),
            rule_violation_score=_optional_float(observation.get("rule_violation_score")),
        )
        gates = compute_gates(emotion)
        writes = tuple(self._write_memories(state, label_probabilities, emotion, gates))
        action_plan = tuple(self._action_plan(gates))
        return ACMMStepResult(
            state=state,
            predicted_embedding=predicted,
            label_probabilities=label_probabilities,
            recalled=recalled,
            emotion=emotion,
            gates=gates,
            memory_writes=writes,
            action_plan=action_plan,
        )

    def objectify(self, observation: Mapping[str, object]) -> WorldState:
        objects = tuple(_parse_object(item) for item in _as_sequence(observation.get("objects", ())))
        relations = tuple(_parse_relation(item) for item in _as_sequence(observation.get("relations", ())))
        return WorldState(
            objects=objects,
            relations=relations,
            timestamp=str(observation.get("timestamp", "")),
            label=_optional_str(observation.get("label")),
        )

    def compute_emotion(
        self,
        *,
        state: WorldState,
        predicted_embedding: Sequence[float],
        label_probabilities: Mapping[str, float],
        recalled: Sequence[RetrievedMemory],
        risk_score: float | None = None,
        task_value: float | None = None,
        rule_violation_score: float | None = None,
    ) -> ACMMEmotionVector:
        state_embedding = state.embedding(self.config.embedding_dim)
        max_similarity = max((item.similarity for item in recalled), default=0.0)
        risk = _clamp(risk_score if risk_score is not None else self._risk_score(state))
        value = _clamp(task_value if task_value is not None else (0.55 + 0.35 * risk))
        conflict = _clamp(
            rule_violation_score if rule_violation_score is not None else self.causal_graph.violation_score(state)
        )
        surprise = cosine_distance(state_embedding, predicted_embedding)
        uncertainty = entropy(list(label_probabilities.values()))
        novelty = _clamp(1.0 - max_similarity)
        curiosity = _clamp(uncertainty * novelty * value)
        return ACMMEmotionVector(
            surprise=round(surprise, 4),
            uncertainty=round(uncertainty, 4),
            novelty=round(novelty, 4),
            risk=round(risk, 4),
            value=round(value, 4),
            conflict=round(conflict, 4),
            curiosity=round(curiosity, 4),
        )

    def _predicted_embedding(self, observation: Mapping[str, object], state: WorldState) -> Tuple[float, ...]:
        raw = observation.get("predicted_embedding")
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
            values = tuple(float(item) for item in raw)
            if len(values) == self.config.embedding_dim:
                return values
        return self.causal_graph.predict_embedding(state, dim=self.config.embedding_dim)

    def _label_probabilities(self, observation: Mapping[str, object]) -> Dict[str, float]:
        raw = observation.get("label_probabilities")
        if isinstance(raw, Mapping) and raw:
            values = {str(key): max(0.0, float(value)) for key, value in raw.items()}
            total = sum(values.values())
            if total > 1e-12:
                return {key: value / total for key, value in values.items()}
        return {"normal": 0.34, "risk": 0.33, "unknown": 0.33}

    def _risk_score(self, state: WorldState) -> float:
        text = " ".join(
            [obj.object_type for obj in state.objects]
            + [obj.state for obj in state.objects]
            + [relation.relation for relation in state.relations]
        )
        hits = sum(1 for keyword in self.config.risk_keywords if keyword in text)
        return _clamp(hits / 3.0)

    def _write_memories(
        self,
        state: WorldState,
        label_probabilities: Mapping[str, float],
        emotion: ACMMEmotionVector,
        gates: GateDecision,
    ) -> Iterable[MemoryItem]:
        if gates.write_memory < self.config.write_threshold:
            return []
        label = max(label_probabilities.items(), key=lambda item: item[1])[0] if label_probabilities else None
        memory_types = self._memory_types_for_emotion(emotion)
        return [self.memory.write(state, label, emotion, memory_type) for memory_type in memory_types]

    def _memory_types_for_emotion(self, emotion: ACMMEmotionVector) -> Tuple[str, ...]:
        selected: List[str] = []
        if emotion.novelty >= 0.45 or emotion.surprise >= 0.45:
            selected.append("episodic")
        if emotion.conflict >= 0.40:
            selected.append("counterexample")
        if emotion.risk >= 0.55:
            selected.append("rule")
        if emotion.surprise >= 0.50 or emotion.conflict >= 0.50:
            selected.append("causal")
        if emotion.value >= 0.70 and emotion.uncertainty <= 0.45:
            selected.append("semantic")
        if not selected:
            selected.append("episodic")
        return tuple(dict.fromkeys(selected))

    def _action_plan(self, gates: GateDecision) -> List[str]:
        actions: List[str] = []
        if gates.trigger_alert >= self.config.alert_threshold:
            actions.append("trigger_risk_alert")
        if gates.request_review >= self.config.review_threshold:
            actions.append("request_human_review")
        if gates.update_rule >= self.config.rule_update_threshold:
            actions.append("update_causal_or_rule_memory")
        if gates.write_memory >= self.config.write_threshold:
            actions.append("write_layered_memory")
        if gates.update_model >= 0.65:
            actions.append("increase_training_weight")
        if not actions:
            actions.append("continue_regular_inference")
        return actions


def compute_gates(e: ACMMEmotionVector) -> GateDecision:
    write_score = 1.2 * e.surprise + 1.0 * e.novelty + 0.8 * e.value + 0.6 * e.conflict
    review_score = 1.5 * e.uncertainty + 1.2 * e.risk + 1.0 * e.conflict
    update_score = 1.0 * e.surprise + 1.0 * e.conflict + 0.8 * e.curiosity
    rule_score = 1.4 * e.conflict + 1.0 * e.surprise + 0.7 * e.risk
    alert_score = 1.5 * e.risk + 1.2 * e.conflict
    return GateDecision(
        write_memory=round(_sigmoid(write_score - 1.5), 4),
        update_model=round(_sigmoid(update_score - 1.5), 4),
        request_review=round(_sigmoid(review_score - 1.5), 4),
        update_rule=round(_sigmoid(rule_score - 1.6), 4),
        trigger_alert=round(_sigmoid(alert_score - 1.8), 4),
    )


def acmm_loss_spec() -> Dict[str, str]:
    return {
        "task_loss": "L_task = CE(y_t, y_hat_t) 或分割/检测任务损失。",
        "prediction_loss": "L_pred = d(z_{t+1}, z_hat_{t+1})，监督因果时序预测。",
        "causal_loss": "L_causal = sum_k violation_k(z_t, G_t)，惩罚违反因果图的状态解释。",
        "memory_loss": "L_memory = d(z_t, Retrieve(M_t, z_t))，约束记忆召回与当前判断一致。",
        "rule_loss": "L_rule = sum max(0, rule_violation_k)，约束规则空间稳定性。",
        "feedback_loss": "L_feedback = CE(y_human, y_hat_t) 或人工复核偏好损失。",
        "emotion_gate_update": "theta_{t+1} = theta_t - eta * g(e_t) * grad_theta L。",
        "total": "L = L_task + lambda_1 L_pred + lambda_2 L_causal + lambda_3 L_memory + lambda_4 L_rule + lambda_5 L_feedback。",
    }


def acmm_model_spec() -> Dict[str, object]:
    return {
        "name": "Affective-Causal Memory Model",
        "short_name": "ACMM",
        "state_loop": "x_t -> O_t -> z_t -> G_t -> e_t -> M_{t+1}",
        "emotion_vector": ["Surprise", "Uncertainty", "Novelty", "Risk", "Value", "Conflict", "Curiosity"],
        "memory_types": list(ACMM_MEMORY_TYPES),
        "gates": ["write_memory", "update_model", "request_review", "update_rule", "trigger_alert"],
        "loss": acmm_loss_spec(),
    }


def _parse_object(raw: object) -> ObjectState:
    if isinstance(raw, ObjectState):
        return raw
    if not isinstance(raw, Mapping):
        text = str(raw)
        return ObjectState(object_id=text, object_type=text)
    object_id = str(raw.get("object_id", raw.get("id", raw.get("name", raw.get("type", "object")))))
    object_type = str(raw.get("object_type", raw.get("type", raw.get("name", object_id))))
    attributes = {
        str(key): float(value)
        for key, value in dict(raw.get("attributes", raw.get("attr", {})) or {}).items()
        if _is_number(value)
    }
    return ObjectState(
        object_id=object_id,
        object_type=object_type,
        attributes=attributes,
        location=raw.get("location") if isinstance(raw.get("location"), dict) else None,
        state=str(raw.get("state", "unknown")),
        confidence=float(raw.get("confidence", 1.0) or 1.0),
        timestamp=str(raw.get("timestamp", "")),
    )


def _parse_relation(raw: object) -> RelationState:
    if isinstance(raw, RelationState):
        return raw
    if not isinstance(raw, Mapping):
        return RelationState(source="", relation=str(raw), target="")
    return RelationState(
        source=str(raw.get("source", raw.get("src", ""))),
        relation=str(raw.get("relation", raw.get("type", ""))),
        target=str(raw.get("target", raw.get("dst", ""))),
        confidence=float(raw.get("confidence", 1.0) or 1.0),
    )


def _as_sequence(value: object) -> Sequence[object]:
    if value is None:
        return ()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return value
    return (value,)


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    if _is_number(value):
        return float(value)
    return None


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)
