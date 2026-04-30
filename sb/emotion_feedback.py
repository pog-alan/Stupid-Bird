from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Tuple


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


@dataclass(frozen=True)
class EmotionFeedbackConfig:
    """机器情绪反馈配置。

    这里的“情绪”不是主观体验，而是可解释的控制信号：它把不确定性、
    新颖性、冲突、风险和运行压力压缩成一组反馈量，用来调节提问、学习、
    记忆合并和输出风格。
    """

    confident_gap: float = 0.18
    ambiguous_gap: float = 0.06
    high_confusion: float = 0.55
    high_curiosity: float = 0.52
    high_caution: float = 0.50
    high_urgency: float = 0.55
    high_fatigue: float = 0.60
    high_confidence: float = 0.72
    risk_keywords: Tuple[str, ...] = (
        "污染",
        "泄漏",
        "发黑",
        "异常",
        "危险",
        "翻倒",
        "撞击",
        "散落",
        "腐烂",
        "积水",
        "垃圾",
    )


@dataclass(frozen=True)
class EmotionSupervisionConfig:
    """情绪监督配置。

    情绪监督把反馈向量从“运行日志”升级为训练、推理、预测共同使用的
    监督对象。它可以来自规则标注、人类偏好、任务结果或模型自评。
    """

    emotion_vector_loss_weight: float = 0.15
    action_policy_loss_weight: float = 0.10
    next_emotion_loss_weight: float = 0.08
    risk_prediction_loss_weight: float = 0.12
    calibration_loss_weight: float = 0.10
    confidence_gate_floor: float = 0.20
    caution_gate_ceiling: float = 0.95
    curiosity_write_gain: float = 0.25
    fatigue_summary_gain: float = 0.45


@dataclass(frozen=True)
class EmotionVector:
    confidence: float
    curiosity: float
    caution: float
    urgency: float
    fatigue: float
    satisfaction: float
    confusion: float
    risk: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "confidence": self.confidence,
            "curiosity": self.curiosity,
            "caution": self.caution,
            "urgency": self.urgency,
            "fatigue": self.fatigue,
            "satisfaction": self.satisfaction,
            "confusion": self.confusion,
            "risk": self.risk,
        }


@dataclass(frozen=True)
class EmotionAction:
    name: str
    reason: str
    intensity: float

    def as_dict(self) -> Dict[str, object]:
        return {"name": self.name, "reason": self.reason, "intensity": self.intensity}


@dataclass(frozen=True)
class EmotionFeedback:
    vector: EmotionVector
    dominant_state: str
    actions: Tuple[EmotionAction, ...]
    learning_policy: Dict[str, float]
    explanation: Tuple[str, ...]

    def as_dict(self) -> Dict[str, object]:
        return {
            "vector": self.vector.as_dict(),
            "dominant_state": self.dominant_state,
            "actions": [action.as_dict() for action in self.actions],
            "learning_policy": dict(self.learning_policy),
            "explanation": list(self.explanation),
        }


@dataclass(frozen=True)
class EmotionSupervisionTarget:
    vector_target: EmotionVector
    action_targets: Tuple[str, ...]
    inference_gates: Dict[str, float]
    loss_weights: Dict[str, float]
    prediction_targets: Dict[str, float]
    label_source: str

    def as_dict(self) -> Dict[str, object]:
        return {
            "vector_target": self.vector_target.as_dict(),
            "action_targets": list(self.action_targets),
            "inference_gates": dict(self.inference_gates),
            "loss_weights": dict(self.loss_weights),
            "prediction_targets": dict(self.prediction_targets),
            "label_source": self.label_source,
        }


class MachineEmotionFeedbackEngine:
    """把 SB 的解释状态转成机器情绪反馈。

    反馈向量只作为算法控制量使用：
    - confidence 控制回答直接程度和自动提交程度
    - curiosity 控制新概念学习、主动探索和提问
    - caution 控制自动学习降权、冲突复核和保守输出
    - urgency 控制风险相关场景的优先级
    - fatigue 控制摘要、合并和遗忘
    """

    def __init__(self, config: EmotionFeedbackConfig | None = None) -> None:
        self.config = config or EmotionFeedbackConfig()

    def evaluate(
        self,
        result: Mapping[str, object] | object,
        *,
        runtime: Mapping[str, object] | None = None,
    ) -> EmotionFeedback:
        scene_hypotheses = list(_field(result, "scene_hypotheses", []))
        objects = list(_field(result, "objects", []))
        attributes = list(_field(result, "attributes", []))
        relations = list(_field(result, "relations", []))
        events = list(_field(result, "events", []))
        conflicts = list(_field(result, "conflicts", []))
        temporary = list(_field(result, "temporary_concepts", []))
        candidates = list(_field(result, "candidate_concepts", []))

        top_score, gap = self._hypothesis_scores(scene_hypotheses)
        completeness = self._evidence_completeness(objects, attributes, relations, events)
        missing_ratio = 1.0 - completeness
        ambiguity = 1.0 if 0.0 <= gap < self.config.ambiguous_gap else _clamp(1.0 - gap / self.config.confident_gap)
        conflict_pressure = _clamp(len(conflicts) / 3.0)
        novelty = _clamp((len(temporary) + 0.5 * len(candidates)) / 4.0)
        risk = self._risk_score(result)
        fatigue = self._fatigue_score(runtime or {})

        confidence = _clamp(0.52 * top_score + 0.28 * (gap / self.config.confident_gap) + 0.20 * completeness)
        confusion = _clamp(0.42 * missing_ratio + 0.34 * ambiguity + 0.24 * conflict_pressure)
        curiosity = _clamp(0.44 * novelty + 0.32 * missing_ratio + 0.24 * (1.0 - confidence))
        caution = _clamp(0.36 * conflict_pressure + 0.30 * risk + 0.22 * (1.0 - confidence) + 0.12 * ambiguity)
        urgency = _clamp(0.58 * risk + 0.22 * _presence(events) + 0.20 * caution)
        satisfaction = _clamp(0.72 * confidence + 0.28 * completeness - 0.20 * conflict_pressure)

        vector = EmotionVector(
            confidence=round(confidence, 4),
            curiosity=round(curiosity, 4),
            caution=round(caution, 4),
            urgency=round(urgency, 4),
            fatigue=round(fatigue, 4),
            satisfaction=round(satisfaction, 4),
            confusion=round(confusion, 4),
            risk=round(risk, 4),
        )
        actions = tuple(self._actions(vector))
        return EmotionFeedback(
            vector=vector,
            dominant_state=self._dominant_state(vector),
            actions=actions,
            learning_policy=self._learning_policy(vector),
            explanation=tuple(
                self._explain(
                    top_score=top_score,
                    gap=gap,
                    completeness=completeness,
                    conflict_pressure=conflict_pressure,
                    novelty=novelty,
                    risk=risk,
                    fatigue=fatigue,
                )
            ),
        )

    def build_supervision(
        self,
        feedback: EmotionFeedback,
        *,
        next_feedback: EmotionFeedback | None = None,
        label_source: str = "heuristic",
        config: EmotionSupervisionConfig | None = None,
    ) -> EmotionSupervisionTarget:
        supervision_config = config or EmotionSupervisionConfig()
        vector = feedback.vector
        next_vector = next_feedback.vector if next_feedback is not None else vector
        action_targets = tuple(action.name for action in feedback.actions)
        inference_gates = {
            "memory_write_gate": round(
                _clamp(
                    0.45
                    + supervision_config.curiosity_write_gain * vector.curiosity
                    + 0.15 * vector.confidence
                    - 0.20 * vector.caution
                ),
                4,
            ),
            "replay_gate": round(_clamp(0.35 + 0.30 * vector.confusion + 0.25 * vector.caution), 4),
            "ask_user_gate": round(_clamp(max(vector.confusion, vector.curiosity, vector.caution)), 4),
            "risk_gate": round(_clamp(0.30 + 0.55 * vector.risk + 0.20 * vector.urgency), 4),
            "summary_gate": round(_clamp(0.25 + supervision_config.fatigue_summary_gain * vector.fatigue), 4),
            "forget_gate": round(_clamp(0.15 + 0.45 * vector.fatigue + 0.15 * vector.confidence), 4),
            "answer_direct_gate": round(
                _clamp(vector.confidence - 0.45 * vector.caution, supervision_config.confidence_gate_floor, 1.0),
                4,
            ),
            "auto_promote_gate": round(
                _clamp(
                    0.30 + 0.45 * vector.confidence - 0.35 * vector.caution,
                    0.0,
                    supervision_config.caution_gate_ceiling,
                ),
                4,
            ),
        }
        loss_weights = {
            "emotion_vector_loss": supervision_config.emotion_vector_loss_weight,
            "action_policy_loss": supervision_config.action_policy_loss_weight,
            "next_emotion_loss": supervision_config.next_emotion_loss_weight,
            "risk_prediction_loss": supervision_config.risk_prediction_loss_weight,
            "calibration_loss": supervision_config.calibration_loss_weight,
        }
        prediction_targets = {
            "next_confidence": next_vector.confidence,
            "next_confusion": next_vector.confusion,
            "next_curiosity": next_vector.curiosity,
            "next_caution": next_vector.caution,
            "next_risk": next_vector.risk,
            "risk_delta": round(next_vector.risk - vector.risk, 4),
            "confusion_delta": round(next_vector.confusion - vector.confusion, 4),
        }
        return EmotionSupervisionTarget(
            vector_target=vector,
            action_targets=action_targets,
            inference_gates=inference_gates,
            loss_weights=loss_weights,
            prediction_targets=prediction_targets,
            label_source=label_source,
        )

    def _hypothesis_scores(self, hypotheses: List[object]) -> Tuple[float, float]:
        if not hypotheses:
            return 0.0, 0.0
        scores = sorted((float(_field(item, "score", 0.0)) for item in hypotheses), reverse=True)
        top = _clamp(scores[0])
        gap = _clamp(scores[0] - scores[1]) if len(scores) >= 2 else top
        return top, gap

    def _evidence_completeness(
        self,
        objects: List[object],
        attributes: List[object],
        relations: List[object],
        events: List[object],
    ) -> float:
        parts = [
            _presence(objects),
            _presence(attributes),
            _presence(relations),
            0.75 * _presence(events) + 0.25,
        ]
        return _clamp(sum(parts) / len(parts))

    def _risk_score(self, result: Mapping[str, object] | object) -> float:
        text = " ".join(_flatten_text(result))
        if not text:
            return 0.0
        hits = sum(1 for keyword in self.config.risk_keywords if keyword in text)
        return _clamp(hits / 4.0)

    def _fatigue_score(self, runtime: Mapping[str, object]) -> float:
        memory_pressure = float(runtime.get("memory_pressure", 0.0) or 0.0)
        token_ratio = float(runtime.get("token_budget_ratio", 0.0) or 0.0)
        propagation_ratio = float(runtime.get("propagation_budget_ratio", 0.0) or 0.0)
        repeated_failures = float(runtime.get("repeated_failures", 0.0) or 0.0)
        return _clamp(
            0.35 * memory_pressure
            + 0.25 * token_ratio
            + 0.20 * propagation_ratio
            + 0.20 * _clamp(repeated_failures / 3.0)
        )

    def _actions(self, vector: EmotionVector) -> List[EmotionAction]:
        actions: List[EmotionAction] = []
        if vector.confusion >= self.config.high_confusion:
            actions.append(EmotionAction("ask_clarifying_question", "解释缺口或候选竞争过高", vector.confusion))
        if vector.curiosity >= self.config.high_curiosity:
            actions.append(EmotionAction("expand_observation_buffer", "存在新概念或缺失证据，适合继续收集样本", vector.curiosity))
        if vector.caution >= self.config.high_caution:
            actions.append(EmotionAction("lower_auto_learning", "冲突、风险或不确定性较高，降低自动固化概率", vector.caution))
        if vector.urgency >= self.config.high_urgency:
            actions.append(EmotionAction("prioritize_risk_response", "风险线索较强，优先输出风险解释和追问", vector.urgency))
        if vector.fatigue >= self.config.high_fatigue:
            actions.append(EmotionAction("summarize_and_forget", "运行压力较高，触发摘要合并和逐步遗忘", vector.fatigue))
        if vector.confidence >= self.config.high_confidence and vector.caution < self.config.high_caution:
            actions.append(EmotionAction("answer_directly", "证据完整且候选区分明显", vector.confidence))
        if not actions:
            actions.append(EmotionAction("continue_balanced_reasoning", "状态未触发强反馈，保持常规推理", 0.5))
        actions.sort(key=lambda item: item.intensity, reverse=True)
        return actions

    def _dominant_state(self, vector: EmotionVector) -> str:
        values = vector.as_dict()
        return max(values.items(), key=lambda item: item[1])[0]

    def _learning_policy(self, vector: EmotionVector) -> Dict[str, float]:
        return {
            "memory_write_gain": round(_clamp(0.55 + 0.25 * vector.curiosity - 0.18 * vector.caution), 4),
            "replay_gain": round(_clamp(0.48 + 0.28 * vector.confusion + 0.18 * vector.caution), 4),
            "ask_user_gain": round(_clamp(max(vector.confusion, vector.curiosity, vector.caution)), 4),
            "auto_promote_threshold_delta": round(_clamp(0.10 * vector.caution - 0.05 * vector.curiosity, -0.08, 0.12), 4),
            "summary_gain": round(_clamp(0.35 + 0.45 * vector.fatigue + 0.15 * vector.confusion), 4),
            "forgetting_gain": round(_clamp(0.20 + 0.55 * vector.fatigue + 0.10 * vector.confidence), 4),
            "output_hedging_gain": round(_clamp(0.15 + 0.55 * vector.caution + 0.25 * vector.confusion), 4),
        }

    def _explain(
        self,
        *,
        top_score: float,
        gap: float,
        completeness: float,
        conflict_pressure: float,
        novelty: float,
        risk: float,
        fatigue: float,
    ) -> Iterable[str]:
        yield f"top_hypothesis_score={top_score:.3f}, hypothesis_gap={gap:.3f}"
        yield f"evidence_completeness={completeness:.3f}, conflict_pressure={conflict_pressure:.3f}"
        yield f"novelty={novelty:.3f}, risk={risk:.3f}, fatigue={fatigue:.3f}"


def build_emotion_feedback(
    result: Mapping[str, object] | object,
    *,
    runtime: Mapping[str, object] | None = None,
    config: EmotionFeedbackConfig | None = None,
) -> EmotionFeedback:
    return MachineEmotionFeedbackEngine(config).evaluate(result, runtime=runtime)


def build_emotion_supervision(
    feedback: EmotionFeedback,
    *,
    next_feedback: EmotionFeedback | None = None,
    label_source: str = "heuristic",
    config: EmotionSupervisionConfig | None = None,
) -> EmotionSupervisionTarget:
    return MachineEmotionFeedbackEngine().build_supervision(
        feedback,
        next_feedback=next_feedback,
        label_source=label_source,
        config=config,
    )


def emotion_supervision_loss_spec() -> Dict[str, str]:
    """返回情绪监督在训练目标中的形式化位置。"""

    return {
        "emotion_vector_loss": "L_emo = MSE(e_hat_t, e_target_t)，监督 confidence/curiosity/caution/urgency/fatigue 等反馈向量。",
        "action_policy_loss": "L_act = CE(pi_hat_t, action_target_t)，监督 ask/replay/write/summary/answer 等控制动作。",
        "next_emotion_loss": "L_next = MSE(e_hat_{t+1}, e_target_{t+1})，训练模型预测下一步困惑、风险和信心变化。",
        "risk_prediction_loss": "L_risk = BCE(risk_hat_t, risk_target_t)，把风险线索作为预测监督。",
        "calibration_loss": "L_cal = |confidence_t - task_success_t|，约束置信度与真实任务成功率一致。",
        "total": (
            "L = L_task + lambda_emo L_emo + lambda_act L_act + lambda_next L_next "
            "+ lambda_risk L_risk + lambda_cal L_cal"
        ),
    }


def append_emotion_feedback_to_payload(
    payload: Dict[str, object],
    result: Mapping[str, object] | object,
    *,
    runtime: Mapping[str, object] | None = None,
    config: EmotionFeedbackConfig | None = None,
) -> Dict[str, object]:
    payload["emotion_feedback"] = build_emotion_feedback(result, runtime=runtime, config=config).as_dict()
    return payload


def append_emotion_supervision_to_payload(
    payload: Dict[str, object],
    feedback: EmotionFeedback,
    *,
    next_feedback: EmotionFeedback | None = None,
    label_source: str = "heuristic",
    config: EmotionSupervisionConfig | None = None,
) -> Dict[str, object]:
    payload["emotion_supervision"] = build_emotion_supervision(
        feedback,
        next_feedback=next_feedback,
        label_source=label_source,
        config=config,
    ).as_dict()
    return payload


def _presence(items: List[object]) -> float:
    return 1.0 if items else 0.0


def _field(source: Mapping[str, object] | object, name: str, default: object = None) -> object:
    if isinstance(source, Mapping):
        return source.get(name, default)
    return getattr(source, name, default)


def _flatten_text(value: object) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping):
        texts: List[str] = []
        for item in value.values():
            texts.extend(_flatten_text(item))
        return texts
    if isinstance(value, (list, tuple, set)):
        texts = []
        for item in value:
            texts.extend(_flatten_text(item))
        return texts
    return [str(value)]
