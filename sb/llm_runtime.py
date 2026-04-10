from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence


@dataclass(frozen=True)
class SBLLMRuntimeConfig:
    max_history_turns: int = 4
    max_history_chars: int = 400
    max_structured_facts: int = 18
    max_memory_items: int = 8
    max_reasoning_steps: int = 4
    max_total_chars: int = 6000


class SBLLMRuntime:
    def __init__(self, config: SBLLMRuntimeConfig | None = None) -> None:
        self.config = config or SBLLMRuntimeConfig()

    def build_packet(
        self,
        input_text: str,
        analysis: Dict[str, object],
        retrieved_memories: Sequence[Dict[str, object]],
        dialog_state: Dict[str, object] | None = None,
        history_summary: str = "",
    ) -> Dict[str, object]:
        structured_facts = self._build_structured_facts(analysis)
        memory_items = self._select_memories(retrieved_memories)
        reasoning_notes = self._build_reasoning_notes(analysis)
        response_schema = self._build_response_schema()

        system_prompt = (
            "你是笨鸟神经网络的语言生成层。"
            "你的任务不是自由发挥，而是把结构化分析结果、检索记忆和会话历史组织成可靠回答。"
            "如果证据不足，就明确说不确定，并优先提出澄清问题。"
        )
        developer_prompt = (
            "回答必须遵守以下约束："
            "1. 优先依据 structured_facts 和 retrieved_memories；"
            "2. 不要捏造输入中不存在的对象、事件或关系；"
            "3. 如果存在 proactive_questions，可在回答末尾自然提出其中最关键的一条；"
            "4. 输出风格简洁、 grounded、可审计。"
        )

        user_sections = [
            f"当前用户输入：{input_text}",
            f"最佳场景假设：{self._best_hypothesis_line(analysis)}",
        ]
        if history_summary:
            user_sections.append(f"近期对话摘要：{history_summary}")
        if structured_facts:
            user_sections.append("结构化事实：\n- " + "\n- ".join(structured_facts))
        if memory_items:
            user_sections.append("检索记忆：\n- " + "\n- ".join(memory_items))
        if reasoning_notes:
            user_sections.append("推理提示：\n- " + "\n- ".join(reasoning_notes))
        proactive_lines = self._build_proactive_lines(analysis)
        if proactive_lines:
            user_sections.append("可选澄清问题：\n- " + "\n- ".join(proactive_lines))
        user_prompt = "\n\n".join(user_sections)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "developer", "content": developer_prompt},
            {"role": "user", "content": self._truncate(user_prompt)},
        ]

        return {
            "messages": messages,
            "response_schema": response_schema,
            "memory_blocks": {
                "history_summary": history_summary,
                "structured_facts": structured_facts,
                "retrieved_memories": memory_items,
                "reasoning_notes": reasoning_notes,
            },
            "metadata": {
                "input_text": input_text,
                "best_hypothesis": analysis.get("best_hypothesis"),
                "dialog_state": dialog_state or {},
                "grounding_rules": [
                    "只使用输入、结构化结果和检索记忆中的事实",
                    "证据不足时明确说明不确定",
                    "优先回答，再决定是否提出澄清问题",
                ],
                "generation_mode": self._generation_mode(analysis),
            },
        }

    def _build_structured_facts(self, analysis: Dict[str, object]) -> List[str]:
        facts: List[str] = []
        for item in analysis.get("objects", [])[: self.config.max_structured_facts]:
            facts.append(f"对象：{item['label']}（类别：{item['category']}）")
        for item in analysis.get("attributes", [])[: self.config.max_structured_facts]:
            target = item.get("target_label") or "scene"
            facts.append(f"属性：{target} -> {item['label']}")
        for item in analysis.get("relations", [])[: self.config.max_structured_facts]:
            facts.append(f"关系：{item['source_label']} {item['type']} {item['target_label']}")
        for item in analysis.get("events", [])[: self.config.max_structured_facts]:
            target = item.get("target_label") or "unknown"
            facts.append(f"事件：{item['type']} -> {target}")
        deduped: List[str] = []
        seen = set()
        for fact in facts:
            if fact in seen:
                continue
            seen.add(fact)
            deduped.append(fact)
            if len(deduped) >= self.config.max_structured_facts:
                break
        return deduped

    def _select_memories(self, retrieved_memories: Sequence[Dict[str, object]]) -> List[str]:
        selected: List[str] = []
        for item in retrieved_memories[: self.config.max_memory_items]:
            supports = "、".join(item.get("supports", []))
            line = f"{item['label']}（{item['space_type']}，score={item['score']}）"
            if supports:
                line += f"，支持场景：{supports}"
            selected.append(line)
        return selected

    def _build_reasoning_notes(self, analysis: Dict[str, object]) -> List[str]:
        notes: List[str] = []
        for item in analysis.get("reasoning_path", [])[: self.config.max_reasoning_steps]:
            notes.append(item["contribution"])
        return notes

    def _build_proactive_lines(self, analysis: Dict[str, object]) -> List[str]:
        lines: List[str] = []
        for item in analysis.get("proactive_questions", [])[:2]:
            question = item.get("question")
            if question:
                lines.append(str(question))
        return lines

    def _build_response_schema(self) -> Dict[str, object]:
        return {
            "answer": "string",
            "grounded_facts": ["string"],
            "uncertainties": ["string"],
            "follow_up_question": "string|null",
        }

    def _best_hypothesis_line(self, analysis: Dict[str, object]) -> str:
        best = analysis.get("best_hypothesis")
        if not isinstance(best, dict) or not best:
            return "无"
        return f"{best.get('label', '未知')} ({best.get('score', 0.0)})"

    def _generation_mode(self, analysis: Dict[str, object]) -> str:
        proactive = analysis.get("proactive_questions", [])
        if proactive:
            return "grounded_answer_with_clarification"
        return "grounded_answer"

    def _truncate(self, text: str) -> str:
        if len(text) <= self.config.max_total_chars:
            return text
        return text[: self.config.max_total_chars - 1] + "…"
