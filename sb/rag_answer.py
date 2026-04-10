from __future__ import annotations

from typing import Dict, List, Sequence


def build_grounded_answer(
    query_text: str,
    analysis: Dict[str, object],
    retrieved_documents: Sequence[Dict[str, object]],
) -> Dict[str, object]:
    grounded_facts = _build_grounded_facts(analysis)
    uncertainties = _build_uncertainties(analysis)
    citations = _build_citations(retrieved_documents)
    answer = _compose_answer(query_text, analysis, grounded_facts, citations, uncertainties)
    follow_up_question = None
    proactive_questions = analysis.get("proactive_questions", [])
    if isinstance(proactive_questions, list) and proactive_questions:
        first = proactive_questions[0]
        if isinstance(first, dict):
            follow_up_question = first.get("question")

    return {
        "answer": answer,
        "grounded_facts": grounded_facts,
        "uncertainties": uncertainties,
        "follow_up_question": follow_up_question,
        "citations": citations,
    }


def _compose_answer(
    query_text: str,
    analysis: Dict[str, object],
    grounded_facts: Sequence[str],
    citations: Sequence[Dict[str, object]],
    uncertainties: Sequence[str],
) -> str:
    best = analysis.get("best_hypothesis")
    if isinstance(best, dict) and best.get("label"):
        answer = f"针对“{query_text}”，当前更像“{best['label']}”场景，置信分约为 {float(best.get('score', 0.0)):.3f}。"
    else:
        answer = f"针对“{query_text}”，当前还不能稳定判断具体场景类型。"

    if grounded_facts:
        answer += " 主要依据是：" + "；".join(grounded_facts[:3]) + "。"

    if citations:
        titles = [item["title"] for item in citations[:2] if item.get("title")]
        if titles:
            answer += " 外部文档检索命中了：" + "、".join(titles) + "。"

    if uncertainties:
        answer += " 目前仍不确定的是：" + "；".join(uncertainties[:2]) + "。"

    return answer


def _build_grounded_facts(analysis: Dict[str, object]) -> List[str]:
    facts: List[str] = []

    for item in analysis.get("objects", [])[:3]:
        label = item.get("label")
        category = item.get("category")
        if label and category:
            facts.append(f"识别到对象“{label}”（{category}）")
    for item in analysis.get("attributes", [])[:2]:
        target = item.get("target_label")
        label = item.get("label")
        if target and label:
            facts.append(f"对象“{target}”具有“{label}”属性")
    for item in analysis.get("relations", [])[:2]:
        source = item.get("source_label")
        relation = item.get("type")
        target = item.get("target_label")
        if source and relation and target:
            facts.append(f"对象关系为“{source} {relation} {target}”")
    for item in analysis.get("events", [])[:1]:
        event_type = item.get("type")
        target = item.get("target_label")
        if event_type and target:
            facts.append(f"识别到事件“{event_type} -> {target}”")

    deduped: List[str] = []
    seen = set()
    for fact in facts:
        if fact in seen:
            continue
        seen.add(fact)
        deduped.append(fact)
    return deduped[:5]


def _build_uncertainties(analysis: Dict[str, object]) -> List[str]:
    uncertainties: List[str] = []
    hypotheses = analysis.get("scene_hypotheses", [])
    if isinstance(hypotheses, list) and len(hypotheses) >= 2:
        top_1 = float(hypotheses[0].get("score", 0.0))
        top_2 = float(hypotheses[1].get("score", 0.0))
        if abs(top_1 - top_2) < 0.08:
            uncertainties.append("前两名场景解释分数接近")
    if not analysis.get("relations"):
        uncertainties.append("对象之间的位置关系还不完整")
    if not analysis.get("events"):
        uncertainties.append("事件线索不足")
    return uncertainties[:3]


def _build_citations(retrieved_documents: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    citations: List[Dict[str, object]] = []
    for item in retrieved_documents[:3]:
        citations.append(
            {
                "title": item.get("title", ""),
                "source_url": item.get("source_url", ""),
                "score": item.get("score", 0.0),
                "matched_terms": list(item.get("matched_terms", [])),
            }
        )
    return citations
