from __future__ import annotations

from typing import Dict, List, Sequence

from .vector_memory import VectorHit, VectorMemoryIndex


def build_llm_context(
    input_text: str,
    analysis: Dict[str, object],
    vector_hits: Sequence[VectorHit],
) -> Dict[str, object]:
    scene_lines = []
    for item in analysis.get("scene_hypotheses", [])[:3]:
        scene_lines.append(f"{item['label']} ({item['score']})")

    memory_lines = []
    for hit in vector_hits[:6]:
        memory_lines.append(
            {
                "label": hit.label,
                "space_type": hit.space_type,
                "score": round(hit.score, 3),
                "supports": list(hit.supports),
                "text": hit.text,
            }
        )

    system_prompt = (
        "你是笨鸟的语言生成层。"
        "必须优先依据结构化分析结果和检索到的记忆回答，"
        "不要捏造未在输入、结构化结果或记忆中出现的事实。"
    )
    user_prompt = (
        f"用户输入：{input_text}\n"
        f"候选场景：{'；'.join(scene_lines) if scene_lines else '无'}\n"
        "请先参考结构化结果，再组织自然语言回复。"
    )

    return {
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "structured_analysis": analysis,
        "retrieved_memories": memory_lines,
    }


def retrieve_for_llm(
    index: VectorMemoryIndex,
    input_text: str,
    analysis: Dict[str, object],
    top_k: int = 6,
) -> List[VectorHit]:
    query_terms: List[str] = [input_text]
    for field in ("objects", "attributes", "relations", "events"):
        for item in analysis.get(field, []):
            label = item.get("label") or item.get("type") or item.get("normalized_type")
            if label:
                query_terms.append(str(label))
    seen = set()
    hits: List[VectorHit] = []
    for query in query_terms:
        for hit in index.search(query, top_k=top_k, min_score=0.18):
            if hit.memory_id in seen:
                continue
            seen.add(hit.memory_id)
            hits.append(hit)
    hits.sort(key=lambda item: item.score, reverse=True)
    return hits[:top_k]
