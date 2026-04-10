from __future__ import annotations

from typing import Dict, List, Sequence

from .llm_runtime import SBLLMRuntime
from .vector_memory import VectorHit, VectorMemoryIndex


def build_llm_context(
    input_text: str,
    analysis: Dict[str, object],
    vector_hits: Sequence[VectorHit],
    dialog_state: Dict[str, object] | None = None,
    history_summary: str = "",
) -> Dict[str, object]:
    runtime = SBLLMRuntime()
    memory_lines = [
        {
            "label": hit.label,
            "space_type": hit.space_type,
            "score": round(hit.score, 3),
            "supports": list(hit.supports),
            "text": hit.text,
        }
        for hit in vector_hits[:6]
    ]
    packet = runtime.build_packet(
        input_text=input_text,
        analysis=analysis,
        retrieved_memories=memory_lines,
        dialog_state=dialog_state,
        history_summary=history_summary,
    )
    packet["structured_analysis"] = analysis
    packet["retrieved_memories"] = memory_lines
    return packet


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
