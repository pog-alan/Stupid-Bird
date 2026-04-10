from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence

from .embedding_backends import EmbeddingEncoder, HashedVectorEncoder
from .llm_runtime import SBLLMRuntime
from .rag_answer import build_grounded_answer
from .rag_store import DocumentChunk, RAGKnowledgeBase
from .reasoner import SBV01Engine


@dataclass(frozen=True)
class RAGSearchHit:
    chunk_id: str
    doc_id: str
    source_url: str
    title: str
    text: str
    score: float
    vector_score: float
    lexical_score: float
    matched_terms: tuple[str, ...] = ()


@dataclass(frozen=True)
class SBRAGConfig:
    top_k_docs: int = 6
    per_query_top_k: int = 6
    min_vector_score: float = 0.18
    chunk_size: int = 220
    chunk_overlap: int = 40
    max_context_docs: int = 6


class DocumentVectorIndex:
    def __init__(
        self,
        chunks: Sequence[DocumentChunk],
        encoder: EmbeddingEncoder | None = None,
    ) -> None:
        self.chunks = list(chunks)
        self.encoder = encoder or HashedVectorEncoder()
        encoded = self.encoder.encode_documents([chunk.text for chunk in self.chunks]) if self.chunks else []
        self._vectors = {
            chunk.chunk_id: vector
            for chunk, vector in zip(self.chunks, encoded)
        }

    def search(self, query: str, top_k: int = 6, min_score: float = 0.18) -> List[RAGSearchHit]:
        if not query.strip() or not self.chunks:
            return []
        query_vector = self.encoder.encode_query(query)
        hits: List[RAGSearchHit] = []
        for chunk in self.chunks:
            vector_score = _cosine(query_vector, self._vectors[chunk.chunk_id])
            if vector_score < min_score:
                continue
            hits.append(
                RAGSearchHit(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    source_url=chunk.source_url,
                    title=chunk.title,
                    text=chunk.text,
                    score=vector_score,
                    vector_score=vector_score,
                    lexical_score=0.0,
                )
            )
        hits.sort(key=lambda item: item.score, reverse=True)
        return hits[:top_k]


class SBRAGPipeline:
    def __init__(
        self,
        engine: SBV01Engine,
        knowledge_base: RAGKnowledgeBase,
        encoder: EmbeddingEncoder | None = None,
        config: SBRAGConfig | None = None,
    ) -> None:
        self.engine = engine
        self.knowledge_base = knowledge_base
        self.encoder = encoder or HashedVectorEncoder()
        self.config = config or SBRAGConfig()
        self.runtime = SBLLMRuntime()
        self._index = DocumentVectorIndex(self.knowledge_base.chunks, encoder=self.encoder)

    def refresh(self) -> None:
        self._index = DocumentVectorIndex(self.knowledge_base.chunks, encoder=self.encoder)

    def ingest_documents(
        self,
        documents: Iterable[Mapping[str, object]],
    ) -> int:
        changed = 0
        for item in documents:
            updated = self.knowledge_base.upsert_document(
                source_url=str(item.get("source_url", "")),
                title=str(item.get("title", "")),
                text=str(item.get("text", "")),
                metadata=dict(item.get("metadata", {})) if item.get("metadata") else {},
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
            )
            if updated:
                changed += 1
        if changed:
            self.knowledge_base.save()
            self.refresh()
        return changed

    def query(
        self,
        text: str,
        dialog_state: Dict[str, object] | None = None,
        history_summary: str = "",
    ) -> Dict[str, object]:
        analysis = self.engine.analyze(text)
        memory_hits = self.engine.retrieve_memories(text, analysis, top_k=self.config.top_k_docs)
        document_hits = self.retrieve_documents(text, analysis)
        llm_packet = self.build_llm_packet(
            text,
            analysis,
            memory_hits,
            document_hits,
            dialog_state=dialog_state,
            history_summary=history_summary,
        )
        answer_draft = build_grounded_answer(text, analysis, document_hits)
        return {
            "analysis": analysis,
            "retrieved_memories": memory_hits,
            "retrieved_documents": document_hits,
            "answer_draft": answer_draft,
            "llm_packet": llm_packet,
        }

    def retrieve_documents(self, text: str, analysis: Dict[str, object]) -> List[Dict[str, object]]:
        query_terms = self._build_query_terms(text, analysis)
        aggregated: Dict[str, Dict[str, object]] = {}
        for query in query_terms:
            for hit in self._index.search(
                query,
                top_k=self.config.per_query_top_k,
                min_score=self.config.min_vector_score,
            ):
                lexical_score, matched_terms = self._lexical_bonus(hit.text, analysis)
                current = aggregated.get(hit.chunk_id)
                combined = hit.vector_score + lexical_score
                if current is None or combined > float(current["score"]):
                    aggregated[hit.chunk_id] = {
                        "chunk_id": hit.chunk_id,
                        "doc_id": hit.doc_id,
                        "source_url": hit.source_url,
                        "title": hit.title,
                        "text": hit.text,
                        "score": round(combined, 3),
                        "vector_score": round(hit.vector_score, 3),
                        "lexical_score": round(lexical_score, 3),
                        "matched_terms": matched_terms,
                    }
        ranked = sorted(aggregated.values(), key=lambda item: float(item["score"]), reverse=True)
        return ranked[: self.config.top_k_docs]

    def build_llm_packet(
        self,
        text: str,
        analysis: Dict[str, object],
        retrieved_memories: Sequence[Dict[str, object]],
        retrieved_documents: Sequence[Dict[str, object]],
        dialog_state: Dict[str, object] | None = None,
        history_summary: str = "",
    ) -> Dict[str, object]:
        packet = self.runtime.build_packet(
            input_text=text,
            analysis=analysis,
            retrieved_memories=retrieved_memories,
            dialog_state=dialog_state,
            history_summary=history_summary,
        )
        evidence_blocks = [
            {
                "title": item["title"],
                "source_url": item["source_url"],
                "score": item["score"],
                "matched_terms": list(item.get("matched_terms", [])),
                "text": item["text"],
            }
            for item in retrieved_documents[: self.config.max_context_docs]
        ]
        if evidence_blocks:
            doc_lines = [
                f"{item['title']}（score={item['score']}）：{item['text']}"
                for item in evidence_blocks
            ]
            packet["messages"][-1]["content"] += "\n\n外部文档证据：\n- " + "\n- ".join(doc_lines)
        packet["memory_blocks"]["retrieved_documents"] = evidence_blocks
        packet["metadata"]["rag_mode"] = "hybrid_structured_vector_rag"
        packet["structured_analysis"] = analysis
        packet["retrieved_memories"] = list(retrieved_memories)
        packet["retrieved_documents"] = evidence_blocks
        return packet

    def _build_query_terms(self, text: str, analysis: Dict[str, object]) -> List[str]:
        terms = [text]
        best = analysis.get("best_hypothesis")
        if isinstance(best, dict) and best.get("label"):
            terms.append(str(best["label"]))
        for field in ("objects", "attributes", "relations", "events"):
            for item in analysis.get(field, []):
                label = item.get("label") or item.get("type") or item.get("normalized_type")
                if label:
                    terms.append(str(label))
        deduped: List[str] = []
        seen = set()
        for term in terms:
            if term in seen:
                continue
            seen.add(term)
            deduped.append(term)
        return deduped

    def _lexical_bonus(self, text: str, analysis: Dict[str, object]) -> tuple[float, tuple[str, ...]]:
        labels: List[str] = []
        for field in ("objects", "attributes", "relations", "events"):
            for item in analysis.get(field, []):
                label = item.get("label") or item.get("type") or item.get("normalized_type")
                if label:
                    labels.append(str(label))
        best = analysis.get("best_hypothesis")
        if isinstance(best, dict) and best.get("label"):
            labels.append(str(best["label"]))
        matched = sorted({label for label in labels if label and label in text})
        bonus = min(0.25, 0.04 * len(matched))
        return bonus, tuple(matched)


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    left_norm = sum(value * value for value in left) ** 0.5
    right_norm = sum(value * value for value in right) ** 0.5
    if left_norm <= 0.0 or right_norm <= 0.0:
        return 0.0
    return sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)
