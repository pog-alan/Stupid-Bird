from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional

from .auto_crawl import CrawlResult


@dataclass
class SourceDocument:
    doc_id: str
    source_url: str
    title: str
    text: str
    updated_at: str
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass
class DocumentChunk:
    chunk_id: str
    doc_id: str
    source_url: str
    title: str
    text: str
    position: int
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass
class RAGKnowledgeBase:
    path: Path | str
    documents: Dict[str, SourceDocument] = field(default_factory=dict)
    chunks: List[DocumentChunk] = field(default_factory=list)

    @classmethod
    def load(cls, path: str | Path) -> "RAGKnowledgeBase":
        path = Path(path)
        if not path.exists():
            return cls(path=path)
        raw = json.loads(path.read_text(encoding="utf-8"))
        documents = {
            item["doc_id"]: SourceDocument(
                doc_id=item["doc_id"],
                source_url=item["source_url"],
                title=item["title"],
                text=item["text"],
                updated_at=item["updated_at"],
                metadata=dict(item.get("metadata", {})),
            )
            for item in raw.get("documents", [])
        }
        chunks = [
            DocumentChunk(
                chunk_id=item["chunk_id"],
                doc_id=item["doc_id"],
                source_url=item["source_url"],
                title=item["title"],
                text=item["text"],
                position=int(item["position"]),
                metadata=dict(item.get("metadata", {})),
            )
            for item in raw.get("chunks", [])
        ]
        return cls(path=path, documents=documents, chunks=chunks)

    def save(self) -> None:
        payload = {
            "documents": [asdict(item) for item in self.documents.values()],
            "chunks": [asdict(item) for item in self.chunks],
        }
        Path(self.path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def stats(self) -> Dict[str, int]:
        return {
            "documents": len(self.documents),
            "chunks": len(self.chunks),
        }

    def upsert_document(
        self,
        source_url: str,
        title: str,
        text: str,
        metadata: Optional[Mapping[str, object]] = None,
        chunk_size: int = 220,
        chunk_overlap: int = 40,
    ) -> bool:
        normalized_text = normalize_document_text(text)
        if not normalized_text:
            return False

        doc_id = _document_id(source_url or title or normalized_text[:32])
        metadata_dict = dict(metadata or {})
        existing = self.documents.get(doc_id)
        if existing is not None and existing.text == normalized_text and existing.title == title:
            return False

        self.documents[doc_id] = SourceDocument(
            doc_id=doc_id,
            source_url=source_url,
            title=title or doc_id,
            text=normalized_text,
            updated_at=_now(),
            metadata=metadata_dict,
        )
        self.chunks = [item for item in self.chunks if item.doc_id != doc_id]
        for index, chunk_text in enumerate(split_text_into_chunks(normalized_text, chunk_size, chunk_overlap), start=1):
            self.chunks.append(
                DocumentChunk(
                    chunk_id=f"{doc_id}_chunk_{index:03d}",
                    doc_id=doc_id,
                    source_url=source_url,
                    title=title or doc_id,
                    text=chunk_text,
                    position=index,
                    metadata=metadata_dict,
                )
            )
        return True

    def ingest_crawl_results(
        self,
        results: Iterable[CrawlResult],
        chunk_size: int = 220,
        chunk_overlap: int = 40,
    ) -> int:
        changed = 0
        for result in results:
            if result.error or not result.text:
                continue
            title = extract_title(result.text) or result.url
            text = extract_body_text(result.text)
            updated = self.upsert_document(
                source_url=result.url,
                title=title,
                text=text,
                metadata={"content_type": result.content_type},
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            if updated:
                changed += 1
        return changed


def split_text_into_chunks(text: str, chunk_size: int = 220, chunk_overlap: int = 40) -> List[str]:
    cleaned = normalize_document_text(text)
    if not cleaned:
        return []
    if len(cleaned) <= chunk_size:
        return [cleaned]

    units = re.split(r"(?<=[。！？；\n])", cleaned)
    units = [unit.strip() for unit in units if unit.strip()]
    chunks: List[str] = []
    buffer = ""
    for unit in units:
        if not buffer:
            buffer = unit
            continue
        if len(buffer) + len(unit) <= chunk_size:
            buffer += unit
            continue
        chunks.append(buffer)
        if chunk_overlap > 0:
            overlap_text = buffer[-chunk_overlap:]
            buffer = overlap_text + unit
        else:
            buffer = unit
    if buffer:
        chunks.append(buffer)
    return [chunk.strip() for chunk in chunks if chunk.strip()]


def normalize_document_text(text: str) -> str:
    normalized = re.sub(r"<script.*?>.*?</script>", " ", text, flags=re.IGNORECASE | re.DOTALL)
    normalized = re.sub(r"<style.*?>.*?</style>", " ", normalized, flags=re.IGNORECASE | re.DOTALL)
    normalized = re.sub(r"<[^>]+>", " ", normalized)
    normalized = re.sub(r"&nbsp;|&#160;", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def extract_title(text: str) -> str:
    match = re.search(r"<title>(.*?)</title>", text, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return ""
    return normalize_document_text(match.group(1))


def extract_body_text(text: str) -> str:
    body_match = re.search(r"<body.*?>(.*?)</body>", text, flags=re.IGNORECASE | re.DOTALL)
    if body_match:
        return normalize_document_text(body_match.group(1))
    return normalize_document_text(text)


def _document_id(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9\u4e00-\u9fff]+", "_", value).strip("_").lower()
    return normalized[:80] or "document"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
