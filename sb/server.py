from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Dict, Optional, Tuple

from .auto_crawl import CrawlSource, Crawler, CrawlerConfig
from .dialog import DialogStore
from .embedding_backends import EmbeddingEncoder, create_embedding_encoder
from .extractor import SimpleExtractor, merge_candidates
from .ingest import Ingestor, KnowledgeEntry, KnowledgeStore, apply_stable_entries_to_config
from .llm_client import LLMConfig, create_llm_client, load_llm_config
from .ontology import load_default_ontology
from .proactive import append_questions_to_payload
from .quality_gate import QualityGateConfig
from .rag_answer import build_grounded_answer
from .rag_pipeline import SBRAGConfig, SBRAGPipeline
from .rag_store import RAGKnowledgeBase
from .reasoner import SBV01Engine


@dataclass
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 8088
    knowledge_store_path: str = "文档/笨鸟v0.1采集知识库.json"
    online_config_path: str = "文档/笨鸟v0.1配置在线.json"
    base_config_path: str = "文档/笨鸟v0.1配置样例.json"
    rag_store_path: str = "文档/笨鸟v0.1文档知识库.json"
    refresh_seconds: int = 1800
    crawl_interval_seconds: int = 3600
    max_pages: int = 200
    timeout_seconds: int = 10
    allow_robots: bool = True
    allow_redirects: bool = True
    accepted_mime_prefixes: Tuple[str, ...] = ("text/", "application/json")
    sources: Tuple[CrawlSource, ...] = ()
    quality_gate: QualityGateConfig = field(default_factory=QualityGateConfig)
    proactive_template_path: str = "文档/笨鸟v0.1主动提问模板.json"
    vector_top_k: int = 6
    vector_min_score: float = 0.2
    embedding_backend: Dict[str, object] = field(default_factory=dict)
    rag_top_k_docs: int = 6
    rag_per_query_top_k: int = 6
    rag_chunk_size: int = 220
    rag_chunk_overlap: int = 40
    llm: LLMConfig = field(default_factory=LLMConfig)


class KnowledgeSnapshot:
    def __init__(
        self,
        config_path: str,
        knowledge_store_path: Optional[str] = None,
        rag_store_path: Optional[str] = None,
        embedding_encoder: EmbeddingEncoder | None = None,
        vector_top_k: int = 6,
        vector_min_score: float = 0.2,
        rag_config: SBRAGConfig | None = None,
    ) -> None:
        self._lock = threading.Lock()
        self._config_path = config_path
        self._knowledge_store_path = knowledge_store_path
        self._rag_store_path = rag_store_path
        self._embedding_encoder = embedding_encoder
        self._vector_top_k = vector_top_k
        self._vector_min_score = vector_min_score
        self._rag_config = rag_config or SBRAGConfig()
        self._proactive_template_path: Optional[str] = None
        self._ontology = load_default_ontology(config_path)
        self._engine = self._build_engine()
        self._rag_knowledge_base = self._load_rag_knowledge_base()
        self._rag_pipeline = self._build_rag_pipeline()
        self._updated_at = datetime.now(timezone.utc)

    def refresh(
        self,
        config_path: Optional[str] = None,
        proactive_template_path: Optional[str] = None,
        knowledge_store_path: Optional[str] = None,
        rag_store_path: Optional[str] = None,
    ) -> None:
        with self._lock:
            if config_path is not None:
                self._config_path = config_path
            if proactive_template_path is not None:
                self._proactive_template_path = proactive_template_path
            if knowledge_store_path is not None:
                self._knowledge_store_path = knowledge_store_path
            if rag_store_path is not None:
                self._rag_store_path = rag_store_path
            self._ontology = load_default_ontology(self._config_path)
            self._engine = self._build_engine()
            self._rag_knowledge_base = self._load_rag_knowledge_base()
            self._rag_pipeline = self._build_rag_pipeline()
            self._updated_at = datetime.now(timezone.utc)

    def analyze(self, text: str) -> Dict[str, object]:
        with self._lock:
            result = self._engine.analyze(text)
            result["retrieved_memories"] = self._engine.retrieve_memories(text, result, top_k=6)
            if self._proactive_template_path:
                append_questions_to_payload(result, result, self._proactive_template_path)
            return result

    def query_rag(
        self,
        text: str,
        dialog_state: Dict[str, object] | None = None,
        history_summary: str = "",
    ) -> Dict[str, object]:
        with self._lock:
            result = self._rag_pipeline.query(text, dialog_state=dialog_state, history_summary=history_summary)
            if self._proactive_template_path:
                append_questions_to_payload(result["analysis"], result["analysis"], self._proactive_template_path)
                result["answer_draft"] = build_grounded_answer(
                    text,
                    result["analysis"],
                    result["retrieved_documents"],
                )
                result["llm_packet"] = self._rag_pipeline.build_llm_packet(
                    text,
                    result["analysis"],
                    result["retrieved_memories"],
                    result["retrieved_documents"],
                    dialog_state=dialog_state,
                    history_summary=history_summary,
                )
            return result

    def build_llm_payload(
        self,
        text: str,
        analysis: Dict[str, object],
        dialog_state: Dict[str, object] | None = None,
        history_summary: str = "",
    ) -> Dict[str, object]:
        with self._lock:
            retrieved_memories = analysis.get("retrieved_memories")
            retrieved_documents = analysis.get("retrieved_documents")
            if isinstance(retrieved_memories, list) and isinstance(retrieved_documents, list):
                return self._rag_pipeline.build_llm_packet(
                    text,
                    analysis,
                    retrieved_memories,
                    retrieved_documents,
                    dialog_state=dialog_state,
                    history_summary=history_summary,
                )
            return self._engine.build_llm_payload(
                text,
                analysis,
                dialog_state=dialog_state,
                history_summary=history_summary,
            )

    def status(self) -> Dict[str, str]:
        with self._lock:
            rag_stats = self._rag_knowledge_base.stats()
            return {
                "config_path": self._config_path,
                "knowledge_store_path": self._knowledge_store_path or "",
                "rag_store_path": self._rag_store_path or "",
                "embedding_backend": getattr(self._embedding_encoder, "backend_name", "unknown"),
                "embedding_model": getattr(self._embedding_encoder, "model_name", ""),
                "rag_documents": str(rag_stats["documents"]),
                "rag_chunks": str(rag_stats["chunks"]),
                "updated_at": self._updated_at.isoformat(),
            }

    def _build_engine(self) -> SBV01Engine:
        stable_entries = self._load_stable_entries()
        return SBV01Engine(
            self._ontology,
            stable_entries=stable_entries,
            embedding_encoder=self._embedding_encoder,
            vector_top_k=self._vector_top_k,
            vector_min_score=self._vector_min_score,
        )

    def _build_rag_pipeline(self) -> SBRAGPipeline:
        return SBRAGPipeline(
            self._engine,
            self._rag_knowledge_base,
            encoder=self._embedding_encoder,
            config=self._rag_config,
        )

    def _load_stable_entries(self) -> list[KnowledgeEntry]:
        if not self._knowledge_store_path:
            return []
        store_path = Path(self._knowledge_store_path)
        if not store_path.exists():
            return []
        return KnowledgeStore.load(store_path).stable_entries()

    def _load_rag_knowledge_base(self) -> RAGKnowledgeBase:
        if not self._rag_store_path:
            return RAGKnowledgeBase(path=Path("文档") / "笨鸟v0.1文档知识库.json")
        return RAGKnowledgeBase.load(self._rag_store_path)


class BackgroundLearner(threading.Thread):
    def __init__(self, config: ServerConfig, snapshot: KnowledgeSnapshot) -> None:
        super().__init__(daemon=True)
        self.config = config
        self.snapshot = snapshot
        self._stop_event = threading.Event()
        self._last_crawl = 0.0

    def run(self) -> None:
        while not self._stop_event.is_set():
            now = time.time()
            if now - self._last_crawl >= self.config.crawl_interval_seconds:
                self._last_crawl = now
                self._run_cycle()
            time.sleep(1.0)

    def stop(self) -> None:
        self._stop_event.set()

    def _run_cycle(self) -> None:
        if not self.config.sources:
            return

        crawler = Crawler(
            CrawlerConfig(
                sources=self.config.sources,
                max_pages=self.config.max_pages,
                timeout_seconds=self.config.timeout_seconds,
                allow_robots=self.config.allow_robots,
                allow_redirects=self.config.allow_redirects,
                accepted_mime_prefixes=self.config.accepted_mime_prefixes,
            )
        )
        store = KnowledgeStore.load(self.config.knowledge_store_path)
        ingestor = Ingestor(store, self.config.quality_gate)
        extractor = SimpleExtractor(self.snapshot._ontology)
        rag_knowledge_base = RAGKnowledgeBase.load(self.config.rag_store_path)

        candidates = []
        crawl_results = crawler.crawl()
        for result in crawl_results:
            if result.error or not result.text:
                continue
            candidates.extend(extractor.extract(result.text, source_url=result.url))

        rag_knowledge_base.ingest_crawl_results(
            crawl_results,
            chunk_size=self.config.rag_chunk_size,
            chunk_overlap=self.config.rag_chunk_overlap,
        )
        rag_knowledge_base.save()

        if candidates:
            merged = merge_candidates(candidates)
            ingestor.ingest(merged)
            store.save()
            output_path = apply_stable_entries_to_config(
                store,
                config_path=self.config.base_config_path,
                output_path=self.config.online_config_path,
            )
            config_path = str(output_path)
        else:
            config_path = (
                self.config.online_config_path
                if Path(self.config.online_config_path).exists()
                else self.config.base_config_path
            )

        self.snapshot.refresh(
            config_path,
            self.config.proactive_template_path,
            self.config.knowledge_store_path,
            self.config.rag_store_path,
        )


class ChatHandler(BaseHTTPRequestHandler):
    snapshot: KnowledgeSnapshot
    dialog_store: DialogStore
    llm_client = None
    llm_config: LLMConfig
    allow_origin: str = "*"
    ui_path: Path = Path(__file__).with_name("web_ui.html")

    def _send_json(self, status: int, payload: Dict[str, object]) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._send_bytes(status, data, "application/json; charset=utf-8")

    def _send_bytes(self, status: int, data: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", self.allow_origin)
        self.end_headers()
        self.wfile.write(data)

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", self.allow_origin)
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        if self.path in {"/", "/index.html"}:
            self._serve_ui()
            return
        if self.path == "/favicon.ico":
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", self.allow_origin)
            self.end_headers()
            return
        if self.path == "/status":
            payload = self.snapshot.status()
            payload["llm_enabled"] = str(self.llm_config.enabled).lower()
            payload["llm_provider"] = self.llm_config.provider
            payload["llm_model"] = self.llm_config.model
            payload["llm_endpoint"] = self.llm_config.endpoint
            payload["llm_api_key_env"] = self.llm_config.api_key_env
            payload["llm_api_key_present"] = str(bool(self.llm_config.api_key)).lower()
            payload["llm_ready"] = str(bool(self.llm_client and self.llm_client.is_ready())).lower()
            self._send_json(200, payload)
            return
        self._send_json(404, {"error": "Unknown endpoint"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path not in {"/chat", "/llm_context", "/rag_query", "/rag_ingest", "/generate"}:
            self._send_json(404, {"error": "Unknown endpoint"})
            return

        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length).decode("utf-8")
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            self._send_json(400, {"error": "Invalid JSON"})
            return

        text = str(payload.get("text") or "").strip()
        session_id = str(payload.get("session_id") or "default").strip()

        if self.path == "/rag_ingest":
            source_url = str(payload.get("source_url", "")).strip()
            title = str(payload.get("title", "")).strip() or source_url or "manual"
            if not text:
                self._send_json(400, {"error": "Missing text"})
                return
            changed = self.snapshot._rag_pipeline.ingest_documents(
                [
                    {
                        "source_url": source_url,
                        "title": title,
                        "text": text,
                        "metadata": payload.get("metadata", {}),
                    }
                ]
            )
            self._send_json(
                200,
                {
                    "ingested_documents": changed,
                    "rag_stats": self.snapshot._rag_knowledge_base.stats(),
                },
            )
            return

        if not text:
            self._send_json(400, {"error": "Missing text"})
            return

        context_text, dialog_state = self._build_context(session_id, text)
        history_summary = str(dialog_state["history_summary"])

        llm_payload = None
        if self.path in {"/rag_query", "/llm_context", "/generate"}:
            rag_result = self.snapshot.query_rag(
                context_text,
                dialog_state=dialog_state,
                history_summary=history_summary,
            )
            result = dict(rag_result["analysis"])
            result["retrieved_memories"] = rag_result["retrieved_memories"]
            result["retrieved_documents"] = rag_result["retrieved_documents"]
            result["answer_draft"] = rag_result["answer_draft"]
            llm_payload = rag_result["llm_packet"]
        else:
            result = self.snapshot.analyze(context_text)

        dialog_state = self._update_dialog_state(session_id, text, result)
        result["dialog_state"] = dialog_state
        if context_text != text:
            result["context_text"] = context_text

        if llm_payload is not None:
            llm_payload["dialog_state"] = dialog_state
            if "context_text" in result:
                llm_payload["context_text"] = result["context_text"]
            if "answer_draft" in result:
                llm_payload["answer_draft"] = result["answer_draft"]

        if self.path == "/chat":
            self._send_json(200, result)
            return

        if self.path == "/rag_query":
            if llm_payload is not None:
                result["llm_packet"] = llm_payload
            self._send_json(200, result)
            return

        if self.path == "/llm_context":
            if llm_payload is None:
                llm_payload = self.snapshot.build_llm_payload(
                    context_text,
                    result,
                    dialog_state=dialog_state,
                    history_summary=str(dialog_state["history_summary"]),
                )
            self._send_json(200, llm_payload)
            return

        if not self.llm_client or not self.llm_client.is_ready():
            self._send_json(
                503,
                {
                    "error": "LLM client not ready",
                    "detail": f"Please set env var {self.llm_config.api_key_env} and llm.base_url.",
                },
            )
            return

        assert llm_payload is not None
        try:
            llm_response = self.llm_client.generate(llm_payload)
        except Exception as exc:  # pragma: no cover - network/provider dependent
            self._send_json(502, {"error": "LLM generation failed", "detail": str(exc)})
            return

        self._send_json(
            200,
            {
                "analysis": result,
                "llm_packet": llm_payload,
                "llm_response": llm_response,
            },
        )

    def _build_context(self, session_id: str, text: str) -> tuple[str, Dict[str, object]]:
        context_text = self.dialog_store.build_context_text(session_id, text)
        preview_state = self.dialog_store.get(session_id)
        history_summary = self.dialog_store.build_history_summary(session_id)
        dialog_state = {
            "session_id": preview_state.session_id,
            "turns": len(preview_state.turns),
            "last_questions": preview_state.last_questions,
            "updated_at": preview_state.updated_at,
            "history_summary": history_summary,
        }
        return context_text, dialog_state

    def _update_dialog_state(self, session_id: str, text: str, result: Dict[str, object]) -> Dict[str, object]:
        questions = [item["question"] for item in result.get("proactive_questions", [])]
        state = self.dialog_store.update(session_id=session_id, text=text, questions=questions)
        history_summary = self.dialog_store.build_history_summary(session_id)
        return {
            "session_id": state.session_id,
            "turns": len(state.turns),
            "last_questions": state.last_questions,
            "updated_at": state.updated_at,
            "history_summary": history_summary,
        }

    def _serve_ui(self) -> None:
        if not self.ui_path.exists():
            self._send_json(500, {"error": "UI file missing"})
            return
        data = self.ui_path.read_bytes()
        self._send_bytes(200, data, "text/html; charset=utf-8")


def load_server_config(path: str | Path) -> ServerConfig:
    raw = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    sources = tuple(
        CrawlSource(
            name=item["name"],
            seed_urls=tuple(item["seed_urls"]),
            allowed_domains=tuple(item["allowed_domains"]),
            max_depth=item.get("max_depth", 1),
            rate_limit_seconds=item.get("rate_limit_seconds", 1.0),
            user_agent=item.get("user_agent", "SB-V01-Crawler/0.1"),
        )
        for item in raw.get("sources", [])
    )
    quality = raw.get("quality_gate", {})
    return ServerConfig(
        host=raw.get("server", {}).get("host", "127.0.0.1"),
        port=int(raw.get("server", {}).get("port", 8088)),
        knowledge_store_path=raw.get("knowledge_store", {}).get("path", "文档/笨鸟v0.1采集知识库.json"),
        rag_store_path=raw.get("knowledge_store", {}).get("rag_store_path", "文档/笨鸟v0.1文档知识库.json"),
        online_config_path=raw.get("knowledge_store", {}).get("online_config_path", "文档/笨鸟v0.1配置在线.json"),
        base_config_path=raw.get("knowledge_store", {}).get("base_config_path", "文档/笨鸟v0.1配置样例.json"),
        refresh_seconds=int(raw.get("refresh_seconds", 1800)),
        crawl_interval_seconds=int(raw.get("crawl_interval_seconds", 3600)),
        max_pages=int(raw.get("crawler", {}).get("max_pages", 200)),
        timeout_seconds=int(raw.get("crawler", {}).get("timeout_seconds", 10)),
        allow_robots=bool(raw.get("crawler", {}).get("allow_robots", True)),
        allow_redirects=bool(raw.get("crawler", {}).get("allow_redirects", True)),
        accepted_mime_prefixes=tuple(raw.get("crawler", {}).get("accepted_mime_prefixes", ["text/"])),
        sources=sources,
        quality_gate=QualityGateConfig(
            promote_to_candidate=float(quality.get("promote_to_candidate", 0.85)),
            promote_to_temporary=float(quality.get("promote_to_temporary", 0.70)),
            min_sources_for_stable=int(quality.get("min_sources_for_stable", 3)),
            conflict_penalty=float(quality.get("conflict_penalty", 0.15)),
        ),
        proactive_template_path=raw.get("proactive", {}).get("template_path", "文档/笨鸟v0.1主动提问模板.json"),
        vector_top_k=int(raw.get("vector_retrieval", {}).get("top_k", 6)),
        vector_min_score=float(raw.get("vector_retrieval", {}).get("min_score", 0.2)),
        embedding_backend=dict(raw.get("embedding_backend", {})),
        rag_top_k_docs=int(raw.get("rag", {}).get("top_k_docs", 6)),
        rag_per_query_top_k=int(raw.get("rag", {}).get("per_query_top_k", 6)),
        rag_chunk_size=int(raw.get("rag", {}).get("chunk_size", 220)),
        rag_chunk_overlap=int(raw.get("rag", {}).get("chunk_overlap", 40)),
        llm=load_llm_config(raw.get("llm", {})),
    )


def run_server(config_path: str | Path) -> None:
    config = load_server_config(config_path)
    embedding_encoder = create_embedding_encoder(config.embedding_backend)
    effective_config_path = (
        config.online_config_path if Path(config.online_config_path).exists() else config.base_config_path
    )
    snapshot = KnowledgeSnapshot(
        effective_config_path,
        config.knowledge_store_path,
        config.rag_store_path,
        embedding_encoder=embedding_encoder,
        vector_top_k=config.vector_top_k,
        vector_min_score=config.vector_min_score,
        rag_config=SBRAGConfig(
            top_k_docs=config.rag_top_k_docs,
            per_query_top_k=config.rag_per_query_top_k,
            min_vector_score=config.vector_min_score,
            chunk_size=config.rag_chunk_size,
            chunk_overlap=config.rag_chunk_overlap,
        ),
    )
    snapshot.refresh(
        effective_config_path,
        config.proactive_template_path,
        config.knowledge_store_path,
        config.rag_store_path,
    )
    ChatHandler.snapshot = snapshot
    ChatHandler.dialog_store = DialogStore()
    ChatHandler.llm_config = config.llm
    ChatHandler.llm_client = create_llm_client(config.llm)

    learner = BackgroundLearner(config, snapshot)
    learner.start()

    server = HTTPServer((config.host, config.port), ChatHandler)
    try:
        server.serve_forever()
    finally:
        learner.stop()


__all__ = ["run_server", "load_server_config", "ServerConfig"]
