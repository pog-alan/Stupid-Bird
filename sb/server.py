from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .auto_crawl import CrawlSource, Crawler, CrawlerConfig
from .dialog import DialogStore
from .embedding_backends import EmbeddingEncoder, create_embedding_encoder
from .extractor import SimpleExtractor, merge_candidates
from .ingest import Ingestor, KnowledgeEntry, KnowledgeStore, apply_stable_entries_to_config
from .ontology import load_default_ontology
from .reasoner import SBV01Engine
from .quality_gate import QualityGateConfig


@dataclass
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 8088
    knowledge_store_path: str = "文档/笨鸟v0.1采集知识库.json"
    online_config_path: str = "文档/笨鸟v0.1配置在线.json"
    base_config_path: str = "文档/笨鸟v0.1配置样例.json"
    refresh_seconds: int = 1800
    crawl_interval_seconds: int = 3600
    max_pages: int = 200
    timeout_seconds: int = 10
    allow_robots: bool = True
    allow_redirects: bool = True
    accepted_mime_prefixes: Tuple[str, ...] = ("text/", "application/json")
    sources: Tuple[CrawlSource, ...] = ()
    quality_gate: QualityGateConfig = QualityGateConfig()
    proactive_template_path: str = "文档/笨鸟v0.1主动提问模板.json"
    vector_top_k: int = 6
    vector_min_score: float = 0.2
    embedding_backend: Dict[str, object] = field(default_factory=dict)


class KnowledgeSnapshot:
    def __init__(
        self,
        config_path: str,
        knowledge_store_path: Optional[str] = None,
        embedding_encoder: EmbeddingEncoder | None = None,
        vector_top_k: int = 6,
        vector_min_score: float = 0.2,
    ) -> None:
        self._lock = threading.Lock()
        self._config_path = config_path
        self._knowledge_store_path = knowledge_store_path
        self._embedding_encoder = embedding_encoder
        self._vector_top_k = vector_top_k
        self._vector_min_score = vector_min_score
        self._ontology = load_default_ontology(config_path)
        stable_entries = self._load_stable_entries()
        self._engine = SBV01Engine(
            self._ontology,
            stable_entries=stable_entries,
            embedding_encoder=self._embedding_encoder,
            vector_top_k=self._vector_top_k,
            vector_min_score=self._vector_min_score,
        )
        self._updated_at = datetime.now(timezone.utc)
        self._proactive_template_path: Optional[str] = None

    def refresh(
        self,
        config_path: Optional[str] = None,
        proactive_template_path: Optional[str] = None,
        knowledge_store_path: Optional[str] = None,
    ) -> None:
        with self._lock:
            if config_path is not None:
                self._config_path = config_path
            if proactive_template_path is not None:
                self._proactive_template_path = proactive_template_path
            if knowledge_store_path is not None:
                self._knowledge_store_path = knowledge_store_path
            self._ontology = load_default_ontology(self._config_path)
            stable_entries = self._load_stable_entries()
            self._engine = SBV01Engine(
                self._ontology,
                stable_entries=stable_entries,
                embedding_encoder=self._embedding_encoder,
                vector_top_k=self._vector_top_k,
                vector_min_score=self._vector_min_score,
            )
            self._updated_at = datetime.now(timezone.utc)

    def analyze(self, text: str) -> Dict[str, object]:
        with self._lock:
            result = self._engine.analyze(text)
            result["retrieved_memories"] = self._engine.retrieve_memories(text, result, top_k=6)
            if self._proactive_template_path:
                from .proactive import propose_questions

                questions = propose_questions(result, self._proactive_template_path)
                result["proactive_questions"] = [
                    {"question": item.question, "reason": item.reason, "priority": item.priority}
                    for item in questions[:2]
                ]
            return result

    def build_llm_payload(self, text: str, analysis: Dict[str, object]) -> Dict[str, object]:
        with self._lock:
            return self._engine.build_llm_payload(text, analysis)

    def status(self) -> Dict[str, str]:
        with self._lock:
            return {
                "config_path": self._config_path,
                "knowledge_store_path": self._knowledge_store_path or "",
                "embedding_backend": getattr(self._embedding_encoder, "backend_name", "unknown"),
                "embedding_model": getattr(self._embedding_encoder, "model_name", ""),
                "updated_at": self._updated_at.isoformat(),
            }

    def _load_stable_entries(self) -> List[KnowledgeEntry]:
        if not self._knowledge_store_path:
            return []
        store_path = Path(self._knowledge_store_path)
        if not store_path.exists():
            return []
        return KnowledgeStore.load(store_path).stable_entries()


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

        candidates = []
        for result in crawler.crawl():
            if result.error or not result.text:
                continue
            candidates.extend(extractor.extract(result.text, source_url=result.url))
        if not candidates:
            return
        merged = merge_candidates(candidates)
        ingestor.ingest(merged)
        store.save()

        output_path = apply_stable_entries_to_config(
            store,
            config_path=self.config.base_config_path,
            output_path=self.config.online_config_path,
        )
        self.snapshot.refresh(
            str(output_path),
            self.config.proactive_template_path,
            self.config.knowledge_store_path,
        )


class ChatHandler(BaseHTTPRequestHandler):
    snapshot: KnowledgeSnapshot
    dialog_store: DialogStore
    allow_origin: str = "*"

    def _send_json(self, status: int, payload: Dict[str, object]) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
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
        if self.path == "/status":
            self._send_json(200, self.snapshot.status())
            return
        self._send_json(404, {"error": "Unknown endpoint"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path not in {"/chat", "/llm_context"}:
            self._send_json(404, {"error": "Unknown endpoint"})
            return
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length).decode("utf-8")
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            self._send_json(400, {"error": "Invalid JSON"})
            return
        text = (payload.get("text") or "").strip()
        session_id = (payload.get("session_id") or "default").strip()
        if not text:
            self._send_json(400, {"error": "Missing text"})
            return
        context_text = self.dialog_store.build_context_text(session_id, text)
        result = self.snapshot.analyze(context_text)
        questions = [item["question"] for item in result.get("proactive_questions", [])]
        state = self.dialog_store.update(session_id=session_id, text=text, questions=questions)
        result["dialog_state"] = {
            "session_id": state.session_id,
            "turns": len(state.turns),
            "last_questions": state.last_questions,
            "updated_at": state.updated_at,
        }
        if context_text != text:
            result["context_text"] = context_text
        if self.path == "/chat":
            self._send_json(200, result)
            return

        llm_payload = self.snapshot.build_llm_payload(context_text, result)
        llm_payload["dialog_state"] = result["dialog_state"]
        if "context_text" in result:
            llm_payload["context_text"] = result["context_text"]
        self._send_json(200, llm_payload)


def load_server_config(path: str | Path) -> ServerConfig:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
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
        knowledge_store_path=raw.get("knowledge_store", {}).get(
            "path", "文档/笨鸟v0.1采集知识库.json"
        ),
        online_config_path=raw.get("knowledge_store", {}).get(
            "online_config_path", "文档/笨鸟v0.1配置在线.json"
        ),
        base_config_path=raw.get("knowledge_store", {}).get(
            "base_config_path", "文档/笨鸟v0.1配置样例.json"
        ),
        refresh_seconds=int(raw.get("refresh_seconds", 1800)),
        crawl_interval_seconds=int(raw.get("crawl_interval_seconds", 3600)),
        max_pages=int(raw.get("crawler", {}).get("max_pages", 200)),
        timeout_seconds=int(raw.get("crawler", {}).get("timeout_seconds", 10)),
        allow_robots=bool(raw.get("crawler", {}).get("allow_robots", True)),
        allow_redirects=bool(raw.get("crawler", {}).get("allow_redirects", True)),
        accepted_mime_prefixes=tuple(
            raw.get("crawler", {}).get("accepted_mime_prefixes", ["text/"])
        ),
        sources=sources,
        quality_gate=QualityGateConfig(
            promote_to_candidate=float(quality.get("promote_to_candidate", 0.85)),
            promote_to_temporary=float(quality.get("promote_to_temporary", 0.70)),
            min_sources_for_stable=int(quality.get("min_sources_for_stable", 3)),
            conflict_penalty=float(quality.get("conflict_penalty", 0.15)),
        ),
        proactive_template_path=raw.get("proactive", {}).get(
            "template_path", "文档/笨鸟v0.1主动提问模板.json"
        ),
        vector_top_k=int(raw.get("vector_retrieval", {}).get("top_k", 6)),
        vector_min_score=float(raw.get("vector_retrieval", {}).get("min_score", 0.2)),
        embedding_backend=dict(raw.get("embedding_backend", {})),
    )


def run_server(config_path: str | Path) -> None:
    config = load_server_config(config_path)
    embedding_encoder = create_embedding_encoder(config.embedding_backend)
    snapshot = KnowledgeSnapshot(
        config.base_config_path,
        config.knowledge_store_path,
        embedding_encoder=embedding_encoder,
        vector_top_k=config.vector_top_k,
        vector_min_score=config.vector_min_score,
    )
    snapshot.refresh(
        config.base_config_path,
        config.proactive_template_path,
        config.knowledge_store_path,
    )
    ChatHandler.snapshot = snapshot
    ChatHandler.dialog_store = DialogStore()

    learner = BackgroundLearner(config, snapshot)
    learner.start()

    server = HTTPServer((config.host, config.port), ChatHandler)
    try:
        server.serve_forever()
    finally:
        learner.stop()


__all__ = ["run_server", "load_server_config", "ServerConfig"]
