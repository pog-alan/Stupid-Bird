from __future__ import annotations

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from sb import LLMConfig, OpenAICompatibleLLMClient, SBRAGConfig, SBRAGPipeline, SBV01Engine, load_default_ontology
from sb.rag_store import RAGKnowledgeBase


class MockLLMHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length).decode("utf-8")
        payload = json.loads(body)
        messages = payload.get("messages", [])
        user_message = messages[-1]["content"] if messages else ""
        response = {
            "id": "mock-chatcmpl-001",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": json.dumps(
                            {
                                "answer": "从现有证据看，更像建筑废弃物堆放场景。",
                                "grounded_facts": ["识别到碎砖、木板、小卡车和裸露地面"],
                                "uncertainties": [],
                                "follow_up_question": None,
                                "user_prompt_preview": user_message[:80],
                            },
                            ensure_ascii=False,
                        ),
                    },
                }
            ],
            "usage": {"prompt_tokens": 128, "completion_tokens": 42, "total_tokens": 170},
        }
        data = json.dumps(response, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A003
        return


def main() -> None:
    server = HTTPServer(("127.0.0.1", 18081), MockLLMHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        os.environ["MIMO_API_KEY"] = "demo-key"

        ontology = load_default_ontology()
        engine = SBV01Engine(ontology)
        knowledge_base = RAGKnowledgeBase(path="文档/笨鸟v0.1文档知识库.demo.json")
        pipeline = SBRAGPipeline(
            engine,
            knowledge_base,
            config=SBRAGConfig(top_k_docs=3, per_query_top_k=3, chunk_size=120, chunk_overlap=20),
        )
        pipeline.ingest_documents(
            [
                {
                    "source_url": "https://demo.local/construction",
                    "title": "施工堆放说明",
                    "text": "屋后空地、碎砖、木板、小卡车和裸露地面同时出现时，常见于建筑废弃物堆放场景。",
                }
            ]
        )

        result = pipeline.query("屋后空地上堆着碎砖和木板，旁边停着一辆小卡车，地面裸露。")
        client = OpenAICompatibleLLMClient(
            LLMConfig(
                enabled=True,
                base_url="http://127.0.0.1:18081/v1",
                model="mimo-v2-pro",
                api_key_env="MIMO_API_KEY",
                include_response_format=True,
            )
        )
        generated = client.generate(result["llm_packet"])

        print("=== 回答草案 ===")
        print(json.dumps(result["answer_draft"], ensure_ascii=False, indent=2))
        print()
        print("=== 模型输出 ===")
        print(json.dumps(generated, ensure_ascii=False, indent=2))
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
