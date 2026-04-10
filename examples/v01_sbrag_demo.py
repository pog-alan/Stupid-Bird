from __future__ import annotations

import json
from pathlib import Path

from sb import SBRAGConfig, SBRAGPipeline, SBV01Engine, load_default_ontology
from sb.rag_store import RAGKnowledgeBase


def main() -> None:
    ontology = load_default_ontology()
    engine = SBV01Engine(ontology)
    store_path = Path("文档") / "笨鸟v0.1文档知识库.demo.json"
    knowledge_base = RAGKnowledgeBase(path=store_path)
    pipeline = SBRAGPipeline(
        engine,
        knowledge_base,
        config=SBRAGConfig(top_k_docs=4, per_query_top_k=4, chunk_size=120, chunk_overlap=20),
    )

    pipeline.ingest_documents(
        [
            {
                "source_url": "https://demo.local/construction",
                "title": "建筑废弃物堆放巡查要点",
                "text": (
                    "屋后空地、裸土地面、碎砖木板混合堆放，旁边出现小卡车或施工痕迹时，"
                    "通常要优先排查建筑废弃物堆放或临时施工材料堆放场景。"
                ),
                "metadata": {"source_type": "manual_demo"},
            },
            {
                "source_url": "https://demo.local/clutter",
                "title": "生活杂物堆放识别提示",
                "text": (
                    "塑料袋、旧纸箱、生活杂物、翻倒垃圾桶、散乱地面经常共同出现，"
                    "更偏向生活杂物堆放或倾倒事件后的局部异常场景。"
                ),
                "metadata": {"source_type": "manual_demo"},
            },
            {
                "source_url": "https://demo.local/water",
                "title": "异常积水排查指引",
                "text": (
                    "沟渠、发黑积水、靠近居民房、周围杂草较多时，"
                    "通常需要判断为污染或异常积水场景，并优先关注污染风险。"
                ),
                "metadata": {"source_type": "manual_demo"},
            },
        ]
    )

    query = "屋后空地上堆着碎砖和木板，旁边停着一辆小卡车，地面裸露，没有整齐堆放。"
    result = pipeline.query(query)

    print("=== SB-RAG 分析 ===")
    print(json.dumps(result["analysis"], ensure_ascii=False, indent=2))
    print()
    print("=== 命中文档 ===")
    print(json.dumps(result["retrieved_documents"], ensure_ascii=False, indent=2))
    print()
    print("=== 回答草案 ===")
    print(json.dumps(result["answer_draft"], ensure_ascii=False, indent=2))
    print()
    print("=== LLM 上下文包 ===")
    print(json.dumps(result["llm_packet"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
