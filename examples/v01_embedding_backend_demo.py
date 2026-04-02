from __future__ import annotations

import json
from pathlib import Path

from sb import SBV01Engine, create_embedding_encoder, load_default_ontology, resolve_embedding_backend_config


def main() -> None:
    ontology = load_default_ontology()
    config_path = Path("文档") / "笨鸟v0.1本地句向量配置样例.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    embedding_config = resolve_embedding_backend_config(config["embedding_backend"])

    encoder = create_embedding_encoder(embedding_config)

    engine = SBV01Engine(
        ontology,
        embedding_encoder=encoder,
        vector_top_k=6,
        vector_min_score=0.2,
    )

    analysis = engine.analyze("路边沟渠里有一片发黑的积水，靠近居民房。")
    memories = engine.retrieve_memories("路边沟渠里有一片发黑的积水，靠近居民房。", analysis)

    print("嵌入后端:", getattr(encoder, "backend_name", "unknown"))
    print("模型名称:", embedding_config.get("model_name", ""))
    print("命中记忆数量:", len(memories))
    for item in memories[:5]:
        print(f"{item['label']} | {item['space_type']} | {item['score']}")


if __name__ == "__main__":
    main()
