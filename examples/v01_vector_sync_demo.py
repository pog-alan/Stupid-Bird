from __future__ import annotations

from pathlib import Path

from sb import KnowledgeStore, SBV01Engine, load_default_ontology


def main() -> None:
    store_path = Path("文档") / "笨鸟v0.1采集知识库.json"
    ontology = load_default_ontology()
    stable_entries = KnowledgeStore.load(store_path).stable_entries() if store_path.exists() else []
    engine = SBV01Engine(ontology, stable_entries=stable_entries)

    analysis = engine.analyze("院角堆着几包杂物，边上放着旧纸箱。")
    memories = engine.retrieve_memories("院角堆着几包杂物，边上放着旧纸箱。", analysis)

    print("=== 稳定知识条目数量 ===")
    print(len(stable_entries))
    print("=== 检索记忆 ===")
    for item in memories:
        print(f"{item['label']} | {item['space_type']} | {item['score']}")


if __name__ == "__main__":
    main()
