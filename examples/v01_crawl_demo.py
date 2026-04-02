from __future__ import annotations

import json
from pathlib import Path

from sb import (
    CrawlSource,
    Crawler,
    CrawlerConfig,
    Ingestor,
    KnowledgeStore,
    SimpleExtractor,
    apply_stable_entries_to_config,
    load_default_ontology,
)


def main() -> None:
    config_path = Path("文档") / "笨鸟v0.1采集配置样例.json"
    if not config_path.exists():
        raise SystemExit("缺少采集配置样例文件。")

    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    sources = [
        CrawlSource(
            name=item["name"],
            seed_urls=tuple(item["seed_urls"]),
            allowed_domains=tuple(item["allowed_domains"]),
            max_depth=item.get("max_depth", 1),
            rate_limit_seconds=item.get("rate_limit_seconds", 1.0),
            user_agent=item.get("user_agent", "SB-V01-Crawler/0.1"),
        )
        for item in cfg["sources"]
    ]
    crawler = Crawler(
        CrawlerConfig(
            sources=tuple(sources),
            max_pages=cfg["crawler"].get("max_pages", 200),
            timeout_seconds=cfg["crawler"].get("timeout_seconds", 10),
            allow_robots=cfg["crawler"].get("allow_robots", True),
            allow_redirects=cfg["crawler"].get("allow_redirects", True),
            accepted_mime_prefixes=tuple(cfg["crawler"].get("accepted_mime_prefixes", ["text/"])),
        )
    )

    ontology = load_default_ontology()
    extractor = SimpleExtractor(ontology)
    store = KnowledgeStore.load(cfg["knowledge_store"]["path"])
    ingestor = Ingestor(store)

    candidates = []
    for result in crawler.crawl():
        if result.error or not result.text:
            continue
        candidates.extend(extractor.extract(result.text, source_url=result.url))

    report = ingestor.ingest(candidates)
    store.save()

    apply_stable_entries_to_config(
        store,
        config_path=Path("文档") / "笨鸟v0.1配置样例.json",
    )

    print(f"入库更新: {len(report.updated)}")
    print(f"晋级稳定: {len(report.promoted)}")
    print(f"丢弃: {len(report.dropped)}")


if __name__ == "__main__":
    main()
