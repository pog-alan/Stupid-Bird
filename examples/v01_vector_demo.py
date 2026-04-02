from __future__ import annotations

import json

from sb import SBV01Engine


def main() -> None:
    engine = SBV01Engine.from_default_config()
    text = "空地上乱堆着砖头和木料，旁边还有货车。"
    analysis = engine.analyze(text)
    llm_payload = engine.build_llm_payload(text, analysis)

    print("=== 结构化分析 ===")
    print(json.dumps(analysis, ensure_ascii=False, indent=2))
    print()
    print("=== LLM 上下文 ===")
    print(json.dumps(llm_payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
