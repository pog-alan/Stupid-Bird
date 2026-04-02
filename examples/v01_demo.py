from __future__ import annotations

import json

from sb import SBV01Engine


EXAMPLES = [
    "屋后空地上堆着碎砖和木板，旁边停着一辆小卡车，地面裸露，没有整齐堆放。",
    "桌子左边有一个红色杯子，杯子里有水，猫碰到了杯子，杯子翻倒了，水洒在桌面上。",
    "路边沟渠里有一片发黑的积水，周围有杂草，靠近居民房。",
]


def main() -> None:
    engine = SBV01Engine.from_default_config()
    for index, text in enumerate(EXAMPLES, start=1):
        print(f"=== 示例 {index} ===")
        print(text)
        result = engine.analyze(text)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print()


if __name__ == "__main__":
    main()
