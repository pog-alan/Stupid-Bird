from __future__ import annotations

import json

from sb import CORE_OBJECTIVE_NON_ATTENTION_RECALL, SBCoreConfig, SBCoreModelSpec


def main() -> None:
    spec = SBCoreModelSpec(
        SBCoreConfig(
            vocab_size=128,
            d_model=64,
            num_layers=2,
            state_dim=96,
            memory_slots=128,
            router_top_k=4,
            recall_horizon=2048,
        )
    )
    summary = spec.formal_summary()

    assert summary["config"]["objective"] == CORE_OBJECTIVE_NON_ATTENTION_RECALL
    assert summary["config"]["use_attention"] is False
    assert summary["config"]["use_kv_cache"] is False
    assert "recall_read" in summary["math"]["equations"]
    assert "self_attention" in summary["math"]["forbidden_operations"]
    assert "O(T^2 d)" in summary["math"]["equations"]["complexity"]

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
