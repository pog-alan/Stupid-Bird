from __future__ import annotations

import json

import torch

from sb import SBCoreMiniLM, SBCoreMiniTorchConfig, SBCoreStateCache, SBStateCacheConfig


def main() -> None:
    torch.manual_seed(7)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = SBCoreMiniTorchConfig(
        vocab_size=32,
        d_model=16,
        state_dim=24,
        num_layers=1,
        semantic_memory_slots=8,
        working_memory_slots=4,
        episodic_memory_slots=2,
        episodic_key_slots=2,
        summary_memory_slots=2,
        scene_memory_slots=1,
        router_top_k=2,
        max_seq_len=8,
        dropout=0.0,
    )
    model = SBCoreMiniLM(config).to(device)
    model.eval()

    cache = SBCoreStateCache(SBStateCacheConfig(max_sessions=2, token_history_limit=16))
    first_prompt = torch.tensor([[1, 2, 3, 4]], dtype=torch.long, device=device)
    extended_prompt = torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.long, device=device)

    with torch.no_grad():
        first = cache.advance_from_prompt(model, first_prompt, session_id="demo", stage_name="smoke")
        second = cache.advance_from_prompt(model, extended_prompt, session_id="demo", stage_name="smoke")

    assert first.computed_tokens == 4
    assert first.cache_hit is False
    assert second.cache_hit is True
    assert second.reused_tokens == 4
    assert second.computed_tokens == 2
    assert second.output is not None
    assert tuple(second.metadata["last_aux"].keys())

    print(
        json.dumps(
            {
                "device": str(device),
                "first": {
                    "computed_tokens": first.computed_tokens,
                    "reused_tokens": first.reused_tokens,
                    "cache_hit": first.cache_hit,
                    "reset_reason": first.reset_reason,
                },
                "second": {
                    "computed_tokens": second.computed_tokens,
                    "reused_tokens": second.reused_tokens,
                    "cache_hit": second.cache_hit,
                    "reset_reason": second.reset_reason,
                },
                "cache": cache.stats(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
