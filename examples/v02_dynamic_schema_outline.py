from __future__ import annotations

import json

import torch

from sb.core_lm_torch import SBCoreMiniLM, SBCoreMiniTorchConfig, runtime_device_report


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(13)

    config = SBCoreMiniTorchConfig(
        vocab_size=32,
        d_model=32,
        state_dim=48,
        num_layers=1,
        semantic_memory_slots=8,
        working_memory_slots=6,
        episodic_memory_slots=4,
        episodic_key_slots=2,
        summary_memory_slots=2,
        scene_memory_slots=1,
        router_top_k=2,
        max_seq_len=24,
        signal_schema_slots=7,
        dropout=0.0,
    )
    model = SBCoreMiniLM(config).to(device)

    inputs = torch.randint(low=0, high=config.vocab_size, size=(2, 12), device=device)
    with torch.no_grad():
        output = model(inputs)

    payload = {
        "runtime": runtime_device_report(device),
        "config": {
            "signal_schema_slots": config.signal_schema_slots,
            "signal_abstraction_levels": config.signal_abstraction_levels,
            "signal_stop_threshold": config.signal_stop_threshold,
        },
        "schema_probe": {
            "abstraction_entropy_mean": output["aux"]["abstraction_entropy_mean"],
            "abstraction_anchor_entropy_mean": output["aux"]["abstraction_anchor_entropy_mean"],
            "abstraction_schema_active_ratio_mean": output["aux"]["abstraction_schema_active_ratio_mean"],
            "abstraction_schema_peak_mean": output["aux"]["abstraction_schema_peak_mean"],
            "abstraction_schema_widen_mean": output["aux"]["abstraction_schema_widen_mean"],
            "abstraction_schema_narrow_mean": output["aux"]["abstraction_schema_narrow_mean"],
            "abstraction_schema_split_mean": output["aux"]["abstraction_schema_split_mean"],
            "abstraction_schema_merge_mean": output["aux"]["abstraction_schema_merge_mean"],
            "abstraction_schema_suspend_mean": output["aux"]["abstraction_schema_suspend_mean"],
            "abstraction_schema_temperature_mean": output["aux"]["abstraction_schema_temperature_mean"],
            "abstraction_entity_weight_mean": output["aux"]["abstraction_entity_weight_mean"],
            "abstraction_relation_weight_mean": output["aux"]["abstraction_relation_weight_mean"],
            "abstraction_event_weight_mean": output["aux"]["abstraction_event_weight_mean"],
            "episodic_replay_schema_alignment_mean": output["aux"]["episodic_replay_schema_alignment_mean"],
            "episodic_replay_branch_alignment_mean": output["aux"]["episodic_replay_branch_alignment_mean"],
        },
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
