from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import torch

from sb.core_lm_torch import runtime_device_report

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_visual_module() -> object:
    module_path = REPO_ROOT / "sb" / "sb-visual" / "sb_visual_architecture.py"
    spec = importlib.util.spec_from_file_location("sb_visual_architecture", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    module = _load_visual_module()
    config = module.SBVisualConfig(
        image_size=128,
        patch_size=16,
        patch_stride=16,
        d_model=96,
        state_dim=96,
        schema_slots=9,
        object_memory_slots=16,
        relation_memory_slots=24,
        scene_memory_slots=6,
        summary_memory_slots=6,
        scene_classes=8,
        answer_vocab_size=64,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = module.SBVisualCore(config).to(device)
    image = torch.randn(2, 3, config.image_size, config.image_size, device=device)
    text_query = torch.randn(2, config.state_dim, device=device)
    with torch.no_grad():
        forward = model(image, text_query=text_query, return_state=True, return_aux=True)

    result = {
        "runtime": runtime_device_report(device),
        "config": {
            "image_size": config.image_size,
            "patch_size": config.patch_size,
            "state_dim": config.state_dim,
            "schema_slots": config.schema_slots,
            "object_memory_slots": config.object_memory_slots,
            "relation_memory_slots": config.relation_memory_slots,
            "scene_memory_slots": config.scene_memory_slots,
            "summary_memory_slots": config.summary_memory_slots,
        },
        "outputs": {
            "scene_logits_shape": list(forward["scene_logits"].shape),
            "object_scores_shape": list(forward["object_scores"].shape),
            "relation_scores_shape": list(forward["relation_scores"].shape),
            "answer_logits_shape": list(forward["answer_logits"].shape),
            "schema_weights_shape": list(forward["schema_weights"].shape),
        },
        "aux": forward["aux"],
        "state_present": "state" in forward,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
