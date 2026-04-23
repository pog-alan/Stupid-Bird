from __future__ import annotations

import importlib.util
import json
import random
import sys
import tempfile
from pathlib import Path

import torch

from sb.core_lm_torch import runtime_device_report

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module(module_name: str, path: Path) -> object:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    visual_arch = _load_module(
        "sb_visual_architecture",
        REPO_ROOT / "sb" / "sb-visual" / "sb_visual_architecture.py",
    )
    visual_data = _load_module(
        "sb_visual_data",
        REPO_ROOT / "sb" / "sb-visual" / "sb_visual_data.py",
    )

    with tempfile.TemporaryDirectory(prefix="sb_visual_smoke_") as temp_dir:
        dataset_info = visual_data.create_synthetic_scene_dataset(
            temp_dir,
            split="train",
            scene_names=("stacked", "spill", "orderly"),
            images_per_scene=3,
            image_size=96,
        )
        samples = dataset_info["samples"]
        data_config = visual_data.SBVisualDatasetConfig(image_size=128)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        batch = visual_data.sample_visual_batch(
            samples,
            batch_size=3,
            config=data_config,
            device=device,
            rng=random.Random(11),
        )

        model_config = visual_arch.SBVisualConfig(
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
            scene_classes=3,
            answer_vocab_size=48,
        )
        model = visual_arch.SBVisualCore(model_config).to(device)
        patch_payload = visual_data.encode_visual_batch(batch, model.patch_encoder)
        text_query = torch.randn(batch.images.shape[0], model_config.state_dim, device=device)
        with torch.no_grad():
            forward = model(batch.images, text_query=text_query, return_state=True, return_aux=True)

        result = {
            "runtime": runtime_device_report(device),
            "dataset": {
                "root": dataset_info["root"],
                "split": dataset_info["split"],
                "scene_to_index": dataset_info["scene_to_index"],
                "sample_count": len(samples),
                "annotation_manifest": dataset_info.get("annotation_manifest", ""),
            },
            "batch": {
                "images_shape": list(batch.images.shape),
                "scene_labels": batch.scene_labels.tolist(),
                "scene_names": list(batch.scene_names),
                "box_counts": [int(item.shape[0]) for item in batch.boxes_xyxy],
                "relation_counts": [len(item) for item in batch.relations],
                "qa_counts": [len(item) for item in batch.qa_pairs],
                "first_box_labels": list(batch.box_labels[0]) if batch.box_labels else [],
            },
            "patch_encoding": {
                "signals_shape": list(patch_payload["signals"].shape),
                "positions_shape": list(patch_payload["positions"].shape),
                "patch_grid": list(patch_payload["patch_grid"]),
                "encoded_box_counts": [int(item.shape[0]) for item in patch_payload["boxes_xyxy"]],
            },
            "forward": {
                "scene_logits_shape": list(forward["scene_logits"].shape),
                "object_scores_shape": list(forward["object_scores"].shape),
                "relation_scores_shape": list(forward["relation_scores"].shape),
                "schema_weights_shape": list(forward["schema_weights"].shape),
                "aux": forward["aux"],
                "state_present": "state" in forward,
            },
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
