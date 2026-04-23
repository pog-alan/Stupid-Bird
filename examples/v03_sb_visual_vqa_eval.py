from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import torch

from sb.core_lm_torch import runtime_device_report


DEFAULT_OUTPUT_PATH = Path("data/processed/experiments/sb_visual_vqa_eval.json")
REPO_ROOT = Path(__file__).resolve().parents[1]
VISUAL_ARCH_PATH = REPO_ROOT / "sb" / "sb-visual" / "sb_visual_architecture.py"
VISUAL_DATA_PATH = REPO_ROOT / "sb" / "sb-visual" / "sb_visual_data.py"


def _load_module(module_name: str, path: Path) -> object:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _torch_load(path: Path, device: torch.device) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _restore_qa_vocabulary(visual_data, payload: Mapping[str, Any]):
    qa_payload = payload["qa_vocabulary"]
    return visual_data.SBVisualQAVocabulary(
        question_stoi=dict(qa_payload["question_stoi"]),
        question_itos=tuple(qa_payload["question_itos"]),
        answer_stoi=dict(qa_payload["answer_stoi"]),
        answer_itos=tuple(qa_payload["answer_itos"]),
        answer_counts=dict(qa_payload.get("answer_counts", {})),
        pad_token=str(qa_payload.get("pad_token", "<pad>")),
        unk_token=str(qa_payload.get("unk_token", "<unk>")),
    )


def _restore_qa_answer_type_to_indices(payload: Mapping[str, Any]) -> Dict[str, tuple[int, ...]]:
    constraints = payload.get("qa_answer_type_to_indices", {})
    return {
        str(answer_type): tuple(int(index) for index in indices)
        for answer_type, indices in dict(constraints).items()
    }


def _resolve_output_path(path_str: str, experiment_tag: str) -> Path:
    if path_str:
        path = Path(path_str)
    elif experiment_tag:
        path = DEFAULT_OUTPUT_PATH.with_name(f"{DEFAULT_OUTPUT_PATH.stem}_{experiment_tag}{DEFAULT_OUTPUT_PATH.suffix}")
    else:
        path = DEFAULT_OUTPUT_PATH
    if not path.is_absolute():
        path = Path.cwd() / path
    return path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate SB-Visual on VQA / grounded reasoning annotations.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root", default="")
    parser.add_argument("--split", default="val")
    parser.add_argument("--annotation-manifest", default="")
    parser.add_argument("--create-synthetic", action="store_true")
    parser.add_argument("--scene-names", default="stacked,spill,orderly")
    parser.add_argument("--images-per-scene", type=int, default=4)
    parser.add_argument("--synthetic-image-size", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--sample-preview", type=int, default=12)
    parser.add_argument("--experiment-tag", default="")
    parser.add_argument("--output-path", default="")
    return parser


def _batched(items: Sequence[Any], batch_size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def _explode_qa_samples(samples) -> List[Any]:
    exploded = []
    for sample in samples:
        for qa_index, qa_pair in enumerate(sample.qa_pairs):
            metadata = dict(sample.metadata)
            metadata.update(
                {
                    "qa_index": qa_index,
                    "qa_answer_type": qa_pair.answer_type,
                    "has_relations": bool(sample.relations),
                    "has_boxes": bool(sample.boxes),
                }
            )
            exploded.append(replace(sample, qa_pairs=(qa_pair,), metadata=metadata))
    return exploded


def _safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0 else 0.0


def main() -> None:
    args = _build_parser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = Path.cwd() / checkpoint_path
    output_path = _resolve_output_path(args.output_path, args.experiment_tag)

    visual_arch = _load_module(
        "sb_visual_architecture",
        VISUAL_ARCH_PATH,
    )
    visual_data = _load_module(
        "sb_visual_data",
        VISUAL_DATA_PATH,
    )

    payload = _torch_load(checkpoint_path, device)
    qa_vocabulary = _restore_qa_vocabulary(visual_data, payload)
    qa_answer_type_to_indices = _restore_qa_answer_type_to_indices(payload)
    saved_args = payload.get("args", {})
    dataset_root = Path(args.dataset_root or saved_args.get("dataset_root", "data/processed/sb_visual_synth_train"))
    split = args.split
    scene_names = tuple(item.strip() for item in args.scene_names.split(",") if item.strip())

    if args.create_synthetic or not dataset_root.exists():
        dataset_info = visual_data.create_synthetic_scene_dataset(
            dataset_root,
            split=split,
            scene_names=scene_names,
            images_per_scene=args.images_per_scene,
            image_size=args.synthetic_image_size,
            include_annotations=True,
        )
    else:
        annotation_manifest = Path(args.annotation_manifest) if args.annotation_manifest else dataset_root / f"{split}_annotations.jsonl"
        dataset_info = visual_data.discover_visual_samples(
            dataset_root,
            split=split,
            annotation_manifest=annotation_manifest if annotation_manifest.exists() else None,
        )

    samples = dataset_info["samples"]
    if not qa_answer_type_to_indices:
        qa_answer_type_to_indices = visual_data.build_qa_answer_type_constraints(
            samples,
            qa_vocabulary=qa_vocabulary,
        )
    qa_samples = _explode_qa_samples(samples)
    if args.max_samples > 0:
        qa_samples = qa_samples[: args.max_samples]
    if not qa_samples:
        raise RuntimeError("no QA samples found for SB-Visual VQA evaluation.")

    relation_predicate_to_index = dict(payload.get("relation_predicate_to_index", {"<none>": 0}))
    scene_to_index = dataset_info["scene_to_index"]
    scene_classes = max((index for index in scene_to_index.values() if index >= 0), default=-1) + 1
    model_config_kwargs = {
        "image_size": int(saved_args.get("image_size", 128)),
        "patch_size": int(saved_args.get("patch_size", 16)),
        "patch_stride": int(saved_args.get("patch_stride", 16)),
        "d_model": int(saved_args.get("d_model", 96)),
        "state_dim": int(saved_args.get("state_dim", 96)),
        "schema_slots": int(saved_args.get("schema_slots", 9)),
        "object_memory_slots": int(saved_args.get("object_memory_slots", 16)),
        "relation_memory_slots": int(saved_args.get("relation_memory_slots", 24)),
        "scene_memory_slots": int(saved_args.get("scene_memory_slots", 6)),
        "summary_memory_slots": int(saved_args.get("summary_memory_slots", 6)),
        "relation_vocab_size": max(len(relation_predicate_to_index), 1),
        "scene_classes": max(scene_classes, 1),
        "answer_vocab_size": max(qa_vocabulary.answer_vocab_size, 1),
        "question_vocab_size": max(qa_vocabulary.question_vocab_size, 2),
        "question_max_len": int(saved_args.get("question_max_len", 48)),
    }
    if "model_config" in payload:
        model_config_kwargs.update(payload["model_config"])

    model = visual_arch.SBVisualCore(visual_arch.SBVisualConfig(**model_config_kwargs)).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()

    data_config = visual_data.SBVisualDatasetConfig(image_size=model.config.image_size)
    sample_predictions: List[Dict[str, Any]] = []
    answer_type_stats: Dict[str, Dict[str, float]] = {}
    total = 0.0
    correct = 0.0
    grounded_total = 0.0
    grounded_correct = 0.0
    relation_grounded_total = 0.0
    relation_grounded_correct = 0.0
    summary_alignment_sum = 0.0
    scene_alignment_sum = 0.0
    schema_chain_count = 0.0
    batch_count = 0.0

    with torch.no_grad():
        for chunk in _batched(qa_samples, max(1, args.batch_size)):
            batch = visual_data.build_visual_batch(chunk, config=data_config, device=device)
            qa_targets = visual_data.build_qa_targets(
                batch,
                qa_vocabulary=qa_vocabulary,
                max_question_len=model.config.question_max_len,
                answer_type_to_indices=qa_answer_type_to_indices,
                device=device,
            )
            forward = model(
                batch.images,
                question_ids=qa_targets.question_ids,
                question_mask=qa_targets.question_mask,
                return_aux=True,
            )
            masked_answer_logits = forward["answer_logits"]
            if qa_targets.answer_type_mask.numel() > 0:
                masked_answer_logits = masked_answer_logits.masked_fill(qa_targets.answer_type_mask <= 0.5, -1e9)
            pred_ids = masked_answer_logits.argmax(dim=-1)
            summary_alignment = float(forward["aux"]["summary_schema_alignment_mean"])
            scene_alignment = float(forward["aux"]["scene_schema_alignment_mean"])
            summary_alignment_sum += summary_alignment
            scene_alignment_sum += scene_alignment
            schema_chain_count += float((summary_alignment > 0.0) or (scene_alignment > 0.0))
            batch_count += 1.0

            for index, sample in enumerate(chunk):
                if float(qa_targets.answer_valid_mask[index].item()) <= 0.5:
                    continue
                total += 1.0
                gold_id = int(qa_targets.answer_labels[index].item())
                pred_id = int(pred_ids[index].item())
                is_correct = float(pred_id == gold_id)
                correct += is_correct

                qa_pair = sample.qa_pairs[0]
                answer_type = qa_pair.answer_type or "unknown"
                answer_type_stats.setdefault(answer_type, {"total": 0.0, "correct": 0.0})
                answer_type_stats[answer_type]["total"] += 1.0
                answer_type_stats[answer_type]["correct"] += is_correct

                has_boxes = bool(sample.boxes)
                has_relations = bool(sample.relations)
                if has_boxes:
                    grounded_total += 1.0
                    grounded_correct += is_correct
                if has_relations:
                    relation_grounded_total += 1.0
                    relation_grounded_correct += is_correct

                if len(sample_predictions) < max(0, args.sample_preview):
                    sample_predictions.append(
                        {
                            "image_path": sample.image_path,
                            "scene_name": sample.scene_name,
                            "question": qa_pair.question,
                            "gold_answer": qa_pair.answer,
                            "predicted_answer": qa_vocabulary.answer_itos[pred_id] if pred_id < len(qa_vocabulary.answer_itos) else "<out_of_range>",
                            "answer_type": answer_type,
                            "correct": bool(is_correct),
                            "has_boxes": has_boxes,
                            "has_relations": has_relations,
                        }
                    )

    answer_type_accuracy = {
        answer_type: {
            "count": int(stats["total"]),
            "accuracy": _safe_ratio(stats["correct"], stats["total"]),
        }
        for answer_type, stats in sorted(answer_type_stats.items())
    }
    mean_summary_schema_alignment = _safe_ratio(summary_alignment_sum, batch_count)
    mean_scene_schema_alignment = _safe_ratio(scene_alignment_sum, batch_count)
    schema_chain_activated = bool(mean_summary_schema_alignment > 0.0 or mean_scene_schema_alignment > 0.0)

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "runtime": runtime_device_report(device),
        "checkpoint": str(checkpoint_path.resolve()),
        "dataset": {
            "root": str(dataset_root.resolve()),
            "split": dataset_info["split"],
            "sample_count": len(samples),
            "qa_sample_count": len(qa_samples),
            "annotation_manifest": dataset_info.get("annotation_manifest", ""),
        },
        "metrics": {
            "overall_answer_acc": _safe_ratio(correct, total),
            "grounded_answer_acc": _safe_ratio(grounded_correct, grounded_total),
            "relation_grounded_answer_acc": _safe_ratio(relation_grounded_correct, relation_grounded_total),
            "mean_summary_schema_alignment": mean_summary_schema_alignment,
            "mean_scene_schema_alignment": mean_scene_schema_alignment,
            "schema_chain_activated": schema_chain_activated,
            "schema_chain_activated_ratio": _safe_ratio(schema_chain_count, batch_count),
        },
        "answer_type_accuracy": answer_type_accuracy,
        "sample_predictions": sample_predictions,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"report_path={output_path}")


if __name__ == "__main__":
    main()
