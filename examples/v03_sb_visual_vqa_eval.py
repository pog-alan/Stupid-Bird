from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import torch

from sb.core_lm_torch import runtime_device_report


DEFAULT_OUTPUT_PATH = Path("data/processed/experiments/sb_visual_vqa_eval.json")
REPO_ROOT = Path(__file__).resolve().parents[1]
VISUAL_ARCH_PATH = REPO_ROOT / "sb" / "sb-visual" / "sb_visual_architecture.py"
VISUAL_DATA_PATH = REPO_ROOT / "sb" / "sb-visual" / "sb_visual_data.py"
RELATION_TEXT_TO_PREDICATE = (
    ("overlapping", "overlaps"),
    ("overlaps", "overlaps"),
    ("left of", "left_of"),
    ("right of", "right_of"),
    ("above", "above"),
    ("below", "below"),
    ("near", "near"),
    ("next to", "near"),
)
RELATION_GRAPH_SCORE_MODES = ("product", "logit_add", "relation_only", "predicate_only")


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
    parser.add_argument(
        "--relation-answer-mode",
        choices=("qa", "graph", "hybrid"),
        default="hybrid",
        help=(
            "qa keeps the raw QA head answer; graph answers relation questions from the pairwise relation graph; "
            "hybrid uses graph only when the relation question can be parsed."
        ),
    )
    parser.add_argument(
        "--relation-graph-min-score",
        type=float,
        default=0.0,
        help="Minimum graph score required before graph-assisted relation answers override the QA head.",
    )
    parser.add_argument(
        "--relation-graph-score-mode",
        choices=RELATION_GRAPH_SCORE_MODES,
        default="predicate_only",
        help="Score function used by graph-assisted relation answers.",
    )
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


def _calibrated_relation_answer_metrics(rows: Sequence[Mapping[str, float]]) -> Dict[str, float]:
    if not rows:
        return {
            "calibrated_relation_answer_acc": 0.0,
            "calibrated_relation_graph_min_score": 0.0,
            "calibrated_relation_graph_override_count": 0.0,
        }
    thresholds = {
        0.0,
        0.005,
        0.01,
        0.02,
        0.03,
        0.04,
        0.05,
        0.075,
        0.1,
        0.15,
        0.2,
        0.3,
        0.5,
    }
    thresholds.update(float(row.get("graph_score", 0.0)) for row in rows)
    best_threshold = 0.0
    best_correct = -1.0
    best_override_count = 0.0
    for threshold in sorted(thresholds):
        correct = 0.0
        override_count = 0.0
        for row in rows:
            use_graph = bool(row.get("graph_available", 0.0) > 0.5 and row.get("graph_score", 0.0) >= threshold)
            if use_graph:
                correct += float(row.get("graph_correct", 0.0))
                override_count += 1.0
            else:
                correct += float(row.get("raw_correct", 0.0))
        if correct > best_correct:
            best_correct = correct
            best_threshold = float(threshold)
            best_override_count = override_count
    return {
        "calibrated_relation_answer_acc": _safe_ratio(best_correct, float(len(rows))),
        "calibrated_relation_graph_min_score": best_threshold,
        "calibrated_relation_graph_override_count": best_override_count,
    }


def _relation_score_mode_metrics(rows: Sequence[Mapping[str, float]]) -> Dict[str, float]:
    if not rows:
        return {
            "graph_answer_acc": 0.0,
            "graph_attempt_count": 0.0,
            **_calibrated_relation_answer_metrics(()),
        }
    attempt_count = sum(1.0 for row in rows if float(row.get("graph_available", 0.0)) > 0.5)
    graph_correct = sum(float(row.get("graph_correct", 0.0)) for row in rows if float(row.get("graph_available", 0.0)) > 0.5)
    calibrated = _calibrated_relation_answer_metrics(rows)
    return {
        "graph_answer_acc": _safe_ratio(graph_correct, attempt_count),
        "graph_attempt_count": attempt_count,
        **calibrated,
    }


def _best_relation_score_mode(metrics: Mapping[str, Mapping[str, float]]) -> Dict[str, float | str]:
    if not metrics:
        return {
            "recommended_relation_graph_score_mode": "",
            "recommended_relation_graph_answer_acc": 0.0,
            "recommended_calibrated_relation_answer_acc": 0.0,
        }
    best_mode = max(
        metrics,
        key=lambda score_mode: (
            float(metrics[score_mode].get("calibrated_relation_answer_acc", 0.0)),
            float(metrics[score_mode].get("graph_answer_acc", 0.0)),
        ),
    )
    best_metrics = metrics[best_mode]
    return {
        "recommended_relation_graph_score_mode": best_mode,
        "recommended_relation_graph_answer_acc": float(best_metrics.get("graph_answer_acc", 0.0)),
        "recommended_calibrated_relation_answer_acc": float(
            best_metrics.get("calibrated_relation_answer_acc", 0.0)
        ),
    }


def _patch_grid_from_config(config: Any) -> Tuple[int, int]:
    grid_h = int((int(config.image_size) - int(config.patch_size)) / int(config.patch_stride)) + 1
    grid_w = int((int(config.image_size) - int(config.patch_size)) / int(config.patch_stride)) + 1
    return max(1, grid_h), max(1, grid_w)


def _combine_relation_graph_score(
    *,
    relation_logit: float,
    relation_score: float,
    predicate_score: float,
    score_mode: str,
) -> float:
    if score_mode == "product":
        return relation_score * predicate_score
    if score_mode == "logit_add":
        return relation_logit + math.log(max(predicate_score, 1e-8))
    if score_mode == "relation_only":
        return relation_score
    if score_mode == "predicate_only":
        return predicate_score
    raise ValueError(f"unsupported relation graph score mode: {score_mode}")


def _parse_relation_answer_question(question: str, labels: Sequence[str]) -> tuple[str, str] | None:
    normalized = question.strip().lower().rstrip("?").strip()
    if not normalized.startswith("what is "):
        return None
    body = normalized[len("what is ") :].strip()
    label_lookup = {label.lower(): label for label in labels}
    for relation_text, predicate in RELATION_TEXT_TO_PREDICATE:
        prefix = f"{relation_text} "
        if not body.startswith(prefix):
            continue
        object_label = body[len(prefix) :].strip()
        if object_label.startswith("the "):
            object_label = object_label[len("the ") :].strip()
        if object_label in label_lookup:
            return predicate, label_lookup[object_label]
    return None


def _box_patch_indices(
    visual_data: Any,
    batch: Any,
    batch_index: int,
    *,
    image_size: int,
    patch_grid: Tuple[int, int],
) -> List[int]:
    boxes = batch.boxes_xyxy[batch_index].detach().cpu()
    return [
        int(
            visual_data._box_center_to_patch_index(  # noqa: SLF001 - experiment evaluator mirrors training target logic.
                tuple(float(value) for value in box.tolist()),
                image_size=image_size,
                patch_grid=patch_grid,
            )
        )
        for box in boxes
    ]


def _relation_graph_answer(
    visual_data: Any,
    batch: Any,
    forward: Mapping[str, torch.Tensor],
    *,
    batch_index: int,
    qa_pair: Any,
    qa_vocabulary: Any,
    relation_predicate_to_index: Mapping[str, int],
    image_size: int,
    patch_grid: Tuple[int, int],
    min_score: float,
    score_mode: str = "product",
) -> Dict[str, Any] | None:
    labels = tuple(str(label) for label in batch.box_labels[batch_index])
    parsed = _parse_relation_answer_question(str(qa_pair.question), labels)
    if parsed is None:
        return None
    predicate, object_label = parsed
    object_box_indices = [index for index, label in enumerate(labels) if label.lower() == object_label.lower()]
    if not object_box_indices:
        return None

    patch_indices = _box_patch_indices(
        visual_data,
        batch,
        batch_index,
        image_size=image_size,
        patch_grid=patch_grid,
    )
    if not patch_indices:
        return None

    pair_logits = forward["pair_relation_scores"][batch_index].detach()
    pair_scores = torch.sigmoid(pair_logits)
    predicate_probs = torch.softmax(forward["pair_relation_predicate_logits"][batch_index].detach(), dim=-1)
    predicate_index = int(relation_predicate_to_index.get(predicate, 0))
    best: Dict[str, Any] | None = None

    for object_box_index in object_box_indices:
        if object_box_index >= len(patch_indices):
            continue
        object_patch = patch_indices[object_box_index]
        for subject_box_index, subject_label in enumerate(labels):
            if subject_box_index == object_box_index or subject_box_index >= len(patch_indices):
                continue
            subject_patch = patch_indices[subject_box_index]
            relation_logit = float(pair_logits[subject_patch, object_patch].item())
            relation_score = float(pair_scores[subject_patch, object_patch].item())
            predicate_score = 1.0
            if 0 <= predicate_index < predicate_probs.shape[-1]:
                predicate_score = float(predicate_probs[subject_patch, object_patch, predicate_index].item())
            combined_score = _combine_relation_graph_score(
                relation_logit=relation_logit,
                relation_score=relation_score,
                predicate_score=predicate_score,
                score_mode=score_mode,
            )
            if best is None or combined_score > float(best["score"]):
                best = {
                    "answer": subject_label,
                    "score": combined_score,
                    "relation_score": relation_score,
                    "predicate_score": predicate_score,
                    "score_mode": score_mode,
                    "predicate": predicate,
                    "object_label": object_label,
                }

    if best is None or float(best["score"]) < float(min_score):
        return None
    answer_id = qa_vocabulary.answer_stoi.get(str(best["answer"]))
    if answer_id is None:
        return None
    best["answer_id"] = int(answer_id)
    return best


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
    raw_answer_type_stats: Dict[str, Dict[str, float]] = {}
    total = 0.0
    correct = 0.0
    raw_correct = 0.0
    grounded_total = 0.0
    grounded_correct = 0.0
    relation_grounded_total = 0.0
    relation_grounded_correct = 0.0
    relation_total = 0.0
    relation_raw_correct = 0.0
    relation_selected_correct = 0.0
    relation_graph_attempt_total = 0.0
    relation_graph_correct = 0.0
    relation_graph_override_total = 0.0
    relation_graph_override_correct = 0.0
    relation_calibration_rows: List[Dict[str, float]] = []
    relation_score_mode_rows: Dict[str, List[Dict[str, float]]] = {
        score_mode: []
        for score_mode in RELATION_GRAPH_SCORE_MODES
    }
    summary_alignment_sum = 0.0
    scene_alignment_sum = 0.0
    schema_chain_count = 0.0
    batch_count = 0.0
    patch_grid = _patch_grid_from_config(model.config)

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
            raw_pred_ids = masked_answer_logits.argmax(dim=-1)
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
                raw_pred_id = int(raw_pred_ids[index].item())
                pred_id = raw_pred_id
                graph_answer = None
                graph_candidate = None
                graph_candidates_by_mode: Dict[str, Dict[str, Any]] = {}
                qa_pair = sample.qa_pairs[0]
                answer_type = qa_pair.answer_type or "unknown"
                if answer_type == "relation":
                    for score_mode in RELATION_GRAPH_SCORE_MODES:
                        mode_candidate = _relation_graph_answer(
                            visual_data,
                            batch,
                            forward,
                            batch_index=index,
                            qa_pair=qa_pair,
                            qa_vocabulary=qa_vocabulary,
                            relation_predicate_to_index=relation_predicate_to_index,
                            image_size=model.config.image_size,
                            patch_grid=patch_grid,
                            min_score=float("-inf"),
                            score_mode=score_mode,
                        )
                        if mode_candidate is not None:
                            graph_candidates_by_mode[score_mode] = mode_candidate
                    graph_candidate = graph_candidates_by_mode.get(args.relation_graph_score_mode)
                if answer_type == "relation" and args.relation_answer_mode in {"graph", "hybrid"}:
                    if graph_candidate is not None:
                        relation_graph_attempt_total += 1.0
                    if graph_candidate is not None and float(graph_candidate["score"]) >= float(args.relation_graph_min_score):
                        graph_answer = graph_candidate
                        pred_id = int(graph_answer["answer_id"])
                        relation_graph_override_total += 1.0
                is_correct = float(pred_id == gold_id)
                raw_is_correct = float(raw_pred_id == gold_id)
                graph_is_correct = float(int(graph_candidate["answer_id"]) == gold_id) if graph_candidate is not None else 0.0
                correct += is_correct
                raw_correct += raw_is_correct

                answer_type_stats.setdefault(answer_type, {"total": 0.0, "correct": 0.0})
                answer_type_stats[answer_type]["total"] += 1.0
                answer_type_stats[answer_type]["correct"] += is_correct
                raw_answer_type_stats.setdefault(answer_type, {"total": 0.0, "correct": 0.0})
                raw_answer_type_stats[answer_type]["total"] += 1.0
                raw_answer_type_stats[answer_type]["correct"] += raw_is_correct
                if answer_type == "relation":
                    relation_total += 1.0
                    relation_raw_correct += raw_is_correct
                    relation_selected_correct += is_correct
                    for score_mode in RELATION_GRAPH_SCORE_MODES:
                        mode_candidate = graph_candidates_by_mode.get(score_mode)
                        relation_score_mode_rows[score_mode].append(
                            {
                                "raw_correct": raw_is_correct,
                                "graph_correct": float(int(mode_candidate["answer_id"]) == gold_id)
                                if mode_candidate is not None
                                else 0.0,
                                "graph_available": 1.0 if mode_candidate is not None else 0.0,
                                "graph_score": float(mode_candidate["score"]) if mode_candidate is not None else 0.0,
                            }
                        )
                    if graph_candidate is not None:
                        relation_graph_correct += graph_is_correct
                    if graph_answer is not None:
                        relation_graph_override_correct += is_correct
                    relation_calibration_rows.append(
                        {
                            "raw_correct": raw_is_correct,
                            "graph_correct": graph_is_correct,
                            "graph_available": 1.0 if graph_candidate is not None else 0.0,
                            "graph_score": float(graph_candidate["score"]) if graph_candidate is not None else 0.0,
                        }
                    )

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
                            "raw_predicted_answer": qa_vocabulary.answer_itos[raw_pred_id]
                            if raw_pred_id < len(qa_vocabulary.answer_itos)
                            else "<out_of_range>",
                            "graph_predicted_answer": str(graph_answer["answer"]) if graph_answer is not None else "",
                            "graph_score": float(graph_answer["score"]) if graph_answer is not None else 0.0,
                            "graph_score_mode": str(graph_answer["score_mode"]) if graph_answer is not None else args.relation_graph_score_mode,
                            "used_graph_answer": bool(graph_answer is not None and pred_id != raw_pred_id),
                            "answer_type": answer_type,
                            "correct": bool(is_correct),
                            "raw_correct": bool(raw_is_correct),
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
    raw_answer_type_accuracy = {
        answer_type: {
            "count": int(stats["total"]),
            "accuracy": _safe_ratio(stats["correct"], stats["total"]),
        }
        for answer_type, stats in sorted(raw_answer_type_stats.items())
    }
    mean_summary_schema_alignment = _safe_ratio(summary_alignment_sum, batch_count)
    mean_scene_schema_alignment = _safe_ratio(scene_alignment_sum, batch_count)
    schema_chain_activated = bool(mean_summary_schema_alignment > 0.0 or mean_scene_schema_alignment > 0.0)
    calibrated_relation_metrics = _calibrated_relation_answer_metrics(relation_calibration_rows)
    relation_score_mode_metrics = {
        score_mode: _relation_score_mode_metrics(rows)
        for score_mode, rows in relation_score_mode_rows.items()
    }
    best_score_mode_metrics = _best_relation_score_mode(relation_score_mode_metrics)

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
            "raw_overall_answer_acc": _safe_ratio(raw_correct, total),
            "grounded_answer_acc": _safe_ratio(grounded_correct, grounded_total),
            "relation_grounded_answer_acc": _safe_ratio(relation_grounded_correct, relation_grounded_total),
            "raw_relation_answer_acc": _safe_ratio(relation_raw_correct, relation_total),
            "selected_relation_answer_acc": _safe_ratio(relation_selected_correct, relation_total),
            "relation_graph_answer_acc": _safe_ratio(relation_graph_correct, relation_graph_attempt_total),
            "relation_graph_attempt_count": int(relation_graph_attempt_total),
            "relation_graph_override_count": int(relation_graph_override_total),
            "relation_graph_override_acc": _safe_ratio(relation_graph_override_correct, relation_graph_override_total),
            "relation_answer_mode": args.relation_answer_mode,
            "relation_graph_min_score": float(args.relation_graph_min_score),
            "relation_graph_score_mode": args.relation_graph_score_mode,
            **calibrated_relation_metrics,
            **best_score_mode_metrics,
            "mean_summary_schema_alignment": mean_summary_schema_alignment,
            "mean_scene_schema_alignment": mean_scene_schema_alignment,
            "schema_chain_activated": schema_chain_activated,
            "schema_chain_activated_ratio": _safe_ratio(schema_chain_count, batch_count),
        },
        "relation_score_mode_metrics": relation_score_mode_metrics,
        "answer_type_accuracy": answer_type_accuracy,
        "raw_answer_type_accuracy": raw_answer_type_accuracy,
        "sample_predictions": sample_predictions,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"report_path={output_path}")


if __name__ == "__main__":
    main()
