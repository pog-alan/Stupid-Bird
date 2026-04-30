from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import torch
import torch.nn.functional as F

from sb.core_lm_torch import runtime_device_report


DEFAULT_OUTPUT_PATH = Path("data/processed/experiments/sb_visual_scene_box_train.json")
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
RELATION_GRAPH_SCORE_MODES = ("product", "logit_add", "combined", "relation_only", "predicate_only")


def _profile_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--train-profile", default="")
    return parser


def _load_module(module_name: str, path: Path) -> object:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train SB-Visual on scene classification, box supervision, relation supervision, and QA."
    )
    parser.add_argument("--train-profile", default="")
    parser.add_argument("--resume-from", default="")
    parser.add_argument("--dataset-root", default="data/processed/sb_visual_synth_train")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--create-synthetic", action="store_true")
    parser.add_argument("--scene-names", default="stacked,spill,orderly")
    parser.add_argument("--images-per-scene", type=int, default=8)
    parser.add_argument("--val-images-per-scene", type=int, default=4)
    parser.add_argument("--synthetic-image-size", type=int, default=96)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--patch-stride", type=int, default=16)
    parser.add_argument("--d-model", type=int, default=96)
    parser.add_argument("--state-dim", type=int, default=96)
    parser.add_argument("--schema-slots", type=int, default=9)
    parser.add_argument("--object-memory-slots", type=int, default=16)
    parser.add_argument("--relation-memory-slots", type=int, default=24)
    parser.add_argument("--scene-memory-slots", type=int, default=6)
    parser.add_argument("--summary-memory-slots", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--print-every", type=int, default=2)
    parser.add_argument("--eval-every", type=int, default=2)
    parser.add_argument("--max-val-batches", type=int, default=2)
    parser.add_argument("--checkpoint-every", type=int, default=0)
    parser.add_argument("--checkpoint-dir", default="data/processed/checkpoints/sb_visual_scene_box")
    parser.add_argument("--events-path", default="")
    parser.add_argument("--scene-loss-weight", type=float, default=1.0)
    parser.add_argument("--object-loss-weight", type=float, default=0.5)
    parser.add_argument("--box-loss-weight", type=float, default=2.0)
    parser.add_argument("--relation-loss-weight", type=float, default=0.35)
    parser.add_argument("--relation-positive-weight", type=float, default=0.0)
    parser.add_argument("--pair-relation-positive-weight", type=float, default=1.0)
    parser.add_argument("--relation-predicate-loss-weight", type=float, default=0.5)
    parser.add_argument("--relation-ranking-loss-weight", type=float, default=0.0)
    parser.add_argument("--relation-answer-graph-loss-weight", type=float, default=0.0)
    parser.add_argument(
        "--relation-ranking-score-mode",
        choices=("combined", "relation_only", "predicate_only"),
        default="combined",
        help=(
            "Score used by relation subject ranking: combined trains relationness and predicate jointly, "
            "relation_only targets the pair relationness head, predicate_only targets the predicate head."
        ),
    )
    parser.add_argument(
        "--relation-answer-graph-score-mode",
        choices=RELATION_GRAPH_SCORE_MODES,
        default="predicate_only",
        help="Differentiable score used to train relation QA answers from the pairwise graph.",
    )
    parser.add_argument("--qa-loss-weight", type=float, default=0.5)
    parser.add_argument("--qa-answer-weight-power", type=float, default=0.5)
    parser.add_argument("--qa-answer-weight-max", type=float, default=4.0)
    parser.add_argument("--qa-type-sampling-profile", default="grounded_focus")
    parser.add_argument("--qa-type-loss-profile", default="uniform")
    parser.add_argument("--max-question-vocab-size", type=int, default=256)
    parser.add_argument("--question-max-len", type=int, default=48)
    parser.add_argument("--experiment-tag", default="")
    parser.add_argument("--output-path", default="")
    return parser


def _load_profile_defaults(path_str: str) -> Dict[str, Any]:
    path = Path(path_str)
    if not path.is_absolute():
        path = Path.cwd() / path
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("train profile must be a JSON object.")
    return payload


def _parse_args() -> argparse.Namespace:
    profile_args, _ = _profile_parser().parse_known_args()
    parser = _build_parser()
    if profile_args.train_profile:
        parser.set_defaults(**_load_profile_defaults(profile_args.train_profile))
    return parser.parse_args()


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


def _resolve_checkpoint_dir(path_str: str, experiment_tag: str) -> Path:
    path = Path(path_str)
    if experiment_tag:
        path = path.with_name(f"{path.name}_{experiment_tag}")
    if not path.is_absolute():
        path = Path.cwd() / path
    return path


def _resolve_events_path(path_str: str, checkpoint_dir: Path) -> Path:
    if path_str:
        path = Path(path_str)
        if not path.is_absolute():
            path = Path.cwd() / path
        return path
    return checkpoint_dir / "events.jsonl"


def _set_seed(seed: int) -> random.Random:
    rng = random.Random(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return rng


def _capture_rng_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "python_random_state": random.getstate(),
        "torch_rng_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: Mapping[str, Any] | None) -> None:
    if not state:
        return
    python_state = state.get("python_random_state")
    if python_state is not None:
        random.setstate(python_state)
    torch_state = state.get("torch_rng_state")
    if torch_state is not None:
        torch.set_rng_state(torch_state.cpu() if hasattr(torch_state, "cpu") else torch_state)
    cuda_state = state.get("torch_cuda_rng_state_all")
    if cuda_state is not None and torch.cuda.is_available():
        normalized = [item.cpu() if hasattr(item, "cpu") else item for item in cuda_state]
        torch.cuda.set_rng_state_all(normalized)


def _torch_load(path: Path, device: torch.device) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _config_to_dict(config: object) -> Dict[str, Any]:
    if is_dataclass(config):
        return asdict(config)
    if hasattr(config, "__dict__"):
        return dict(config.__dict__)
    raise TypeError("unsupported config object for serialization")


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


def _write_event(events_path: Path, event_type: str, payload: Mapping[str, Any]) -> None:
    events_path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "type": event_type,
        **payload,
    }
    with events_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _serialize_dataset_info(dataset_info: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "root": str(dataset_info.get("root", "")),
        "base_dir": str(dataset_info.get("base_dir", "")),
        "split": str(dataset_info.get("split", "")),
        "scene_to_index": dict(dataset_info.get("scene_to_index", {})),
        "annotation_manifest": str(dataset_info.get("annotation_manifest", "")),
        "sample_count": len(dataset_info.get("samples", ()) or ()),
    }


def _save_checkpoint(
    path: Path,
    *,
    model,
    optimizer,
    args: argparse.Namespace,
    train_dataset_info: Dict[str, Any],
    val_dataset_info: Dict[str, Any],
    relation_predicate_to_index: Dict[str, int],
    qa_vocabulary,
    qa_answer_type_to_indices: Mapping[str, Sequence[int]],
    step: int,
    history,
    val_history,
    best_val,
    best_relation_val,
    loss_weights: Dict[str, float],
    rng_state: Mapping[str, Any],
    events_path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "model_config": _config_to_dict(model.config),
        "args": vars(args),
        "train_dataset_info": _serialize_dataset_info(train_dataset_info),
        "val_dataset_info": _serialize_dataset_info(val_dataset_info),
        "relation_predicate_to_index": relation_predicate_to_index,
        "qa_vocabulary": {
            "question_stoi": qa_vocabulary.question_stoi,
            "question_itos": list(qa_vocabulary.question_itos),
            "answer_stoi": qa_vocabulary.answer_stoi,
            "answer_itos": list(qa_vocabulary.answer_itos),
            "answer_counts": qa_vocabulary.answer_counts,
            "pad_token": qa_vocabulary.pad_token,
            "unk_token": qa_vocabulary.unk_token,
        },
        "qa_answer_type_to_indices": {
            answer_type: list(indices)
            for answer_type, indices in qa_answer_type_to_indices.items()
        },
        "step": step,
        "history": history,
        "val_history": val_history,
        "best_val": best_val,
        "best_relation_val": best_relation_val or {},
        "loss_weights": loss_weights,
        "rng_state": rng_state,
        "events_path": str(events_path.resolve()),
    }
    torch.save(payload, path)


def _sample_dataset(
    visual_data,
    dataset_info: Dict[str, Any],
    *,
    batch_size: int,
    config,
    device: torch.device | str,
    rng: random.Random,
    qa_type_weights: Mapping[str, float] | None = None,
):
    return visual_data.sample_visual_batch(
        dataset_info["samples"],
        batch_size=batch_size,
        config=config,
        device=device,
        rng=rng,
        qa_type_weights=qa_type_weights,
    )


def _resolve_qa_type_weights(profile_name: str) -> Dict[str, float] | None:
    profile_key = (profile_name or "").strip().lower()
    if profile_key in {"", "uniform", "none"}:
        return None
    if profile_key == "grounded_focus":
        return {
            "classification": 0.5,
            "boolean": 1.0,
            "relation": 1.35,
            "relation_predicate": 1.25,
            "grounded_reasoning": 1.5,
            "causal": 1.35,
            "*": 1.0,
        }
    if profile_key == "relation_recovery":
        return {
            "classification": 0.45,
            "boolean": 0.7,
            "relation": 2.2,
            "relation_predicate": 1.9,
            "grounded_reasoning": 1.6,
            "causal": 1.45,
            "*": 1.0,
        }
    if profile_key == "balanced_relation_grounded":
        return {
            "classification": 0.45,
            "boolean": 0.9,
            "relation": 1.7,
            "relation_predicate": 1.45,
            "grounded_reasoning": 1.55,
            "causal": 1.35,
            "*": 1.0,
        }
    raise ValueError(f"unsupported qa_type_sampling_profile: {profile_name}")


def _build_qa_answer_weight_tensor(
    qa_vocabulary,
    *,
    power: float,
    max_weight: float,
    device: torch.device,
) -> torch.Tensor:
    weights = []
    for answer in qa_vocabulary.answer_itos:
        if answer == "<unk>":
            weights.append(1.0)
            continue
        count = max(int(qa_vocabulary.answer_counts.get(answer, 1)), 1)
        weight = count ** (-power)
        weights.append(weight)
    tensor = torch.tensor(weights, dtype=torch.float32, device=device)
    finite_mask = torch.isfinite(tensor) & (tensor > 0)
    if bool(finite_mask.any()):
        tensor = tensor / tensor[finite_mask].mean().clamp_min(1e-6)
    tensor = tensor.clamp_max(max_weight)
    return tensor


def _resolve_qa_type_loss_weights(profile_name: str) -> Dict[str, float] | None:
    profile_key = (profile_name or "").strip().lower()
    if profile_key in {"", "uniform", "none"}:
        return None
    if profile_key == "relation_aware":
        return {
            "classification": 0.75,
            "boolean": 0.8,
            "relation": 1.95,
            "relation_predicate": 1.65,
            "grounded_reasoning": 1.35,
            "causal": 1.25,
            "*": 1.0,
        }
    if profile_key == "relation_hard":
        return {
            "classification": 0.65,
            "boolean": 0.75,
            "relation": 2.3,
            "relation_predicate": 2.0,
            "grounded_reasoning": 1.5,
            "causal": 1.35,
            "*": 1.0,
        }
    if profile_key == "balanced_relation_grounded":
        return {
            "classification": 0.9,
            "boolean": 0.9,
            "relation": 1.5,
            "relation_predicate": 1.35,
            "grounded_reasoning": 1.3,
            "causal": 1.15,
            "*": 1.0,
        }
    raise ValueError(f"unsupported qa_type_loss_profile: {profile_name}")


def _build_qa_type_weight_tensor(
    batch,
    *,
    device: torch.device,
    qa_type_loss_weights: Mapping[str, float] | None,
) -> torch.Tensor | None:
    if not qa_type_loss_weights:
        return None
    weights = []
    default_weight = float(qa_type_loss_weights.get("*", 1.0))
    for metadata in batch.metadata:
        answer_type = str(metadata.get("sampled_qa_answer_type", "")).strip()
        weights.append(float(qa_type_loss_weights.get(answer_type, default_weight)))
    tensor = torch.tensor(weights, dtype=torch.float32, device=device)
    finite_mask = torch.isfinite(tensor) & (tensor > 0)
    if bool(finite_mask.any()):
        tensor = tensor / tensor[finite_mask].mean().clamp_min(1e-6)
    return tensor


def _parse_relation_answer_question(question: str, labels: Sequence[str]) -> tuple[str, str] | None:
    normalized = question.strip().lower().rstrip("?").strip()
    if not normalized.startswith("what is "):
        return None
    body = normalized[len("what is ") :].strip()
    label_lookup = {str(label).lower(): str(label) for label in labels}
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


def _combine_relation_graph_logits(
    *,
    relation_logits: torch.Tensor,
    predicate_log_probs: torch.Tensor,
    score_mode: str,
) -> torch.Tensor:
    if score_mode == "relation_only":
        return relation_logits
    if score_mode == "predicate_only":
        return predicate_log_probs
    if score_mode == "product":
        return torch.log(torch.sigmoid(relation_logits).clamp_min(1e-8)) + predicate_log_probs
    if score_mode in {"combined", "logit_add"}:
        return relation_logits + predicate_log_probs
    raise ValueError(f"unsupported relation graph score mode: {score_mode}")


def _qa_value(qa_pair: Any, key: str, default: str = "") -> str:
    if isinstance(qa_pair, dict):
        return str(qa_pair.get(key, default))
    return str(getattr(qa_pair, key, default))


def _qa_pairs_for_graph_supervision(batch, batch_index: int) -> Sequence[Any]:
    metadata = batch.metadata[batch_index] if batch_index < len(batch.metadata) else {}
    if isinstance(metadata, dict):
        all_qa_pairs = metadata.get("all_qa_pairs")
        if isinstance(all_qa_pairs, (list, tuple)):
            return all_qa_pairs
    return batch.qa_pairs[batch_index]


def _box_patch_indices_for_ranking(
    visual_data,
    boxes: torch.Tensor,
    *,
    image_size: int,
    patch_grid: tuple[int, int],
) -> list[int]:
    cpu_boxes = boxes.detach().cpu()
    return [
        int(
            visual_data._box_center_to_patch_index(  # noqa: SLF001 - experiment target mirrors patch supervision.
                tuple(float(value) for value in box.tolist()),
                image_size=image_size,
                patch_grid=patch_grid,
            )
        )
        for box in cpu_boxes
    ]


def _relation_subject_ranking_loss(
    *,
    visual_data,
    batch,
    forward: Mapping[str, torch.Tensor],
    patch_grid: tuple[int, int],
    image_size: int,
    relation_predicate_to_index: Mapping[str, int],
    device: torch.device,
    ranking_score_mode: str = "combined",
) -> tuple[torch.Tensor, Dict[str, float]]:
    losses = []
    correct_count = 0.0
    query_count = 0.0
    skipped_duplicate_patch = 0.0
    skipped_invalid = 0.0
    predicate_log_probs = F.log_softmax(forward["pair_relation_predicate_logits"], dim=-1)

    for batch_index, relations in enumerate(batch.relations):
        boxes = batch.boxes_xyxy[batch_index]
        if not boxes.numel() or not relations:
            continue
        box_patch_indices = _box_patch_indices_for_ranking(
            visual_data,
            boxes,
            image_size=image_size,
            patch_grid=patch_grid,
        )
        for relation in relations:
            if (
                relation.subject < 0
                or relation.object < 0
                or relation.subject >= len(box_patch_indices)
                or relation.object >= len(box_patch_indices)
                or relation.subject == relation.object
            ):
                skipped_invalid += 1.0
                continue

            candidates = [
                (box_index, patch_index)
                for box_index, patch_index in enumerate(box_patch_indices)
                if box_index != relation.object
            ]
            candidate_patches = [patch_index for _, patch_index in candidates]
            if len(candidates) < 2 or len(set(candidate_patches)) != len(candidate_patches):
                skipped_duplicate_patch += 1.0
                continue
            target_positions = [
                candidate_index
                for candidate_index, (box_index, _) in enumerate(candidates)
                if box_index == relation.subject
            ]
            if len(target_positions) != 1:
                skipped_invalid += 1.0
                continue

            object_patch = int(box_patch_indices[relation.object])
            subject_patch_tensor = torch.tensor(candidate_patches, dtype=torch.long, device=device)
            relation_logits = forward["pair_relation_scores"][batch_index, subject_patch_tensor, object_patch]
            predicate_index = int(relation_predicate_to_index.get(relation.predicate, 0))
            has_predicate_score = 0 <= predicate_index < predicate_log_probs.shape[-1]
            if has_predicate_score:
                predicate_scores = predicate_log_probs[
                    batch_index,
                    subject_patch_tensor,
                    object_patch,
                    predicate_index,
                ]
            else:
                predicate_scores = relation_logits.new_zeros(relation_logits.shape)

            if ranking_score_mode == "relation_only":
                ranking_logits = relation_logits
            elif ranking_score_mode == "predicate_only" and has_predicate_score:
                ranking_logits = predicate_scores
            else:
                ranking_logits = relation_logits + predicate_scores
            target = torch.tensor([int(target_positions[0])], dtype=torch.long, device=device)
            losses.append(F.cross_entropy(ranking_logits.unsqueeze(0), target))
            query_count += 1.0
            correct_count += float(int(ranking_logits.detach().argmax().item()) == int(target_positions[0]))

    if losses:
        loss = torch.stack(losses).mean()
    else:
        loss = torch.zeros((), device=device)
    metrics = {
        "relation_ranking_acc": correct_count / query_count if query_count > 0.0 else 1.0,
        "relation_ranking_query_count": query_count,
        "relation_ranking_skipped_duplicate_patch": skipped_duplicate_patch,
        "relation_ranking_skipped_invalid": skipped_invalid,
    }
    return loss, metrics


def _relation_answer_graph_loss(
    *,
    visual_data,
    batch,
    forward: Mapping[str, torch.Tensor],
    patch_grid: tuple[int, int],
    image_size: int,
    relation_predicate_to_index: Mapping[str, int],
    device: torch.device,
    score_mode: str = "predicate_only",
) -> tuple[torch.Tensor, Dict[str, float]]:
    losses = []
    correct_count = 0.0
    query_count = 0.0
    skipped_parse = 0.0
    skipped_no_target = 0.0
    skipped_invalid = 0.0
    predicate_log_probs = F.log_softmax(forward["pair_relation_predicate_logits"], dim=-1)

    for batch_index, sampled_qa_pairs in enumerate(batch.qa_pairs):
        qa_pairs = _qa_pairs_for_graph_supervision(batch, batch_index)
        if not qa_pairs and not sampled_qa_pairs:
            continue
        labels = tuple(str(label) for label in batch.box_labels[batch_index])
        boxes = batch.boxes_xyxy[batch_index]
        if not boxes.numel() or not labels:
            skipped_invalid += 1.0
            continue
        box_patch_indices = _box_patch_indices_for_ranking(
            visual_data,
            boxes,
            image_size=image_size,
            patch_grid=patch_grid,
        )

        for qa_pair in qa_pairs:
            if _qa_value(qa_pair, "answer_type").strip() != "relation":
                continue
            parsed = _parse_relation_answer_question(_qa_value(qa_pair, "question"), labels)
            if parsed is None:
                skipped_parse += 1.0
                continue
            predicate, object_label = parsed
            predicate_index = int(relation_predicate_to_index.get(predicate, -1))
            if predicate_index < 0 or predicate_index >= predicate_log_probs.shape[-1]:
                skipped_invalid += 1.0
                continue

            object_box_indices = [
                index
                for index, label in enumerate(labels)
                if label.lower() == object_label.lower() and index < len(box_patch_indices)
            ]
            if not object_box_indices:
                skipped_invalid += 1.0
                continue

            candidate_logits = []
            target_positions = []
            gold_answer = _qa_value(qa_pair, "answer").strip().lower()
            for object_box_index in object_box_indices:
                object_patch = int(box_patch_indices[object_box_index])
                for subject_box_index, subject_label in enumerate(labels):
                    if subject_box_index == object_box_index or subject_box_index >= len(box_patch_indices):
                        continue
                    subject_patch = int(box_patch_indices[subject_box_index])
                    relation_logits = forward["pair_relation_scores"][batch_index, subject_patch, object_patch]
                    predicate_scores = predicate_log_probs[batch_index, subject_patch, object_patch, predicate_index]
                    candidate_logits.append(
                        _combine_relation_graph_logits(
                            relation_logits=relation_logits,
                            predicate_log_probs=predicate_scores,
                            score_mode=score_mode,
                        )
                    )
                    if str(subject_label).strip().lower() == gold_answer:
                        target_positions.append(len(candidate_logits) - 1)

            if len(candidate_logits) < 2:
                skipped_invalid += 1.0
                continue
            if not target_positions:
                skipped_no_target += 1.0
                continue

            logits = torch.stack(candidate_logits)
            target_index_tensor = torch.tensor(target_positions, dtype=torch.long, device=device)
            loss = torch.logsumexp(logits, dim=0) - torch.logsumexp(logits[target_index_tensor], dim=0)
            losses.append(loss)
            query_count += 1.0
            correct_count += float(int(logits.detach().argmax().item()) in set(target_positions))

    if losses:
        loss = torch.stack(losses).mean()
    else:
        loss = torch.zeros((), device=device)
    metrics = {
        "relation_answer_graph_acc": correct_count / query_count if query_count > 0.0 else 1.0,
        "relation_answer_graph_query_count": query_count,
        "relation_answer_graph_skipped_parse": skipped_parse,
        "relation_answer_graph_skipped_no_target": skipped_no_target,
        "relation_answer_graph_skipped_invalid": skipped_invalid,
    }
    return loss, metrics


def _safe_divide(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    return numerator / denominator.clamp_min(1.0)


def _binary_relation_metrics(
    probs: torch.Tensor,
    target: torch.Tensor,
    *,
    device: torch.device,
    valid_mask: torch.Tensor | None = None,
) -> Dict[str, torch.Tensor]:
    if valid_mask is not None:
        active_mask = valid_mask > 0.5
        if bool(active_mask.any()):
            probs = probs[active_mask]
            target = target[active_mask]
        else:
            zero = torch.zeros((), device=device)
            one = torch.ones((), device=device)
            return {
                "acc": one,
                "precision": zero,
                "recall": zero,
                "f1": zero,
                "best_threshold": torch.tensor(0.5, device=device),
                "best_precision": zero,
                "best_recall": zero,
                "best_f1": zero,
            }
    relation_pred = probs > 0.5
    relation_target = target > 0.5
    relation_acc = (relation_pred == relation_target).float().mean()
    relation_tp = (relation_pred & relation_target).sum().float()
    relation_fp = (relation_pred & (~relation_target)).sum().float()
    relation_fn = ((~relation_pred) & relation_target).sum().float()
    relation_precision = _safe_divide(relation_tp, relation_tp + relation_fp)
    relation_recall = _safe_divide(relation_tp, relation_tp + relation_fn)
    relation_f1 = _safe_divide(2.0 * relation_precision * relation_recall, relation_precision + relation_recall)
    best_relation_threshold = torch.tensor(0.5, device=device)
    best_relation_precision = relation_precision
    best_relation_recall = relation_recall
    best_relation_f1 = relation_f1
    for threshold in (0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7):
        threshold_tensor = torch.tensor(threshold, device=device)
        threshold_pred = probs > threshold_tensor
        threshold_tp = (threshold_pred & relation_target).sum().float()
        threshold_fp = (threshold_pred & (~relation_target)).sum().float()
        threshold_fn = ((~threshold_pred) & relation_target).sum().float()
        threshold_precision = _safe_divide(threshold_tp, threshold_tp + threshold_fp)
        threshold_recall = _safe_divide(threshold_tp, threshold_tp + threshold_fn)
        threshold_f1 = _safe_divide(
            2.0 * threshold_precision * threshold_recall,
            threshold_precision + threshold_recall,
        )
        if bool(threshold_f1 > best_relation_f1):
            best_relation_threshold = threshold_tensor
            best_relation_precision = threshold_precision
            best_relation_recall = threshold_recall
            best_relation_f1 = threshold_f1
    return {
        "acc": relation_acc,
        "precision": relation_precision,
        "recall": relation_recall,
        "f1": relation_f1,
        "best_threshold": best_relation_threshold,
        "best_precision": best_relation_precision,
        "best_recall": best_relation_recall,
        "best_f1": best_relation_f1,
    }


def _compute_losses(
    *,
    model,
    visual_data,
    batch,
    data_config,
    relation_predicate_to_index: Dict[str, int],
    qa_vocabulary,
    device: torch.device,
    loss_weights: Dict[str, float],
    qa_answer_weight_tensor: torch.Tensor | None = None,
    qa_type_loss_weights: Mapping[str, float] | None = None,
    qa_answer_type_to_indices: Mapping[str, Sequence[int]] | None = None,
    relation_positive_weight: float = 0.0,
    pair_relation_positive_weight: float = 1.0,
    relation_ranking_loss_weight: float = 0.0,
    relation_ranking_score_mode: str = "combined",
    relation_answer_graph_loss_weight: float = 0.0,
    relation_answer_graph_score_mode: str = "predicate_only",
) -> tuple[torch.Tensor, Dict[str, float]]:
    patch_payload = visual_data.encode_visual_batch(batch, model.patch_encoder)
    patch_targets = visual_data.build_patch_targets(
        batch,
        patch_grid=patch_payload["patch_grid"],
        image_size=data_config.image_size,
        device=device,
    )
    relation_targets = visual_data.build_relation_targets(
        batch,
        patch_grid=patch_payload["patch_grid"],
        image_size=data_config.image_size,
        predicate_to_index=relation_predicate_to_index,
        device=device,
    )
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
        return_state=False,
        return_aux=True,
    )

    scene_loss = F.cross_entropy(forward["scene_logits"], patch_targets.scene_labels)
    object_loss = F.binary_cross_entropy_with_logits(forward["object_scores"], patch_targets.objectness_targets)

    box_valid_mask = patch_targets.box_valid_mask > 0.5
    if bool(box_valid_mask.any()):
        box_loss = F.smooth_l1_loss(forward["box_deltas"][box_valid_mask], patch_targets.box_targets[box_valid_mask])
    else:
        box_loss = torch.zeros((), device=device)

    relation_pos_weight_tensor = (
        torch.tensor(float(relation_positive_weight), dtype=torch.float32, device=device)
        if float(relation_positive_weight) > 0.0
        else None
    )
    pair_relation_pos_weight_tensor = (
        torch.tensor(float(pair_relation_positive_weight), dtype=torch.float32, device=device)
        if float(pair_relation_positive_weight) > 0.0
        else None
    )
    patch_relation_loss = F.binary_cross_entropy_with_logits(
        forward["relation_scores"],
        relation_targets.relationness_targets,
        pos_weight=relation_pos_weight_tensor,
    )
    pair_relation_valid_mask = relation_targets.pair_relation_valid_mask > 0.5
    if bool(pair_relation_valid_mask.any()):
        pair_relation_loss = F.binary_cross_entropy_with_logits(
            forward["pair_relation_scores"][pair_relation_valid_mask],
            relation_targets.pair_relationness_targets[pair_relation_valid_mask],
            pos_weight=pair_relation_pos_weight_tensor,
        )
    else:
        pair_relation_loss = torch.zeros((), device=device)
    relation_loss = pair_relation_loss + 0.25 * patch_relation_loss
    patch_relation_valid_mask = relation_targets.predicate_valid_mask > 0.5
    relation_valid_mask = relation_targets.pair_predicate_valid_mask > 0.5
    if bool(relation_valid_mask.any()):
        relation_predicate_loss = F.cross_entropy(
            forward["pair_relation_predicate_logits"][relation_valid_mask],
            relation_targets.pair_predicate_targets[relation_valid_mask],
        )
    else:
        relation_predicate_loss = torch.zeros((), device=device)
    relation_ranking_loss, relation_ranking_metrics = _relation_subject_ranking_loss(
        visual_data=visual_data,
        batch=batch,
        forward=forward,
        patch_grid=patch_payload["patch_grid"],
        image_size=data_config.image_size,
        relation_predicate_to_index=relation_predicate_to_index,
        device=device,
        ranking_score_mode=relation_ranking_score_mode,
    )
    relation_answer_graph_loss, relation_answer_graph_metrics = _relation_answer_graph_loss(
        visual_data=visual_data,
        batch=batch,
        forward=forward,
        patch_grid=patch_payload["patch_grid"],
        image_size=data_config.image_size,
        relation_predicate_to_index=relation_predicate_to_index,
        device=device,
        score_mode=relation_answer_graph_score_mode,
    )
    if bool(patch_relation_valid_mask.any()):
        patch_relation_predicate_acc_tensor = (
            forward["relation_predicate_logits"][patch_relation_valid_mask].argmax(dim=-1)
            == relation_targets.predicate_targets[patch_relation_valid_mask]
        ).float().mean()
    else:
        patch_relation_predicate_acc_tensor = torch.ones((), device=device)

    qa_valid_mask = qa_targets.answer_valid_mask > 0.5
    masked_answer_logits = forward["answer_logits"]
    if qa_targets.answer_type_mask.numel() > 0:
        masked_answer_logits = masked_answer_logits.masked_fill(qa_targets.answer_type_mask <= 0.5, -1e9)
    qa_type_weight_tensor = _build_qa_type_weight_tensor(
        batch,
        device=device,
        qa_type_loss_weights=qa_type_loss_weights,
    )
    if bool(qa_valid_mask.any()):
        qa_per_sample_loss = F.cross_entropy(
            masked_answer_logits[qa_valid_mask],
            qa_targets.answer_labels[qa_valid_mask],
            weight=qa_answer_weight_tensor,
            reduction="none",
        )
        if qa_type_weight_tensor is not None:
            active_type_weights = qa_type_weight_tensor[qa_valid_mask]
            qa_loss = (qa_per_sample_loss * active_type_weights).sum() / active_type_weights.sum().clamp_min(1e-6)
        else:
            qa_loss = qa_per_sample_loss.mean()
    else:
        qa_loss = torch.zeros((), device=device)

    total_loss = (
        loss_weights["scene"] * scene_loss
        + loss_weights["object"] * object_loss
        + loss_weights["box"] * box_loss
        + loss_weights["relation"] * relation_loss
        + loss_weights["relation_predicate"] * relation_predicate_loss
        + float(relation_ranking_loss_weight) * relation_ranking_loss
        + float(relation_answer_graph_loss_weight) * relation_answer_graph_loss
        + loss_weights["qa"] * qa_loss
    )

    with torch.no_grad():
        scene_acc = (forward["scene_logits"].argmax(dim=-1) == patch_targets.scene_labels).float().mean()
        object_pred = (torch.sigmoid(forward["object_scores"]) > 0.5).float()
        object_acc = (object_pred == patch_targets.objectness_targets).float().mean()
        relation_metrics = _binary_relation_metrics(
            torch.sigmoid(forward["pair_relation_scores"]),
            relation_targets.pair_relationness_targets,
            device=device,
            valid_mask=relation_targets.pair_relation_valid_mask,
        )
        patch_relation_metrics = _binary_relation_metrics(
            torch.sigmoid(forward["relation_scores"]),
            relation_targets.relationness_targets,
            device=device,
        )
        if bool(relation_valid_mask.any()):
            relation_predicate_acc = (
                forward["pair_relation_predicate_logits"][relation_valid_mask].argmax(dim=-1)
                == relation_targets.pair_predicate_targets[relation_valid_mask]
            ).float().mean()
        else:
            relation_predicate_acc = torch.ones((), device=device)
        if bool(qa_valid_mask.any()):
            qa_acc = (
                masked_answer_logits[qa_valid_mask].argmax(dim=-1)
                == qa_targets.answer_labels[qa_valid_mask]
            ).float().mean()
        else:
            qa_acc = torch.ones((), device=device)

    summary_schema_alignment = float(forward["aux"]["summary_schema_alignment_mean"])
    scene_schema_alignment = float(forward["aux"]["scene_schema_alignment_mean"])
    schema_chain_activated = float((summary_schema_alignment > 0.0) or (scene_schema_alignment > 0.0))

    metrics = {
        "total_loss": float(total_loss.detach().cpu()),
        "scene_loss": float(scene_loss.detach().cpu()),
        "object_loss": float(object_loss.detach().cpu()),
        "box_loss": float(box_loss.detach().cpu()),
        "relation_loss": float(relation_loss.detach().cpu()),
        "pair_relation_loss": float(pair_relation_loss.detach().cpu()),
        "patch_relation_loss": float(patch_relation_loss.detach().cpu()),
        "relation_predicate_loss": float(relation_predicate_loss.detach().cpu()),
        "relation_ranking_loss": float(relation_ranking_loss.detach().cpu()),
        "relation_answer_graph_loss": float(relation_answer_graph_loss.detach().cpu()),
        "qa_loss": float(qa_loss.detach().cpu()),
        "scene_acc": float(scene_acc.detach().cpu()),
        "object_acc": float(object_acc.detach().cpu()),
        "relation_acc": float(relation_metrics["acc"].detach().cpu()),
        "relation_precision": float(relation_metrics["precision"].detach().cpu()),
        "relation_recall": float(relation_metrics["recall"].detach().cpu()),
        "relation_f1": float(relation_metrics["f1"].detach().cpu()),
        "relation_best_threshold": float(relation_metrics["best_threshold"].detach().cpu()),
        "relation_best_precision": float(relation_metrics["best_precision"].detach().cpu()),
        "relation_best_recall": float(relation_metrics["best_recall"].detach().cpu()),
        "relation_best_f1": float(relation_metrics["best_f1"].detach().cpu()),
        "relation_positive_ratio": float(relation_targets.pair_relationness_targets.mean().detach().cpu()),
        "relation_candidate_positive_ratio": float(
            relation_targets.pair_relationness_targets[pair_relation_valid_mask].mean().detach().cpu()
        ) if bool(pair_relation_valid_mask.any()) else 0.0,
        "relation_predicate_acc": float(relation_predicate_acc.detach().cpu()),
        "relation_ranking_acc": float(relation_ranking_metrics["relation_ranking_acc"]),
        "relation_ranking_query_count": float(relation_ranking_metrics["relation_ranking_query_count"]),
        "relation_ranking_skipped_duplicate_patch": float(
            relation_ranking_metrics["relation_ranking_skipped_duplicate_patch"]
        ),
        "relation_ranking_skipped_invalid": float(relation_ranking_metrics["relation_ranking_skipped_invalid"]),
        "relation_answer_graph_acc": float(relation_answer_graph_metrics["relation_answer_graph_acc"]),
        "relation_answer_graph_query_count": float(
            relation_answer_graph_metrics["relation_answer_graph_query_count"]
        ),
        "relation_answer_graph_skipped_parse": float(
            relation_answer_graph_metrics["relation_answer_graph_skipped_parse"]
        ),
        "relation_answer_graph_skipped_no_target": float(
            relation_answer_graph_metrics["relation_answer_graph_skipped_no_target"]
        ),
        "relation_answer_graph_skipped_invalid": float(
            relation_answer_graph_metrics["relation_answer_graph_skipped_invalid"]
        ),
        "patch_relation_acc": float(patch_relation_metrics["acc"].detach().cpu()),
        "patch_relation_f1": float(patch_relation_metrics["f1"].detach().cpu()),
        "patch_relation_best_f1": float(patch_relation_metrics["best_f1"].detach().cpu()),
        "patch_relation_predicate_acc": float(patch_relation_predicate_acc_tensor.detach().cpu()),
        "patch_relation_positive_ratio": float(relation_targets.relationness_targets.mean().detach().cpu()),
        "qa_acc": float(qa_acc.detach().cpu()),
        "qa_exact_match": float(qa_acc.detach().cpu()),
        "box_valid_ratio": float(patch_targets.box_valid_mask.mean().detach().cpu()),
        "relation_valid_ratio": float(relation_targets.pair_predicate_valid_mask.mean().detach().cpu()),
        "relation_candidate_valid_ratio": float(relation_targets.pair_relation_valid_mask.mean().detach().cpu()),
        "patch_relation_valid_ratio": float(relation_targets.predicate_valid_mask.mean().detach().cpu()),
        "relation_positive_weight": float(relation_positive_weight),
        "pair_relation_positive_weight": float(pair_relation_positive_weight),
        "relation_ranking_loss_weight": float(relation_ranking_loss_weight),
        "relation_answer_graph_loss_weight": float(relation_answer_graph_loss_weight),
        "qa_valid_ratio": float(qa_targets.answer_valid_mask.mean().detach().cpu()),
        "qa_answer_weight_mean": float(
            qa_answer_weight_tensor[qa_targets.answer_labels[qa_valid_mask]].mean().detach().cpu()
        ) if bool(qa_valid_mask.any()) and qa_answer_weight_tensor is not None else 1.0,
        "qa_type_weight_mean": float(
            qa_type_weight_tensor[qa_valid_mask].mean().detach().cpu()
        ) if bool(qa_valid_mask.any()) and qa_type_weight_tensor is not None else 1.0,
        "summary_schema_alignment_mean": summary_schema_alignment,
        "scene_schema_alignment_mean": scene_schema_alignment,
        "schema_chain_activated": schema_chain_activated,
    }
    return total_loss, metrics


def _evaluate(
    *,
    model,
    visual_data,
    dataset_info: Dict[str, Any],
    data_config,
    relation_predicate_to_index: Dict[str, int],
    qa_vocabulary,
    device: torch.device,
    batch_size: int,
    rng: random.Random,
    batches: int = 2,
    loss_weights: Dict[str, float] | None = None,
    qa_answer_weight_tensor: torch.Tensor | None = None,
    qa_type_loss_weights: Mapping[str, float] | None = None,
    qa_answer_type_to_indices: Mapping[str, Sequence[int]] | None = None,
    relation_positive_weight: float = 0.0,
    pair_relation_positive_weight: float = 1.0,
    relation_ranking_loss_weight: float = 0.0,
    relation_ranking_score_mode: str = "combined",
    relation_answer_graph_loss_weight: float = 0.0,
    relation_answer_graph_score_mode: str = "predicate_only",
) -> Dict[str, float]:
    was_training = model.training
    model.eval()
    rows = []
    with torch.no_grad():
        for _ in range(max(1, batches)):
            batch = _sample_dataset(
                visual_data,
                dataset_info,
                batch_size=batch_size,
                config=data_config,
                device=device,
                rng=rng,
            )
            _, metrics = _compute_losses(
                model=model,
                visual_data=visual_data,
                batch=batch,
                data_config=data_config,
                relation_predicate_to_index=relation_predicate_to_index,
                qa_vocabulary=qa_vocabulary,
                device=device,
                loss_weights=loss_weights
                or {
                    "scene": 1.0,
                    "object": 0.5,
                    "box": 2.0,
                    "relation": 0.35,
                    "relation_predicate": 0.5,
                    "qa": 0.5,
                },
                qa_answer_weight_tensor=qa_answer_weight_tensor,
                qa_type_loss_weights=qa_type_loss_weights,
                qa_answer_type_to_indices=qa_answer_type_to_indices,
                relation_positive_weight=relation_positive_weight,
                pair_relation_positive_weight=pair_relation_positive_weight,
                relation_ranking_loss_weight=relation_ranking_loss_weight,
                relation_ranking_score_mode=relation_ranking_score_mode,
                relation_answer_graph_loss_weight=relation_answer_graph_loss_weight,
                relation_answer_graph_score_mode=relation_answer_graph_score_mode,
            )
            rows.append(metrics)
    if was_training:
        model.train()
    keys = rows[0].keys()
    return {key: sum(float(row[key]) for row in rows) / len(rows) for key in keys}


def main() -> None:
    args = _parse_args()
    rng = _set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_path = _resolve_output_path(args.output_path, args.experiment_tag)
    checkpoint_dir = _resolve_checkpoint_dir(args.checkpoint_dir, args.experiment_tag)
    events_path = _resolve_events_path(args.events_path, checkpoint_dir)
    resume_path = Path(args.resume_from).resolve() if args.resume_from else None

    visual_arch = _load_module(
        "sb_visual_architecture",
        VISUAL_ARCH_PATH,
    )
    visual_data = _load_module(
        "sb_visual_data",
        VISUAL_DATA_PATH,
    )

    resume_payload = _torch_load(resume_path, device) if resume_path is not None else None
    if resume_payload is not None and "rng_state" in resume_payload:
        _restore_rng_state(resume_payload["rng_state"])
        rng = random.Random()
        rng.setstate(random.getstate())

    dataset_root = Path(args.dataset_root)
    scene_names = tuple(item.strip() for item in args.scene_names.split(",") if item.strip())
    if args.create_synthetic or not dataset_root.exists():
        train_dataset_info = visual_data.create_synthetic_scene_dataset(
            dataset_root,
            split=args.train_split,
            scene_names=scene_names,
            images_per_scene=args.images_per_scene,
            image_size=args.synthetic_image_size,
            include_annotations=True,
        )
        val_dataset_info = visual_data.create_synthetic_scene_dataset(
            dataset_root,
            split=args.val_split,
            scene_names=scene_names,
            images_per_scene=args.val_images_per_scene,
            image_size=args.synthetic_image_size,
            include_annotations=True,
        )
    else:
        train_annotation_manifest = dataset_root / f"{args.train_split}_annotations.jsonl"
        val_annotation_manifest = dataset_root / f"{args.val_split}_annotations.jsonl"
        train_dataset_info = visual_data.discover_visual_samples(
            dataset_root,
            split=args.train_split,
            annotation_manifest=train_annotation_manifest if train_annotation_manifest.exists() else None,
        )
        val_dataset_info = visual_data.discover_visual_samples(
            dataset_root,
            split=args.val_split,
            annotation_manifest=val_annotation_manifest if val_annotation_manifest.exists() else None,
        )

    train_samples = train_dataset_info["samples"]
    val_samples = val_dataset_info["samples"]
    if not train_samples:
        raise RuntimeError(f"no visual samples found in {dataset_root} split={args.train_split}")
    if not val_samples:
        raise RuntimeError(f"no visual samples found in {dataset_root} split={args.val_split}")

    data_config = visual_data.SBVisualDatasetConfig(image_size=args.image_size)
    scene_to_index = train_dataset_info["scene_to_index"]
    scene_classes = max((index for index in scene_to_index.values() if index >= 0), default=-1) + 1
    if resume_payload is not None:
        relation_predicate_to_index = dict(resume_payload["relation_predicate_to_index"])
        qa_vocabulary = _restore_qa_vocabulary(visual_data, resume_payload)
        qa_answer_type_to_indices = _restore_qa_answer_type_to_indices(resume_payload)
    else:
        relation_predicate_to_index = visual_data.build_relation_predicate_vocabulary(train_samples)
        qa_vocabulary = visual_data.build_visual_qa_vocabulary(
            train_samples,
            max_question_vocab_size=args.max_question_vocab_size,
        )
        qa_answer_type_to_indices = visual_data.build_qa_answer_type_constraints(
            train_samples,
            qa_vocabulary=qa_vocabulary,
        )

    model_config_kwargs = {
        "image_size": args.image_size,
        "patch_size": args.patch_size,
        "patch_stride": args.patch_stride,
        "d_model": args.d_model,
        "state_dim": args.state_dim,
        "schema_slots": args.schema_slots,
        "object_memory_slots": args.object_memory_slots,
        "relation_memory_slots": args.relation_memory_slots,
        "scene_memory_slots": args.scene_memory_slots,
        "summary_memory_slots": args.summary_memory_slots,
        "relation_vocab_size": max(len(relation_predicate_to_index), 1),
        "scene_classes": max(scene_classes, 1),
        "answer_vocab_size": max(qa_vocabulary.answer_vocab_size, 1),
        "question_vocab_size": max(qa_vocabulary.question_vocab_size, 2),
        "question_max_len": args.question_max_len,
    }
    if resume_payload is not None and "model_config" in resume_payload:
        model_config_kwargs.update(resume_payload["model_config"])

    model_config = visual_arch.SBVisualConfig(**model_config_kwargs)
    model = visual_arch.SBVisualCore(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    loss_weights = {
        "scene": float(args.scene_loss_weight),
        "object": float(args.object_loss_weight),
        "box": float(args.box_loss_weight),
        "relation": float(args.relation_loss_weight),
        "relation_predicate": float(args.relation_predicate_loss_weight),
        "qa": float(args.qa_loss_weight),
    }
    qa_type_weights = _resolve_qa_type_weights(args.qa_type_sampling_profile)
    qa_type_loss_weights = _resolve_qa_type_loss_weights(args.qa_type_loss_profile)
    qa_answer_weight_tensor = _build_qa_answer_weight_tensor(
        qa_vocabulary,
        power=float(args.qa_answer_weight_power),
        max_weight=float(args.qa_answer_weight_max),
        device=device,
    )

    history = []
    val_history = []
    best_val = {"step": 0, "total_loss": float("inf")}
    best_relation_val = {"step": 0, "relation_best_f1": -1.0}
    start_step = 0
    if resume_payload is not None:
        model.load_state_dict(resume_payload["model_state"])
        if "optimizer_state" in resume_payload:
            optimizer.load_state_dict(resume_payload["optimizer_state"])
        history = list(resume_payload.get("history", []))
        val_history = list(resume_payload.get("val_history", []))
        best_val = dict(resume_payload.get("best_val", best_val))
        best_relation_val = dict(resume_payload.get("best_relation_val", best_relation_val))
        start_step = int(resume_payload.get("step", 0))
        if "loss_weights" in resume_payload:
            loss_weights.update({key: float(value) for key, value in resume_payload["loss_weights"].items()})

    _write_event(
        events_path,
        "start",
        {
            "resume_from": str(resume_path) if resume_path is not None else "",
            "start_step": start_step,
            "target_steps": args.steps,
            "runtime": runtime_device_report(device),
            "train_profile": args.train_profile,
        },
    )

    for step in range(start_step + 1, args.steps + 1):
        model.train()
        batch = _sample_dataset(
            visual_data,
            train_dataset_info,
            batch_size=args.batch_size,
            config=data_config,
            device=device,
            rng=rng,
            qa_type_weights=qa_type_weights,
        )
        total_loss, metrics = _compute_losses(
            model=model,
            visual_data=visual_data,
            batch=batch,
            data_config=data_config,
            relation_predicate_to_index=relation_predicate_to_index,
            qa_vocabulary=qa_vocabulary,
            device=device,
            loss_weights=loss_weights,
            qa_answer_weight_tensor=qa_answer_weight_tensor,
            qa_type_loss_weights=qa_type_loss_weights,
            qa_answer_type_to_indices=qa_answer_type_to_indices,
            relation_positive_weight=args.relation_positive_weight,
            pair_relation_positive_weight=args.pair_relation_positive_weight,
            relation_ranking_loss_weight=args.relation_ranking_loss_weight,
            relation_ranking_score_mode=args.relation_ranking_score_mode,
            relation_answer_graph_loss_weight=args.relation_answer_graph_loss_weight,
            relation_answer_graph_score_mode=args.relation_answer_graph_score_mode,
        )

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        step_metrics = {"step": step, **metrics}
        history.append(step_metrics)
        _write_event(events_path, "train_step", step_metrics)
        if args.print_every > 0 and (step == 1 or step % args.print_every == 0 or step == args.steps):
            print(json.dumps(step_metrics, ensure_ascii=False))

        if args.eval_every > 0 and (step == 1 or step % args.eval_every == 0 or step == args.steps):
            eval_metrics = _evaluate(
                model=model,
                visual_data=visual_data,
                dataset_info=val_dataset_info,
                data_config=data_config,
                relation_predicate_to_index=relation_predicate_to_index,
                qa_vocabulary=qa_vocabulary,
                device=device,
                batch_size=args.batch_size,
                rng=rng,
                batches=args.max_val_batches,
                loss_weights=loss_weights,
                qa_answer_weight_tensor=qa_answer_weight_tensor,
                qa_type_loss_weights=qa_type_loss_weights,
                qa_answer_type_to_indices=qa_answer_type_to_indices,
                relation_positive_weight=args.relation_positive_weight,
                pair_relation_positive_weight=args.pair_relation_positive_weight,
                relation_ranking_loss_weight=args.relation_ranking_loss_weight,
                relation_ranking_score_mode=args.relation_ranking_score_mode,
                relation_answer_graph_loss_weight=args.relation_answer_graph_loss_weight,
                relation_answer_graph_score_mode=args.relation_answer_graph_score_mode,
            )
            eval_entry = {"step": step, **eval_metrics}
            val_history.append(eval_entry)
            _write_event(events_path, "val_step", eval_entry)
            if eval_entry["total_loss"] < best_val["total_loss"]:
                best_val = dict(eval_entry)
                _save_checkpoint(
                    checkpoint_dir / "best.pt",
                    model=model,
                    optimizer=optimizer,
                    args=args,
                    train_dataset_info=train_dataset_info,
                    val_dataset_info=val_dataset_info,
                    relation_predicate_to_index=relation_predicate_to_index,
                    qa_vocabulary=qa_vocabulary,
                    qa_answer_type_to_indices=qa_answer_type_to_indices,
                    step=step,
                    history=history,
                    val_history=val_history,
                    best_val=best_val,
                    best_relation_val=best_relation_val,
                    loss_weights=loss_weights,
                    rng_state=_capture_rng_state(),
                    events_path=events_path,
                )
                _write_event(events_path, "best_val", best_val)
            if float(eval_entry.get("relation_best_f1", 0.0)) > float(best_relation_val.get("relation_best_f1", -1.0)):
                best_relation_val = dict(eval_entry)
                _save_checkpoint(
                    checkpoint_dir / "best_relation.pt",
                    model=model,
                    optimizer=optimizer,
                    args=args,
                    train_dataset_info=train_dataset_info,
                    val_dataset_info=val_dataset_info,
                    relation_predicate_to_index=relation_predicate_to_index,
                    qa_vocabulary=qa_vocabulary,
                    qa_answer_type_to_indices=qa_answer_type_to_indices,
                    step=step,
                    history=history,
                    val_history=val_history,
                    best_val=best_val,
                    best_relation_val=best_relation_val,
                    loss_weights=loss_weights,
                    rng_state=_capture_rng_state(),
                    events_path=events_path,
                )
                _write_event(events_path, "best_relation_val", best_relation_val)
            if args.print_every > 0:
                print(json.dumps({"type": "val", **eval_entry}, ensure_ascii=False))

        if args.checkpoint_every > 0 and step % args.checkpoint_every == 0:
            checkpoint_path = checkpoint_dir / f"step_{step}.pt"
            _save_checkpoint(
                checkpoint_path,
                model=model,
                optimizer=optimizer,
                args=args,
                train_dataset_info=train_dataset_info,
                val_dataset_info=val_dataset_info,
                relation_predicate_to_index=relation_predicate_to_index,
                qa_vocabulary=qa_vocabulary,
                qa_answer_type_to_indices=qa_answer_type_to_indices,
                step=step,
                history=history,
                val_history=val_history,
                best_val=best_val,
                best_relation_val=best_relation_val,
                loss_weights=loss_weights,
                rng_state=_capture_rng_state(),
                events_path=events_path,
            )
            _write_event(events_path, "checkpoint", {"step": step, "path": str(checkpoint_path.resolve())})

        _save_checkpoint(
            checkpoint_dir / "last.pt",
            model=model,
            optimizer=optimizer,
            args=args,
            train_dataset_info=train_dataset_info,
            val_dataset_info=val_dataset_info,
            relation_predicate_to_index=relation_predicate_to_index,
            qa_vocabulary=qa_vocabulary,
            qa_answer_type_to_indices=qa_answer_type_to_indices,
            step=step,
            history=history,
            val_history=val_history,
            best_val=best_val,
            best_relation_val=best_relation_val,
            loss_weights=loss_weights,
            rng_state=_capture_rng_state(),
            events_path=events_path,
        )

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "runtime": runtime_device_report(device),
        "dataset": {
            "root": str(dataset_root.resolve()),
            "train_split": train_dataset_info["split"],
            "val_split": val_dataset_info["split"],
            "train_sample_count": len(train_samples),
            "val_sample_count": len(val_samples),
            "scene_to_index": scene_to_index,
            "train_annotation_manifest": train_dataset_info.get("annotation_manifest", ""),
            "val_annotation_manifest": val_dataset_info.get("annotation_manifest", ""),
            "relation_predicate_to_index": relation_predicate_to_index,
            "answer_to_index": qa_vocabulary.answer_stoi,
            "qa_answer_type_to_indices": {
                answer_type: list(indices)
                for answer_type, indices in qa_answer_type_to_indices.items()
            },
        },
        "config": {
            "train_profile": args.train_profile,
            "resume_from": str(resume_path) if resume_path is not None else "",
            "image_size": args.image_size,
            "patch_size": args.patch_size,
            "patch_stride": args.patch_stride,
            "d_model": args.d_model,
            "state_dim": args.state_dim,
            "schema_slots": args.schema_slots,
            "object_memory_slots": args.object_memory_slots,
            "relation_memory_slots": args.relation_memory_slots,
            "scene_memory_slots": args.scene_memory_slots,
            "summary_memory_slots": args.summary_memory_slots,
            "batch_size": args.batch_size,
            "steps": args.steps,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "question_max_len": args.question_max_len,
            "question_vocab_size": qa_vocabulary.question_vocab_size,
            "answer_vocab_size": qa_vocabulary.answer_vocab_size,
            "qa_answer_counts": qa_vocabulary.answer_counts,
            "loss_weights": loss_weights,
            "relation_positive_weight": args.relation_positive_weight,
            "pair_relation_positive_weight": args.pair_relation_positive_weight,
            "relation_ranking_loss_weight": args.relation_ranking_loss_weight,
            "relation_ranking_score_mode": args.relation_ranking_score_mode,
            "relation_answer_graph_loss_weight": args.relation_answer_graph_loss_weight,
            "relation_answer_graph_score_mode": args.relation_answer_graph_score_mode,
            "qa_answer_weight_power": args.qa_answer_weight_power,
            "qa_answer_weight_max": args.qa_answer_weight_max,
            "qa_type_sampling_profile": args.qa_type_sampling_profile,
            "qa_type_sampling_weights": qa_type_weights or {},
            "qa_type_loss_profile": args.qa_type_loss_profile,
            "qa_type_loss_weights": qa_type_loss_weights or {},
            "qa_answer_type_to_indices": {
                answer_type: list(indices)
                for answer_type, indices in qa_answer_type_to_indices.items()
            },
            "qa_answer_weight_tensor": qa_answer_weight_tensor.detach().cpu().tolist(),
            "eval_every": args.eval_every,
            "max_val_batches": args.max_val_batches,
            "checkpoint_every": args.checkpoint_every,
        },
        "events_path": str(events_path.resolve()),
        "checkpoint_dir": str(checkpoint_dir.resolve()),
        "history": history,
        "val_history": val_history,
        "best_val": best_val,
        "best_relation_val": best_relation_val,
        "final": history[-1] if history else {},
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_event(
        events_path,
        "finish",
        {
            "completed_steps": len(history),
            "best_val_total_loss": best_val["total_loss"],
            "best_relation_val_f1": best_relation_val.get("relation_best_f1", 0.0),
            "report_path": str(output_path.resolve()),
        },
    )
    print(f"report_path={output_path}")


if __name__ == "__main__":
    main()
