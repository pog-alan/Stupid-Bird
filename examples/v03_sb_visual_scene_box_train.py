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
    parser.add_argument("--relation-predicate-loss-weight", type=float, default=0.5)
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
            "grounded_reasoning": 1.5,
            "causal": 1.35,
            "*": 1.0,
        }
    if profile_key == "relation_recovery":
        return {
            "classification": 0.45,
            "boolean": 0.7,
            "relation": 2.2,
            "grounded_reasoning": 1.6,
            "causal": 1.45,
            "*": 1.0,
        }
    if profile_key == "balanced_relation_grounded":
        return {
            "classification": 0.45,
            "boolean": 0.9,
            "relation": 1.7,
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
            "grounded_reasoning": 1.35,
            "causal": 1.25,
            "*": 1.0,
        }
    if profile_key == "relation_hard":
        return {
            "classification": 0.65,
            "boolean": 0.75,
            "relation": 2.3,
            "grounded_reasoning": 1.5,
            "causal": 1.35,
            "*": 1.0,
        }
    if profile_key == "balanced_relation_grounded":
        return {
            "classification": 0.9,
            "boolean": 0.9,
            "relation": 1.5,
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


def _safe_divide(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    return numerator / denominator.clamp_min(1.0)


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

    relation_loss = F.binary_cross_entropy_with_logits(
        forward["relation_scores"], relation_targets.relationness_targets
    )
    relation_valid_mask = relation_targets.predicate_valid_mask > 0.5
    if bool(relation_valid_mask.any()):
        relation_predicate_loss = F.cross_entropy(
            forward["relation_predicate_logits"][relation_valid_mask],
            relation_targets.predicate_targets[relation_valid_mask],
        )
    else:
        relation_predicate_loss = torch.zeros((), device=device)

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
        + loss_weights["qa"] * qa_loss
    )

    with torch.no_grad():
        scene_acc = (forward["scene_logits"].argmax(dim=-1) == patch_targets.scene_labels).float().mean()
        object_pred = (torch.sigmoid(forward["object_scores"]) > 0.5).float()
        object_acc = (object_pred == patch_targets.objectness_targets).float().mean()
        relation_pred = torch.sigmoid(forward["relation_scores"]) > 0.5
        relation_target = relation_targets.relationness_targets > 0.5
        relation_acc = (relation_pred == relation_target).float().mean()
        relation_tp = (relation_pred & relation_target).sum().float()
        relation_fp = (relation_pred & (~relation_target)).sum().float()
        relation_fn = ((~relation_pred) & relation_target).sum().float()
        relation_precision = _safe_divide(relation_tp, relation_tp + relation_fp)
        relation_recall = _safe_divide(relation_tp, relation_tp + relation_fn)
        relation_f1 = _safe_divide(2.0 * relation_precision * relation_recall, relation_precision + relation_recall)
        if bool(relation_valid_mask.any()):
            relation_predicate_acc = (
                forward["relation_predicate_logits"][relation_valid_mask].argmax(dim=-1)
                == relation_targets.predicate_targets[relation_valid_mask]
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
        "relation_predicate_loss": float(relation_predicate_loss.detach().cpu()),
        "qa_loss": float(qa_loss.detach().cpu()),
        "scene_acc": float(scene_acc.detach().cpu()),
        "object_acc": float(object_acc.detach().cpu()),
        "relation_acc": float(relation_acc.detach().cpu()),
        "relation_precision": float(relation_precision.detach().cpu()),
        "relation_recall": float(relation_recall.detach().cpu()),
        "relation_f1": float(relation_f1.detach().cpu()),
        "relation_positive_ratio": float(relation_targets.relationness_targets.mean().detach().cpu()),
        "relation_predicate_acc": float(relation_predicate_acc.detach().cpu()),
        "qa_acc": float(qa_acc.detach().cpu()),
        "qa_exact_match": float(qa_acc.detach().cpu()),
        "box_valid_ratio": float(patch_targets.box_valid_mask.mean().detach().cpu()),
        "relation_valid_ratio": float(relation_targets.predicate_valid_mask.mean().detach().cpu()),
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
    start_step = 0
    if resume_payload is not None:
        model.load_state_dict(resume_payload["model_state"])
        if "optimizer_state" in resume_payload:
            optimizer.load_state_dict(resume_payload["optimizer_state"])
        history = list(resume_payload.get("history", []))
        val_history = list(resume_payload.get("val_history", []))
        best_val = dict(resume_payload.get("best_val", best_val))
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
                    loss_weights=loss_weights,
                    rng_state=_capture_rng_state(),
                    events_path=events_path,
                )
                _write_event(events_path, "best_val", best_val)
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
            "report_path": str(output_path.resolve()),
        },
    )
    print(f"report_path={output_path}")


if __name__ == "__main__":
    main()
