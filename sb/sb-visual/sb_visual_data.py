from __future__ import annotations

import json
import os
from collections import Counter
from dataclasses import dataclass, field, replace
import random
import shutil
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import Tensor


@dataclass(frozen=True)
class SBVisualDatasetConfig:
    image_size: int = 224
    normalize_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    normalize_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    allowed_extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

    def validate(self) -> None:
        if self.image_size <= 0:
            raise ValueError("image_size must be positive.")
        if len(self.normalize_mean) != 3 or len(self.normalize_std) != 3:
            raise ValueError("normalize_mean and normalize_std must have 3 channels.")
        if any(value <= 0.0 for value in self.normalize_std):
            raise ValueError("normalize_std must be positive.")


@dataclass(frozen=True)
class SBVisualSample:
    image_path: str
    scene_name: str
    scene_index: int
    split: str
    original_size: Tuple[int, int]
    boxes: Tuple["SBVisualBoxAnnotation", ...] = ()
    relations: Tuple["SBVisualRelationAnnotation", ...] = ()
    qa_pairs: Tuple["SBVisualQAPair", ...] = ()
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class SBVisualBatch:
    images: Tensor
    scene_labels: Tensor
    image_paths: Tuple[str, ...]
    scene_names: Tuple[str, ...]
    original_sizes: Tuple[Tuple[int, int], ...]
    boxes_xyxy: Tuple[Tensor, ...]
    box_labels: Tuple[Tuple[str, ...], ...]
    relations: Tuple[Tuple["SBVisualRelationAnnotation", ...], ...]
    qa_pairs: Tuple[Tuple["SBVisualQAPair", ...], ...]
    metadata: Tuple[Dict[str, object], ...]

    def moved_to(self, device: torch.device | str) -> "SBVisualBatch":
        return SBVisualBatch(
            images=self.images.to(device),
            scene_labels=self.scene_labels.to(device),
            image_paths=self.image_paths,
            scene_names=self.scene_names,
            original_sizes=self.original_sizes,
            boxes_xyxy=tuple(item.to(device) for item in self.boxes_xyxy),
            box_labels=self.box_labels,
            relations=self.relations,
            qa_pairs=self.qa_pairs,
            metadata=self.metadata,
        )


@dataclass(frozen=True)
class SBVisualPatchTargets:
    scene_labels: Tensor
    objectness_targets: Tensor
    box_targets: Tensor
    box_valid_mask: Tensor


@dataclass(frozen=True)
class SBVisualRelationTargets:
    relationness_targets: Tensor
    predicate_targets: Tensor
    predicate_valid_mask: Tensor


@dataclass(frozen=True)
class SBVisualQATargets:
    question_ids: Tensor
    question_mask: Tensor
    answer_labels: Tensor
    answer_valid_mask: Tensor
    answer_type_mask: Tensor
    questions: Tuple[str, ...]
    answers: Tuple[str, ...]
    answer_types: Tuple[str, ...]


@dataclass(frozen=True)
class SBVisualQAVocabulary:
    question_stoi: Dict[str, int]
    question_itos: Tuple[str, ...]
    answer_stoi: Dict[str, int]
    answer_itos: Tuple[str, ...]
    answer_counts: Dict[str, int] = field(default_factory=dict)
    pad_token: str = "<pad>"
    unk_token: str = "<unk>"

    @property
    def pad_id(self) -> int:
        return int(self.question_stoi[self.pad_token])

    @property
    def unk_id(self) -> int:
        return int(self.question_stoi[self.unk_token])

    @property
    def question_vocab_size(self) -> int:
        return len(self.question_itos)

    @property
    def answer_vocab_size(self) -> int:
        return len(self.answer_itos)


@dataclass(frozen=True)
class SBVisualBoxAnnotation:
    label: str
    xyxy: Tuple[float, float, float, float]
    attributes: Tuple[str, ...] = ()


@dataclass(frozen=True)
class SBVisualRelationAnnotation:
    subject: int
    predicate: str
    object: int
    confidence: float = 1.0


@dataclass(frozen=True)
class SBVisualQAPair:
    question: str
    answer: str
    answer_type: str = "free_form"


@dataclass(frozen=True)
class SBVisualAnnotationRecord:
    image_key: str
    split: str
    scene_name: str
    boxes: Tuple[SBVisualBoxAnnotation, ...]
    relations: Tuple[SBVisualRelationAnnotation, ...]
    qa_pairs: Tuple[SBVisualQAPair, ...]
    metadata: Dict[str, object]


def _is_image_file(path: Path, allowed_extensions: Sequence[str]) -> bool:
    return path.is_file() and path.suffix.lower() in set(ext.lower() for ext in allowed_extensions)


def _discover_base_dir(root: Path, split: str) -> Tuple[Path, str]:
    if split:
        split_dir = root / split
        if split_dir.exists():
            return split_dir, split
    return root, split or "unspecified"


def _normalize_image_key(path: str | Path) -> str:
    return Path(path).as_posix().lower()


def _parse_box_annotation(payload: Mapping[str, object]) -> SBVisualBoxAnnotation:
    coords = tuple(float(value) for value in payload.get("xyxy", ()))
    if len(coords) != 4:
        raise ValueError("box annotation requires xyxy with four coordinates.")
    return SBVisualBoxAnnotation(
        label=str(payload.get("label", "object")),
        xyxy=(coords[0], coords[1], coords[2], coords[3]),
        attributes=tuple(str(item) for item in payload.get("attributes", ())),
    )


def _parse_relation_annotation(payload: Mapping[str, object]) -> SBVisualRelationAnnotation:
    return SBVisualRelationAnnotation(
        subject=int(payload.get("subject", 0)),
        predicate=str(payload.get("predicate", "related_to")),
        object=int(payload.get("object", 0)),
        confidence=float(payload.get("confidence", 1.0)),
    )


def _parse_qa_annotation(payload: Mapping[str, object]) -> SBVisualQAPair:
    return SBVisualQAPair(
        question=str(payload.get("question", "")).strip(),
        answer=str(payload.get("answer", "")).strip(),
        answer_type=str(payload.get("answer_type", "free_form")),
    )


def load_visual_annotation_manifest(
    path: str | Path,
    *,
    dataset_root: str | Path | None = None,
) -> Dict[str, SBVisualAnnotationRecord]:
    manifest_path = Path(path)
    dataset_root_path = Path(dataset_root).resolve() if dataset_root is not None else None
    rows: List[Dict[str, object]] = []
    if manifest_path.suffix.lower() == ".jsonl":
        with manifest_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    else:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            rows = list(payload)
        else:
            rows = list(payload.get("records", []))

    records: Dict[str, SBVisualAnnotationRecord] = {}
    for row in rows:
        image_rel_path = str(row.get("image_rel_path", "")).strip()
        image_path = str(row.get("image_path", "")).strip()
        if image_path:
            resolved_path = Path(image_path)
            if not resolved_path.is_absolute() and dataset_root_path is not None:
                resolved_path = (dataset_root_path / resolved_path).resolve()
            image_key = _normalize_image_key(resolved_path)
        elif image_rel_path:
            image_key = _normalize_image_key(image_rel_path)
        else:
            raise ValueError("annotation record requires image_path or image_rel_path.")

        records[image_key] = SBVisualAnnotationRecord(
            image_key=image_key,
            split=str(row.get("split", "")).strip(),
            scene_name=str(row.get("scene_name", "unknown")).strip(),
            boxes=tuple(_parse_box_annotation(item) for item in row.get("boxes", ())),
            relations=tuple(_parse_relation_annotation(item) for item in row.get("relations", ())),
            qa_pairs=tuple(_parse_qa_annotation(item) for item in row.get("qa_pairs", ())),
            metadata=dict(row.get("metadata", {})),
        )
    return records


def discover_visual_samples(
    root: str | Path,
    *,
    split: str = "train",
    allowed_extensions: Sequence[str] | None = None,
    annotation_manifest: str | Path | None = None,
) -> Dict[str, object]:
    root_path = Path(root)
    extensions = tuple(allowed_extensions or SBVisualDatasetConfig().allowed_extensions)
    base_dir, resolved_split = _discover_base_dir(root_path, split)
    annotation_records = (
        load_visual_annotation_manifest(annotation_manifest, dataset_root=root_path)
        if annotation_manifest is not None
        else {}
    )
    class_dirs = sorted(path for path in base_dir.iterdir() if path.is_dir()) if base_dir.exists() else []

    if class_dirs:
        scene_to_index = {path.name: index for index, path in enumerate(class_dirs)}
        samples: List[SBVisualSample] = []
        for class_dir in class_dirs:
            scene_name = class_dir.name
            for image_path in sorted(class_dir.rglob("*")):
                if not _is_image_file(image_path, extensions):
                    continue
                with Image.open(image_path) as image:
                    width, height = image.size
                resolved_path = image_path.resolve()
                relative_key = _normalize_image_key(resolved_path.relative_to(root_path.resolve()))
                absolute_key = _normalize_image_key(resolved_path)
                annotation = annotation_records.get(relative_key) or annotation_records.get(absolute_key)
                samples.append(
                    SBVisualSample(
                        image_path=str(resolved_path),
                        scene_name=scene_name,
                        scene_index=scene_to_index[scene_name],
                        split=resolved_split,
                        original_size=(width, height),
                        boxes=annotation.boxes if annotation is not None else (),
                        relations=annotation.relations if annotation is not None else (),
                        qa_pairs=annotation.qa_pairs if annotation is not None else (),
                        metadata={
                            "class_dir": str(class_dir.resolve()),
                            **(annotation.metadata if annotation is not None else {}),
                        },
                    )
                )
        return {
            "root": str(root_path.resolve()),
            "base_dir": str(base_dir.resolve()),
            "split": resolved_split,
            "scene_to_index": scene_to_index,
            "samples": samples,
            "annotation_manifest": str(Path(annotation_manifest).resolve()) if annotation_manifest is not None else "",
        }

    flat_images = sorted(path for path in base_dir.rglob("*") if _is_image_file(path, extensions))
    samples = []
    for image_path in flat_images:
        with Image.open(image_path) as image:
            width, height = image.size
        resolved_path = image_path.resolve()
        relative_key = _normalize_image_key(resolved_path.relative_to(root_path.resolve()))
        absolute_key = _normalize_image_key(resolved_path)
        annotation = annotation_records.get(relative_key) or annotation_records.get(absolute_key)
        samples.append(
            SBVisualSample(
                image_path=str(resolved_path),
                scene_name=annotation.scene_name if annotation is not None and annotation.scene_name else "unknown",
                scene_index=-1,
                split=resolved_split,
                original_size=(width, height),
                boxes=annotation.boxes if annotation is not None else (),
                relations=annotation.relations if annotation is not None else (),
                qa_pairs=annotation.qa_pairs if annotation is not None else (),
                metadata=dict(annotation.metadata) if annotation is not None else {},
            )
        )
    return {
        "root": str(root_path.resolve()),
        "base_dir": str(base_dir.resolve()),
        "split": resolved_split,
        "scene_to_index": {"unknown": -1},
        "samples": samples,
        "annotation_manifest": str(Path(annotation_manifest).resolve()) if annotation_manifest is not None else "",
    }


def load_image_tensor(path: str | Path, config: SBVisualDatasetConfig) -> Tensor:
    config.validate()
    with Image.open(path) as image:
        image = image.convert("RGB")
        image = image.resize((config.image_size, config.image_size), Image.BILINEAR)
        array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
    mean = torch.tensor(config.normalize_mean, dtype=tensor.dtype).view(3, 1, 1)
    std = torch.tensor(config.normalize_std, dtype=tensor.dtype).view(3, 1, 1)
    return (tensor - mean) / std


def _scale_box_xyxy(
    xyxy: Tuple[float, float, float, float],
    *,
    from_size: Tuple[int, int],
    to_size: int,
) -> Tuple[float, float, float, float]:
    original_width, original_height = from_size
    scale_x = float(to_size) / float(max(original_width, 1))
    scale_y = float(to_size) / float(max(original_height, 1))
    x1, y1, x2, y2 = xyxy
    return (x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y)


def _box_center_to_patch_index(
    xyxy: Tuple[float, float, float, float],
    *,
    image_size: int,
    patch_grid: Tuple[int, int],
) -> int:
    grid_h, grid_w = patch_grid
    x1, y1, x2, y2 = xyxy
    center_x = 0.5 * (x1 + x2)
    center_y = 0.5 * (y1 + y2)
    col = min(grid_w - 1, max(0, int(center_x / max(float(image_size), 1.0) * grid_w)))
    row = min(grid_h - 1, max(0, int(center_y / max(float(image_size), 1.0) * grid_h)))
    return row * grid_w + col


def _boxes_to_tensor(sample: SBVisualSample, image_size: int) -> Tensor:
    if not sample.boxes:
        return torch.zeros(0, 4, dtype=torch.float32)
    scaled = [
        _scale_box_xyxy(box.xyxy, from_size=sample.original_size, to_size=image_size)
        for box in sample.boxes
    ]
    return torch.tensor(scaled, dtype=torch.float32)


def build_visual_batch(
    samples: Sequence[SBVisualSample],
    *,
    config: SBVisualDatasetConfig,
    device: torch.device | str = "cpu",
) -> SBVisualBatch:
    if not samples:
        raise ValueError("samples must not be empty.")
    images = torch.stack([load_image_tensor(sample.image_path, config) for sample in samples], dim=0).to(device)
    scene_labels = torch.tensor([sample.scene_index for sample in samples], dtype=torch.long, device=device)
    boxes_xyxy = tuple(_boxes_to_tensor(sample, config.image_size).to(device) for sample in samples)
    return SBVisualBatch(
        images=images,
        scene_labels=scene_labels,
        image_paths=tuple(sample.image_path for sample in samples),
        scene_names=tuple(sample.scene_name for sample in samples),
        original_sizes=tuple(sample.original_size for sample in samples),
        boxes_xyxy=boxes_xyxy,
        box_labels=tuple(tuple(box.label for box in sample.boxes) for sample in samples),
        relations=tuple(sample.relations for sample in samples),
        qa_pairs=tuple(sample.qa_pairs for sample in samples),
        metadata=tuple(sample.metadata for sample in samples),
    )


def sample_visual_batch(
    samples: Sequence[SBVisualSample],
    *,
    batch_size: int,
    config: SBVisualDatasetConfig,
    device: torch.device | str = "cpu",
    rng,
    qa_type_weights: Mapping[str, float] | None = None,
) -> SBVisualBatch:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if not samples:
        raise ValueError("samples must not be empty.")
    chosen = []
    for _ in range(batch_size):
        sample = samples[rng.randrange(len(samples))]
        if len(sample.qa_pairs) > 1:
            if qa_type_weights:
                weights = [
                    float(qa_type_weights.get(qa.answer_type, qa_type_weights.get("*", 1.0)))
                    for qa in sample.qa_pairs
                ]
                if sum(weights) <= 0.0:
                    qa_index = rng.randrange(len(sample.qa_pairs))
                else:
                    qa_index = rng.choices(range(len(sample.qa_pairs)), weights=weights, k=1)[0]
            else:
                qa_index = rng.randrange(len(sample.qa_pairs))
            metadata = dict(sample.metadata)
            metadata["sampled_qa_index"] = qa_index
            metadata["sampled_qa_answer_type"] = sample.qa_pairs[qa_index].answer_type
            if qa_type_weights:
                metadata["sampled_qa_weight"] = float(
                    qa_type_weights.get(
                        sample.qa_pairs[qa_index].answer_type,
                        qa_type_weights.get("*", 1.0),
                    )
                )
            sample = replace(sample, qa_pairs=(sample.qa_pairs[qa_index],), metadata=metadata)
        chosen.append(sample)
    return build_visual_batch(chosen, config=config, device=device)


def encode_visual_batch(batch: SBVisualBatch, patch_encoder) -> Dict[str, Tensor | Tuple[int, int]]:
    encoded = patch_encoder(batch.images)
    return {
        "signals": encoded.signals,
        "positions": encoded.positions,
        "patch_grid": encoded.patch_grid,
        "scene_labels": batch.scene_labels,
        "boxes_xyxy": batch.boxes_xyxy,
        "box_labels": batch.box_labels,
        "relations": batch.relations,
        "qa_pairs": batch.qa_pairs,
    }


def build_patch_targets(
    batch: SBVisualBatch,
    *,
    patch_grid: Tuple[int, int],
    image_size: int,
    device: torch.device | str | None = None,
) -> SBVisualPatchTargets:
    grid_h, grid_w = patch_grid
    patch_count = int(grid_h * grid_w)
    target_device = device if device is not None else batch.images.device
    objectness_targets = torch.zeros(len(batch.image_paths), patch_count, device=target_device, dtype=torch.float32)
    box_targets = torch.zeros(len(batch.image_paths), patch_count, 4, device=target_device, dtype=torch.float32)
    box_valid_mask = torch.zeros(len(batch.image_paths), patch_count, device=target_device, dtype=torch.float32)

    step_x = float(image_size) / float(max(grid_w, 1))
    step_y = float(image_size) / float(max(grid_h, 1))
    centers = []
    for row in range(grid_h):
        for col in range(grid_w):
            centers.append(((col + 0.5) * step_x, (row + 0.5) * step_y))

    for batch_index, boxes in enumerate(batch.boxes_xyxy):
        if boxes.numel() == 0:
            continue
        boxes_cpu = boxes.detach().cpu()
        areas = (boxes_cpu[:, 2] - boxes_cpu[:, 0]).clamp_min(1e-6) * (boxes_cpu[:, 3] - boxes_cpu[:, 1]).clamp_min(1e-6)
        for patch_index, (center_x, center_y) in enumerate(centers):
            inside = (
                (boxes_cpu[:, 0] <= center_x)
                & (boxes_cpu[:, 2] >= center_x)
                & (boxes_cpu[:, 1] <= center_y)
                & (boxes_cpu[:, 3] >= center_y)
            )
            if not bool(inside.any()):
                continue
            candidate_indices = torch.nonzero(inside, as_tuple=False).squeeze(-1)
            if candidate_indices.numel() == 1:
                chosen_index = int(candidate_indices.item())
            else:
                chosen_index = int(candidate_indices[areas[candidate_indices].argmin()].item())
            chosen_box = boxes_cpu[chosen_index]
            objectness_targets[batch_index, patch_index] = 1.0
            box_valid_mask[batch_index, patch_index] = 1.0
            box_targets[batch_index, patch_index] = torch.tensor(
                [
                    float(chosen_box[0]) / float(image_size),
                    float(chosen_box[1]) / float(image_size),
                    float(chosen_box[2]) / float(image_size),
                    float(chosen_box[3]) / float(image_size),
                ],
                dtype=torch.float32,
                device=target_device,
            )

    return SBVisualPatchTargets(
        scene_labels=batch.scene_labels.to(target_device),
        objectness_targets=objectness_targets,
        box_targets=box_targets,
        box_valid_mask=box_valid_mask,
    )


def build_relation_predicate_vocabulary(samples: Sequence[SBVisualSample]) -> Dict[str, int]:
    predicates = sorted({relation.predicate for sample in samples for relation in sample.relations})
    vocab = {"<none>": 0}
    for predicate in predicates:
        if predicate not in vocab:
            vocab[predicate] = len(vocab)
    return vocab


def build_relation_targets(
    batch: SBVisualBatch,
    *,
    patch_grid: Tuple[int, int],
    image_size: int,
    predicate_to_index: Mapping[str, int],
    device: torch.device | str | None = None,
) -> SBVisualRelationTargets:
    grid_h, grid_w = patch_grid
    patch_count = int(grid_h * grid_w)
    target_device = device if device is not None else batch.images.device
    relationness_targets = torch.zeros(len(batch.image_paths), patch_count, device=target_device, dtype=torch.float32)
    predicate_targets = torch.zeros(len(batch.image_paths), patch_count, device=target_device, dtype=torch.long)
    predicate_valid_mask = torch.zeros(len(batch.image_paths), patch_count, device=target_device, dtype=torch.float32)

    for batch_index, relations in enumerate(batch.relations):
        if not relations or not batch.boxes_xyxy[batch_index].numel():
            continue
        boxes = batch.boxes_xyxy[batch_index].detach().cpu()
        for relation in relations:
            if relation.subject < 0 or relation.subject >= boxes.shape[0]:
                continue
            subject_box = tuple(float(value) for value in boxes[relation.subject].tolist())
            patch_index = _box_center_to_patch_index(subject_box, image_size=image_size, patch_grid=patch_grid)
            relationness_targets[batch_index, patch_index] = 1.0
            predicate_valid_mask[batch_index, patch_index] = 1.0
            predicate_targets[batch_index, patch_index] = int(predicate_to_index.get(relation.predicate, 0))

    return SBVisualRelationTargets(
        relationness_targets=relationness_targets,
        predicate_targets=predicate_targets,
        predicate_valid_mask=predicate_valid_mask,
    )


def build_visual_qa_vocabulary(
    samples: Sequence[SBVisualSample],
    *,
    max_question_vocab_size: int = 256,
) -> SBVisualQAVocabulary:
    counter: Counter[str] = Counter()
    answer_counter: Counter[str] = Counter()
    answers = {"<unk>"}
    for sample in samples:
        for qa_pair in sample.qa_pairs:
            counter.update(qa_pair.question)
            answers.add(qa_pair.answer)
            answer_counter.update([qa_pair.answer])
    special_tokens = ["<pad>", "<unk>"]
    question_tokens = list(special_tokens)
    for char, _ in counter.most_common(max(0, max_question_vocab_size - len(special_tokens))):
        if char not in question_tokens:
            question_tokens.append(char)
    question_stoi = {token: index for index, token in enumerate(question_tokens)}
    answer_itos = tuple(sorted(answers))
    answer_stoi = {token: index for index, token in enumerate(answer_itos)}
    return SBVisualQAVocabulary(
        question_stoi=question_stoi,
        question_itos=tuple(question_tokens),
        answer_stoi=answer_stoi,
        answer_itos=answer_itos,
        answer_counts={answer: int(answer_counter.get(answer, 0)) for answer in answer_itos},
    )


def build_qa_answer_type_constraints(
    samples: Sequence[SBVisualSample],
    *,
    qa_vocabulary: SBVisualQAVocabulary,
) -> Dict[str, Tuple[int, ...]]:
    constraints: Dict[str, set[int]] = {}
    for sample in samples:
        for qa_pair in sample.qa_pairs:
            answer_type = str(qa_pair.answer_type or "").strip() or "unknown"
            answer_id = int(qa_vocabulary.answer_stoi.get(qa_pair.answer, 0))
            constraints.setdefault(answer_type, set()).add(answer_id)
    return {
        answer_type: tuple(sorted(indices))
        for answer_type, indices in sorted(constraints.items())
    }


def build_qa_targets(
    batch: SBVisualBatch,
    *,
    qa_vocabulary: SBVisualQAVocabulary,
    max_question_len: int,
    answer_type_to_indices: Mapping[str, Sequence[int]] | None = None,
    device: torch.device | str | None = None,
) -> SBVisualQATargets:
    target_device = device if device is not None else batch.images.device
    question_ids = torch.full(
        (len(batch.image_paths), max_question_len),
        qa_vocabulary.pad_id,
        device=target_device,
        dtype=torch.long,
    )
    question_mask = torch.zeros(len(batch.image_paths), max_question_len, device=target_device, dtype=torch.float32)
    answer_labels = torch.zeros(len(batch.image_paths), device=target_device, dtype=torch.long)
    answer_valid_mask = torch.zeros(len(batch.image_paths), device=target_device, dtype=torch.float32)
    answer_type_mask = torch.ones(
        len(batch.image_paths),
        qa_vocabulary.answer_vocab_size,
        device=target_device,
        dtype=torch.float32,
    )
    questions: List[str] = []
    answers: List[str] = []
    answer_types: List[str] = []

    for batch_index, qa_pairs in enumerate(batch.qa_pairs):
        if not qa_pairs:
            questions.append("")
            answers.append("")
            answer_types.append("")
            answer_type_mask[batch_index].zero_()
            continue
        qa_pair = qa_pairs[0]
        question = qa_pair.question
        answer = qa_pair.answer
        answer_type = str(qa_pair.answer_type or "").strip() or "unknown"
        encoded = [qa_vocabulary.question_stoi.get(char, qa_vocabulary.unk_id) for char in question[:max_question_len]]
        if encoded:
            question_ids[batch_index, : len(encoded)] = torch.tensor(encoded, device=target_device, dtype=torch.long)
            question_mask[batch_index, : len(encoded)] = 1.0
        answer_labels[batch_index] = int(qa_vocabulary.answer_stoi.get(answer, 0))
        answer_valid_mask[batch_index] = 1.0
        if answer_type_to_indices:
            allowed = tuple(int(index) for index in answer_type_to_indices.get(answer_type, ()))
            if allowed:
                answer_type_mask[batch_index].zero_()
                answer_type_mask[batch_index, list(allowed)] = 1.0
        questions.append(question)
        answers.append(answer)
        answer_types.append(answer_type)

    return SBVisualQATargets(
        question_ids=question_ids,
        question_mask=question_mask,
        answer_labels=answer_labels,
        answer_valid_mask=answer_valid_mask,
        answer_type_mask=answer_type_mask,
        questions=tuple(questions),
        answers=tuple(answers),
        answer_types=tuple(answer_types),
    )


def create_synthetic_scene_dataset(
    root: str | Path,
    *,
    split: str = "train",
    scene_names: Sequence[str] = ("stacked", "spill", "orderly"),
    images_per_scene: int = 4,
    image_size: int = 96,
    include_annotations: bool = True,
) -> Dict[str, object]:
    root_path = Path(root)
    split_dir = root_path / split
    split_dir.mkdir(parents=True, exist_ok=True)
    palette = {
        "stacked": ((194, 124, 64), (110, 76, 44)),
        "spill": ((80, 146, 212), (30, 55, 88)),
        "orderly": ((112, 190, 116), (55, 103, 59)),
        "rural": ((108, 176, 92), (158, 123, 76)),
        "urban": ((144, 156, 176), (78, 88, 102)),
    }
    annotation_records: List[Dict[str, object]] = []
    for scene_name in scene_names:
        scene_dir = split_dir / scene_name
        scene_dir.mkdir(parents=True, exist_ok=True)
        primary, secondary = palette.get(scene_name, ((180, 180, 180), (90, 90, 90)))
        for index in range(images_per_scene):
            image = Image.new("RGB", (image_size, image_size), (240, 240, 240))
            draw = ImageDraw.Draw(image)
            boxes: List[Dict[str, object]] = []
            relations: List[Dict[str, object]] = []
            qa_pairs: List[Dict[str, object]] = [
                {
                    "question": "what scene is shown?",
                    "answer": scene_name,
                    "answer_type": "classification",
                }
            ]
            scene_key = scene_name.lower()
            if "rural" in scene_key:
                sky_box = (0, 0, image_size, int(image_size * 0.35))
                field_box = (0, int(image_size * 0.55), image_size, image_size)
                road_box = (0, int(image_size * 0.42), image_size, int(image_size * 0.62))
                house_box = (12, int(image_size * 0.36), 36, int(image_size * 0.62))
                tree_box = (image_size - 34, int(image_size * 0.28), image_size - 12, int(image_size * 0.62))
                livestock_box = (int(image_size * 0.48), int(image_size * 0.62), int(image_size * 0.66), int(image_size * 0.78))

                draw.rectangle(sky_box, fill=(196, 225, 255))
                draw.rectangle(field_box, fill=primary)
                draw.rectangle(road_box, fill=(193, 171, 138))
                draw.rectangle(house_box, fill=secondary)
                draw.polygon(
                    [
                        (house_box[0], house_box[1]),
                        ((house_box[0] + house_box[2]) / 2.0, house_box[1] - 12),
                        (house_box[2], house_box[1]),
                    ],
                    fill=(120, 72, 52),
                )
                draw.rectangle((tree_box[0] + 8, tree_box[1] + 18, tree_box[0] + 14, tree_box[3]), fill=(92, 64, 40))
                draw.ellipse(tree_box, fill=(72, 138, 76))
                draw.ellipse(livestock_box, fill=(225, 235, 220))
                draw.ellipse((livestock_box[0] + 3, livestock_box[1] - 4, livestock_box[0] + 12, livestock_box[1] + 5), fill=(225, 235, 220))

                boxes.extend(
                    [
                        {"label": "field", "xyxy": field_box, "attributes": ["open"]},
                        {"label": "road", "xyxy": road_box, "attributes": ["country"]},
                        {"label": "house", "xyxy": house_box, "attributes": ["small"]},
                        {"label": "tree", "xyxy": tree_box, "attributes": ["tall"]},
                        {"label": "livestock", "xyxy": livestock_box, "attributes": ["animal"]},
                    ]
                )
                relations.extend(
                    [
                        {"subject": 2, "predicate": "near", "object": 3},
                        {"subject": 1, "predicate": "next_to", "object": 0},
                        {"subject": 2, "predicate": "near", "object": 1},
                        {"subject": 4, "predicate": "on", "object": 0},
                    ]
                )
                qa_pairs.extend(
                    [
                        {
                            "question": "is there a vehicle?",
                            "answer": "no",
                            "answer_type": "boolean",
                        },
                        {
                            "question": "what is next to the field?",
                            "answer": "road",
                            "answer_type": "relation",
                        },
                        {
                            "question": "what is on the field?",
                            "answer": "livestock",
                            "answer_type": "relation",
                        },
                        {
                            "question": "is the house near the tree?",
                            "answer": "yes",
                            "answer_type": "grounded_reasoning",
                        },
                        {
                            "question": "is the house near the road?",
                            "answer": "yes",
                            "answer_type": "grounded_reasoning",
                        },
                        {
                            "question": "is the vehicle on the road?",
                            "answer": "no",
                            "answer_type": "boolean",
                        },
                    ]
                )
            elif "urban" in scene_key:
                sky_box = (0, 0, image_size, int(image_size * 0.28))
                left_building = (8, int(image_size * 0.16), 30, int(image_size * 0.64))
                right_building = (image_size - 34, int(image_size * 0.12), image_size - 8, int(image_size * 0.64))
                road_box = (0, int(image_size * 0.58), image_size, image_size)
                sidewalk_box = (0, int(image_size * 0.5), image_size, int(image_size * 0.58))
                vehicle_box = (int(image_size * 0.42), int(image_size * 0.68), int(image_size * 0.7), int(image_size * 0.84))

                draw.rectangle(sky_box, fill=(200, 220, 245))
                draw.rectangle(road_box, fill=(78, 82, 92))
                draw.rectangle(sidewalk_box, fill=(176, 176, 176))
                draw.rectangle(left_building, fill=secondary)
                draw.rectangle(right_building, fill=(secondary[0] - 12, secondary[1] - 12, secondary[2] - 12))
                draw.rectangle(vehicle_box, fill=(214, 78, 66))
                draw.rectangle((vehicle_box[0] + 6, vehicle_box[1] - 6, vehicle_box[2] - 6, vehicle_box[1] + 2), fill=(210, 232, 242))
                draw.line((0, int(image_size * 0.79), image_size, int(image_size * 0.79)), fill=(242, 242, 196), width=2)

                boxes.extend(
                    [
                        {"label": "building_left", "xyxy": left_building, "attributes": ["tall"]},
                        {"label": "building_right", "xyxy": right_building, "attributes": ["dense"]},
                        {"label": "road", "xyxy": road_box, "attributes": ["city"]},
                        {"label": "sidewalk", "xyxy": sidewalk_box, "attributes": ["paved"]},
                        {"label": "vehicle", "xyxy": vehicle_box, "attributes": ["moving"]},
                    ]
                )
                relations.extend(
                    [
                        {"subject": 4, "predicate": "on", "object": 2},
                        {"subject": 3, "predicate": "next_to", "object": 2},
                        {"subject": 0, "predicate": "near", "object": 2},
                        {"subject": 1, "predicate": "next_to", "object": 3},
                    ]
                )
                qa_pairs.extend(
                    [
                        {
                            "question": "is there a vehicle?",
                            "answer": "yes",
                            "answer_type": "boolean",
                        },
                        {
                            "question": "what is the vehicle on?",
                            "answer": "road",
                            "answer_type": "relation",
                        },
                        {
                            "question": "what is next to the road?",
                            "answer": "sidewalk",
                            "answer_type": "grounded_reasoning",
                        },
                        {
                            "question": "is the vehicle on the field?",
                            "answer": "no",
                            "answer_type": "boolean",
                        },
                        {
                            "question": "is the sidewalk next to the road?",
                            "answer": "yes",
                            "answer_type": "grounded_reasoning",
                        },
                        {
                            "question": "what is next to the sidewalk?",
                            "answer": "building_right",
                            "answer_type": "relation",
                        },
                    ]
                )
            elif "spill" in scene_name:
                spill_box = (12, 28, image_size - 12, image_size - 8)
                cup_box = (34, 8, 62, 38)
                draw.ellipse(spill_box, fill=primary)
                draw.rectangle(cup_box, fill=secondary)
                boxes.extend(
                    [
                        {"label": "liquid", "xyxy": spill_box},
                        {"label": "container", "xyxy": cup_box, "attributes": ["tipped"]},
                    ]
                )
                relations.append({"subject": 1, "predicate": "spills", "object": 0})
                qa_pairs.append(
                    {
                        "question": "what caused the spill?",
                        "answer": "container tipping",
                        "answer_type": "causal",
                    }
                )
            elif "order" in scene_name:
                for column in range(3):
                    x0 = 16 + column * 22
                    box = (x0, 18, x0 + 14, image_size - 18)
                    draw.rectangle(box, fill=primary)
                    boxes.append({"label": f"aligned_object_{column}", "xyxy": box, "attributes": ["vertical"]})
                draw.line((8, image_size - 12, image_size - 8, image_size - 12), fill=secondary, width=3)
                relations.extend(
                    [
                        {"subject": 0, "predicate": "aligned_with", "object": 1},
                        {"subject": 1, "predicate": "aligned_with", "object": 2},
                    ]
                )
                qa_pairs.append(
                    {
                        "question": "are the objects orderly?",
                        "answer": "yes",
                        "answer_type": "boolean",
                    }
                )
            else:
                for row in range(3):
                    y0 = image_size - 24 - row * 16
                    box = (18 + row * 8, y0, image_size - 18 - row * 8, y0 + 10)
                    draw.rectangle(box, fill=primary)
                    boxes.append({"label": f"stack_{row}", "xyxy": box})
                support_box = (22, image_size - 18, 40, image_size - 6)
                draw.rectangle(support_box, fill=secondary)
                boxes.append({"label": "support", "xyxy": support_box})
                relations.extend(
                    [
                        {"subject": 0, "predicate": "on_top_of", "object": 1},
                        {"subject": 1, "predicate": "on_top_of", "object": 2},
                        {"subject": 2, "predicate": "supported_by", "object": 3},
                    ]
                )
                qa_pairs.append(
                    {
                        "question": "how many stacked objects are there?",
                        "answer": "3",
                        "answer_type": "count",
                    }
                )
            image_path = scene_dir / f"{scene_name}_{index:02d}.png"
            image.save(image_path)
            if include_annotations:
                annotation_records.append(
                    {
                        "split": split,
                        "image_rel_path": str(image_path.relative_to(root_path).as_posix()),
                        "scene_name": scene_name,
                        "boxes": boxes,
                        "relations": relations,
                        "qa_pairs": qa_pairs,
                        "metadata": {"synthetic": True},
                    }
                )
    annotation_manifest_path = split_dir.parent / f"{split}_annotations.jsonl"
    if include_annotations:
        with annotation_manifest_path.open("w", encoding="utf-8") as handle:
            for row in annotation_records:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    else:
        annotation_manifest_path = Path("")
    return discover_visual_samples(
        split_dir.parent,
        split=split,
        annotation_manifest=annotation_manifest_path if include_annotations else None,
    )


def _find_flat_scene_images(
    source_root: Path,
    *,
    scene_names: Sequence[str],
    allowed_extensions: Sequence[str],
) -> Dict[str, List[Path]]:
    scene_to_images: Dict[str, List[Path]] = {scene: [] for scene in scene_names}
    for scene_name in scene_names:
        scene_dir = source_root / scene_name
        if not scene_dir.exists():
            continue
        for image_path in sorted(scene_dir.rglob("*")):
            if _is_image_file(image_path, allowed_extensions):
                scene_to_images[scene_name].append(image_path.resolve())
    return scene_to_images


def prepare_real_scene_dataset(
    source_root: str | Path,
    destination_root: str | Path,
    *,
    scene_names: Sequence[str],
    source_annotation_manifest: str | Path | None = None,
    train_ratio: float = 0.8,
    seed: int = 17,
    copy_files: bool = True,
    include_classification_qa: bool = True,
    allowed_extensions: Sequence[str] | None = None,
) -> Dict[str, object]:
    source_root_path = Path(source_root).resolve()
    destination_root_path = Path(destination_root).resolve()
    destination_root_path.mkdir(parents=True, exist_ok=True)
    extensions = tuple(allowed_extensions or SBVisualDatasetConfig().allowed_extensions)
    annotation_records = (
        load_visual_annotation_manifest(source_annotation_manifest, dataset_root=source_root_path)
        if source_annotation_manifest is not None
        else {}
    )
    scene_to_images = _find_flat_scene_images(
        source_root_path,
        scene_names=scene_names,
        allowed_extensions=extensions,
    )
    rng = random.Random(seed)

    manifest_paths: Dict[str, str] = {}
    split_counts: Dict[str, Dict[str, int]] = {"train": {}, "val": {}}
    for split_name in ("train", "val"):
        split_dir = destination_root_path / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        annotation_rows: List[Dict[str, object]] = []
        for scene_name in scene_names:
            images = list(scene_to_images.get(scene_name, ()))
            rng.shuffle(images)
            train_cutoff = int(round(len(images) * train_ratio))
            if len(images) > 1:
                train_cutoff = min(max(train_cutoff, 1), len(images) - 1)
            chosen = images[:train_cutoff] if split_name == "train" else images[train_cutoff:]
            split_counts[split_name][scene_name] = len(chosen)
            scene_dir = split_dir / scene_name
            scene_dir.mkdir(parents=True, exist_ok=True)
            for index, image_path in enumerate(chosen):
                target_path = scene_dir / f"{scene_name}_{index:04d}{image_path.suffix.lower()}"
                if copy_files:
                    shutil.copy2(image_path, target_path)
                else:
                    if target_path.exists():
                        target_path.unlink()
                    os_link_supported = hasattr(target_path, "hardlink_to")
                    if os_link_supported:
                        target_path.hardlink_to(image_path)
                    else:
                        shutil.copy2(image_path, target_path)
                source_rel_key = _normalize_image_key(image_path.relative_to(source_root_path))
                source_parent_rel_key = (
                    _normalize_image_key(image_path.relative_to(source_root_path.parent))
                    if source_root_path.parent in image_path.parents
                    else ""
                )
                source_abs_key = _normalize_image_key(image_path)
                source_name_key = _normalize_image_key(image_path.name)
                annotation = (
                    annotation_records.get(source_rel_key)
                    or annotation_records.get(source_parent_rel_key)
                    or annotation_records.get(source_abs_key)
                    or annotation_records.get(source_name_key)
                )
                metadata = {
                    "synthetic": False,
                    "source_path": str(image_path),
                    "source_scene_name": scene_name,
                }
                if annotation is not None:
                    metadata.update(annotation.metadata)
                qa_pairs = []
                boxes = []
                relations = []
                if annotation is not None:
                    boxes = [
                        {
                            "label": box.label,
                            "xyxy": list(box.xyxy),
                            "attributes": list(box.attributes),
                        }
                        for box in annotation.boxes
                    ]
                    relations = [
                        {
                            "subject": relation.subject,
                            "predicate": relation.predicate,
                            "object": relation.object,
                            "confidence": relation.confidence,
                        }
                        for relation in annotation.relations
                    ]
                    qa_pairs.extend(
                        {
                            "question": qa.question,
                            "answer": qa.answer,
                            "answer_type": qa.answer_type,
                        }
                        for qa in annotation.qa_pairs
                    )
                if include_classification_qa:
                    has_scene_question = any(
                        str(qa.get("question", "")).strip().lower() == "what scene is shown?"
                        for qa in qa_pairs
                    )
                    if not has_scene_question:
                        qa_pairs.append(
                            {
                                "question": "what scene is shown?",
                                "answer": scene_name,
                                "answer_type": "classification",
                            }
                        )
                annotation_rows.append(
                    {
                        "split": split_name,
                        "image_rel_path": str(target_path.relative_to(destination_root_path).as_posix()),
                        "scene_name": scene_name,
                        "boxes": boxes,
                        "relations": relations,
                        "qa_pairs": qa_pairs,
                        "metadata": metadata,
                    }
                )
        manifest_path = destination_root_path / f"{split_name}_annotations.jsonl"
        with manifest_path.open("w", encoding="utf-8") as handle:
            for row in annotation_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        manifest_paths[split_name] = str(manifest_path)

    return {
        "source_root": str(source_root_path),
        "destination_root": str(destination_root_path),
        "scene_names": list(scene_names),
        "source_annotation_manifest": str(Path(source_annotation_manifest).resolve()) if source_annotation_manifest is not None else "",
        "train_ratio": float(train_ratio),
        "seed": int(seed),
        "copy_files": bool(copy_files),
        "include_classification_qa": bool(include_classification_qa),
        "split_counts": split_counts,
        "annotation_manifests": manifest_paths,
    }
