from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Callable, List

import torch
import torch.nn.functional as F

from sb.core_lm_data import (
    ToySequenceBatch,
    sample_needle_in_haystack_batch,
    sample_passkey_batch,
)
from sb.core_lm_torch import (
    SBCoreMiniLM,
    SBCoreMiniTorchConfig,
    SBRuntimeGates,
    runtime_device_report,
    staged_runtime_gates,
)
from sb.eval_long_context import LongContextMeasurement
from sb.transformer_baseline import TinyTransformerConfig, TinyTransformerLM


BatchFactory = Callable[..., ToySequenceBatch]
DEFAULT_OUTPUT_PATH = Path("data/processed/experiments/long_context_compare.json")


@dataclass(frozen=True)
class TaskSpec:
    name: str
    train_batch: BatchFactory
    eval_batch: BatchFactory
    long_eval_batch: BatchFactory
    train_sequence_length: int
    long_sequence_length: int
    seed: int


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare SB-Core against a tiny Transformer on synthetic long-context tasks.")
    parser.add_argument("--vocab-size", type=int, default=40)
    parser.add_argument("--max-seq-len", type=int, default=160)
    parser.add_argument("--train-steps", type=int, default=220)
    parser.add_argument("--train-batch-size", type=int, default=48)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--eval-repeats", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=6e-3)
    parser.add_argument("--experiment-tag", default="")
    parser.add_argument("--output-path", default="")
    return parser


def resolve_output_path(path_str: str, experiment_tag: str) -> Path:
    if path_str:
        path = Path(path_str)
    elif experiment_tag:
        path = DEFAULT_OUTPUT_PATH.with_name(f"{DEFAULT_OUTPUT_PATH.stem}_{experiment_tag}{DEFAULT_OUTPUT_PATH.suffix}")
    else:
        path = DEFAULT_OUTPUT_PATH
    if not path.is_absolute():
        path = Path.cwd() / path
    return path


def masked_next_token_loss(logits: torch.Tensor, targets: torch.Tensor, focus_mask: torch.Tensor) -> torch.Tensor:
    vocab_size = logits.shape[-1]
    losses = F.cross_entropy(
        logits.reshape(-1, vocab_size),
        targets.reshape(-1),
        reduction="none",
    ).reshape_as(targets)
    weights = focus_mask.float()
    return (losses * weights).sum() / weights.sum().clamp_min(1.0)


def masked_metrics(logits: torch.Tensor, targets: torch.Tensor, focus_mask: torch.Tensor) -> tuple[float, float]:
    predictions = logits.argmax(dim=-1)
    masked_correct = ((predictions == targets) & focus_mask).sum()
    masked_total = focus_mask.sum().clamp_min(1)
    token_acc = float((masked_correct.float() / masked_total.float()).cpu())
    exact_match = float(
        (((predictions == targets) | (~focus_mask)).all(dim=-1).float().mean()).cpu()
    )
    return token_acc, exact_match


def count_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def make_batch(factory: BatchFactory, batch_size: int, device: str | torch.device) -> ToySequenceBatch:
    batch = factory(batch_size=batch_size, device=device, return_metadata=True)
    if not isinstance(batch, ToySequenceBatch):
        raise TypeError("expected ToySequenceBatch from retrieval task generator")
    return batch


def sync_device(device: str | torch.device) -> None:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def train_model(
    model: torch.nn.Module,
    train_batch: BatchFactory,
    *,
    device: str | torch.device,
    train_steps: int,
    batch_size: int,
    lr: float,
    use_staged_gates: bool = True,
    step_offset: int = 0,
    stage_total_steps: int | None = None,
) -> float:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    model.train()
    final_loss = 0.0
    sb_warmup_steps = max(20, train_steps // 6) if isinstance(model, SBCoreMiniLM) else 0
    for step_index in range(train_steps):
        if isinstance(model, SBCoreMiniLM):
            if use_staged_gates:
                _, stage_gates = staged_runtime_gates(
                    step_index=step_offset + step_index,
                    total_steps=stage_total_steps or train_steps,
                )
            else:
                stage_gates = SBRuntimeGates()
            model.set_runtime_gates(stage_gates)
        if sb_warmup_steps > 0:
            lr_scale = min(1.0, float(step_index + 1) / float(sb_warmup_steps))
            for group in optimizer.param_groups:
                group["lr"] = lr * lr_scale
        batch = make_batch(train_batch, batch_size=batch_size, device=device)
        inputs = batch.tokens[:, :-1]
        targets = batch.tokens[:, 1:]

        optimizer.zero_grad(set_to_none=True)
        forward = model(inputs)
        logits = forward["logits"]
        loss = masked_next_token_loss(logits, targets, batch.focus_mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        final_loss = float(loss.detach().cpu())
    return final_loss


def evaluate_model(
    model: torch.nn.Module,
    batch_factory: BatchFactory,
    *,
    device: str | torch.device,
    batch_size: int,
    repeats: int,
) -> tuple[float, float, float]:
    token_scores: List[float] = []
    exact_scores: List[float] = []
    elapsed = 0.0

    model.eval()
    with torch.no_grad():
        if isinstance(model, SBCoreMiniLM):
            model.set_runtime_gates(SBRuntimeGates())
        for _ in range(repeats):
            batch = make_batch(batch_factory, batch_size=batch_size, device=device)
            inputs = batch.tokens[:, :-1]
            targets = batch.tokens[:, 1:]
            sync_device(device)
            started = time.perf_counter()
            forward = model(inputs)
            sync_device(device)
            elapsed += time.perf_counter() - started
            token_acc, exact_match = masked_metrics(forward["logits"], targets, batch.focus_mask)
            token_scores.append(token_acc)
            exact_scores.append(exact_match)

    mean_token = sum(token_scores) / max(len(token_scores), 1)
    mean_exact = sum(exact_scores) / max(len(exact_scores), 1)
    ms_per_batch = elapsed * 1000.0 / max(repeats, 1)
    return mean_token, mean_exact, ms_per_batch


def evaluate_segmented_model(
    model: torch.nn.Module,
    batch_factory: BatchFactory,
    *,
    device: str | torch.device,
    batch_size: int,
    repeats: int,
    segment_length: int,
    carry_memory: bool,
) -> tuple[float, float, float]:
    token_scores: List[float] = []
    exact_scores: List[float] = []
    elapsed = 0.0

    model.eval()
    with torch.no_grad():
        if isinstance(model, SBCoreMiniLM):
            model.set_runtime_gates(SBRuntimeGates())
        for _ in range(repeats):
            batch = make_batch(batch_factory, batch_size=batch_size, device=device)
            inputs = batch.tokens[:, :-1]
            targets = batch.tokens[:, 1:]
            focus_mask = batch.focus_mask
            predictions = torch.empty_like(targets)
            state = None

            sync_device(device)
            started = time.perf_counter()
            for start in range(0, inputs.shape[1], segment_length):
                end = min(inputs.shape[1], start + segment_length)
                chunk_inputs = inputs[:, start:end]
                if isinstance(model, SBCoreMiniLM):
                    forward = model(
                        chunk_inputs,
                        memory_state=state if carry_memory else None,
                        return_state=carry_memory,
                    )
                    if carry_memory:
                        state = forward.get("state")
                else:
                    forward = model(chunk_inputs)
                predictions[:, start:end] = forward["logits"].argmax(dim=-1)
            sync_device(device)
            elapsed += time.perf_counter() - started

            masked_correct = ((predictions == targets) & focus_mask).sum()
            masked_total = focus_mask.sum().clamp_min(1)
            token_scores.append(float((masked_correct.float() / masked_total.float()).cpu()))
            exact_scores.append(
                float(((((predictions == targets) | (~focus_mask)).all(dim=-1)).float().mean()).cpu())
            )

    mean_token = sum(token_scores) / max(len(token_scores), 1)
    mean_exact = sum(exact_scores) / max(len(exact_scores), 1)
    ms_per_batch = elapsed * 1000.0 / max(repeats, 1)
    return mean_token, mean_exact, ms_per_batch


def build_tasks(vocab_size: int) -> List[TaskSpec]:
    return [
        TaskSpec(
            name="passkey_retrieval",
            train_batch=partial(
                sample_passkey_batch,
                prefix_len=6,
                filler_len=8,
                key_length=2,
                vocab_size=vocab_size,
            ),
            eval_batch=partial(
                sample_passkey_batch,
                prefix_len=6,
                filler_len=8,
                key_length=2,
                vocab_size=vocab_size,
            ),
            long_eval_batch=partial(
                sample_passkey_batch,
                prefix_len=20,
                filler_len=32,
                key_length=2,
                vocab_size=vocab_size,
            ),
            train_sequence_length=22,
            long_sequence_length=60,
            seed=201,
        ),
        TaskSpec(
            name="needle_in_haystack",
            train_batch=partial(
                sample_needle_in_haystack_batch,
                prefix_len=8,
                suffix_len=8,
                key_length=2,
                value_length=2,
                vocab_size=vocab_size,
            ),
            eval_batch=partial(
                sample_needle_in_haystack_batch,
                prefix_len=8,
                suffix_len=8,
                key_length=2,
                value_length=2,
                vocab_size=vocab_size,
            ),
            long_eval_batch=partial(
                sample_needle_in_haystack_batch,
                prefix_len=28,
                suffix_len=36,
                key_length=2,
                value_length=2,
                vocab_size=vocab_size,
            ),
            train_sequence_length=28,
            long_sequence_length=76,
            seed=401,
        ),
    ]


def build_sb_model(vocab_size: int, max_seq_len: int, device: str | torch.device) -> SBCoreMiniLM:
    config = SBCoreMiniTorchConfig(
        vocab_size=vocab_size,
        d_model=96,
        state_dim=96,
        num_layers=2,
        signal_schema_slots=7,
        semantic_memory_slots=8,
        working_memory_slots=16,
        router_top_k=2,
        dropout=0.0,
        max_seq_len=max_seq_len,
    )
    return SBCoreMiniLM(config).to(device)


def build_transformer_model(vocab_size: int, max_seq_len: int, device: str | torch.device) -> TinyTransformerLM:
    config = TinyTransformerConfig(
        vocab_size=vocab_size,
        d_model=96,
        num_layers=2,
        num_heads=4,
        ff_multiplier=4,
        dropout=0.0,
        max_seq_len=max_seq_len,
    )
    return TinyTransformerLM(config).to(device)


def compare_task(
    task: TaskSpec,
    *,
    device: str | torch.device,
    vocab_size: int,
    max_seq_len: int,
    train_steps: int,
    train_batch_size: int,
    eval_batch_size: int,
    eval_repeats: int,
    lr: float,
) -> List[LongContextMeasurement]:
    results: List[LongContextMeasurement] = []
    model_builders = [
        ("sb_core_mini", build_sb_model),
        ("tiny_transformer", build_transformer_model),
    ]

    for seed_offset, (model_name, builder) in enumerate(model_builders):
        torch.manual_seed(task.seed + seed_offset)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(task.seed + seed_offset)
        model = builder(vocab_size=vocab_size, max_seq_len=max_seq_len, device=device)

        train_loss = train_model(
            model,
            task.train_batch,
            device=device,
            train_steps=train_steps,
            batch_size=train_batch_size,
            lr=lr,
        )
        in_token, in_exact, in_ms = evaluate_model(
            model,
            task.eval_batch,
            device=device,
            batch_size=eval_batch_size,
            repeats=eval_repeats,
        )
        long_token, long_exact, long_ms = evaluate_model(
            model,
            task.long_eval_batch,
            device=device,
            batch_size=eval_batch_size,
            repeats=eval_repeats,
        )
        segmented_length = max(task.train_sequence_length, 8)
        segmented_off_token, segmented_off_exact, segmented_off_ms = evaluate_segmented_model(
            model,
            task.long_eval_batch,
            device=device,
            batch_size=eval_batch_size,
            repeats=eval_repeats,
            segment_length=segmented_length,
            carry_memory=False,
        )
        segmented_on_token, segmented_on_exact, segmented_on_ms = evaluate_segmented_model(
            model,
            task.long_eval_batch,
            device=device,
            batch_size=eval_batch_size,
            repeats=eval_repeats,
            segment_length=segmented_length,
            carry_memory=True,
        )

        results.append(
            LongContextMeasurement(
                model_name=model_name,
                task_name=task.name,
                parameter_count=count_parameters(model),
                train_sequence_length=task.train_sequence_length,
                long_sequence_length=task.long_sequence_length,
                train_loss=train_loss,
                in_distribution_token_acc=in_token,
                in_distribution_exact_match=in_exact,
                long_context_token_acc=long_token,
                long_context_exact_match=long_exact,
                eval_ms_per_batch=(in_ms + long_ms + segmented_off_ms + segmented_on_ms) / 4.0,
                segmented_carry_off_token_acc=segmented_off_token,
                segmented_carry_off_exact_match=segmented_off_exact,
                segmented_token_acc=segmented_on_token,
                segmented_exact_match=segmented_on_exact,
                carry_gain_token_acc=segmented_on_token - segmented_off_token,
                carry_gain_exact_match=segmented_on_exact - segmented_off_exact,
            )
        )
    return results


def main() -> None:
    args = build_parser().parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(11)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(11)

    output_path = resolve_output_path(args.output_path, args.experiment_tag)
    vocab_size = args.vocab_size
    max_seq_len = args.max_seq_len
    train_steps = args.train_steps
    train_batch_size = args.train_batch_size
    eval_batch_size = args.eval_batch_size
    eval_repeats = args.eval_repeats
    lr = args.learning_rate

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment_tag": args.experiment_tag,
        "device": device,
        "runtime": runtime_device_report(device),
        "vocab_size": vocab_size,
        "max_seq_len": max_seq_len,
        "train_steps": train_steps,
        "train_batch_size": train_batch_size,
        "eval_batch_size": eval_batch_size,
        "eval_repeats": eval_repeats,
        "results": [],
    }

    for task in build_tasks(vocab_size=vocab_size):
        print(json.dumps({"task": task.name, "status": "running"}, ensure_ascii=False))
        task_results = compare_task(
            task,
            device=device,
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            train_steps=train_steps,
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            eval_repeats=eval_repeats,
            lr=lr,
        )
        report["results"].extend(asdict(item) for item in task_results)
        print(
            json.dumps(
                {
                    "task": task.name,
                    "status": "done",
                    "results": [asdict(item) for item in task_results],
                },
                ensure_ascii=False,
            )
        )

    print("=== Long Context Comparison ===")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"report_path={output_path}")


if __name__ == "__main__":
    main()
