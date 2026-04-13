from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from functools import partial
from typing import Callable, List

import torch
import torch.nn.functional as F

from sb.core_lm_data import (
    ToySequenceBatch,
    sample_needle_in_haystack_batch,
    sample_passkey_batch,
)
from sb.core_lm_torch import SBCoreMiniLM, SBCoreMiniTorchConfig, runtime_device_report
from sb.eval_long_context import LongContextMeasurement
from sb.transformer_baseline import TinyTransformerConfig, TinyTransformerLM


BatchFactory = Callable[..., ToySequenceBatch]


@dataclass(frozen=True)
class TaskSpec:
    name: str
    train_batch: BatchFactory
    eval_batch: BatchFactory
    long_eval_batch: BatchFactory
    train_sequence_length: int
    long_sequence_length: int
    seed: int


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
) -> float:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    model.train()
    final_loss = 0.0
    for _ in range(train_steps):
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
                eval_ms_per_batch=(in_ms + long_ms) / 2.0,
            )
        )
    return results


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(11)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(11)

    vocab_size = 40
    max_seq_len = 160
    train_steps = 220
    train_batch_size = 48
    eval_batch_size = 64
    eval_repeats = 8
    lr = 6e-3

    report = {
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
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
