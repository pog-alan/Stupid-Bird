from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from functools import partial
from typing import Callable, List

import torch
import torch.nn.functional as F

from sb.core_lm_data import ToySequenceBatch, sample_passkey_batch
from sb.core_lm_torch import SBCoreMiniLM, SBCoreMiniTorchConfig
from sb.train_lm import CurriculumStage, SBCoreTrainingPlan


BatchFactory = Callable[..., ToySequenceBatch]


@dataclass(frozen=True)
class CurriculumEvaluation:
    strategy: str
    total_steps: int
    train_loss: float
    in_distribution_token_acc: float
    in_distribution_exact_match: float
    long_context_token_acc: float
    long_context_exact_match: float
    extra_long_token_acc: float
    extra_long_exact_match: float


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
    token_acc = float((((predictions == targets) & focus_mask).sum().float() / focus_mask.sum().clamp_min(1).float()).cpu())
    exact_match = float(((((predictions == targets) | (~focus_mask)).all(dim=-1)).float().mean()).cpu())
    return token_acc, exact_match


def make_batch(factory: BatchFactory, batch_size: int, device: str | torch.device) -> ToySequenceBatch:
    batch = factory(batch_size=batch_size, device=device, return_metadata=True)
    if not isinstance(batch, ToySequenceBatch):
        raise TypeError("expected ToySequenceBatch from passkey generator")
    return batch


def evaluate(
    model: torch.nn.Module,
    factory: BatchFactory,
    *,
    device: str | torch.device,
    batch_size: int,
    repeats: int,
) -> tuple[float, float]:
    token_scores: List[float] = []
    exact_scores: List[float] = []
    model.eval()
    with torch.no_grad():
        for _ in range(repeats):
            batch = make_batch(factory, batch_size=batch_size, device=device)
            inputs = batch.tokens[:, :-1]
            targets = batch.tokens[:, 1:]
            forward = model(inputs)
            token_acc, exact_match = masked_metrics(forward["logits"], targets, batch.focus_mask)
            token_scores.append(token_acc)
            exact_scores.append(exact_match)
    return sum(token_scores) / len(token_scores), sum(exact_scores) / len(exact_scores)


def stage_factory(stage: CurriculumStage, vocab_size: int) -> BatchFactory:
    return partial(
        sample_passkey_batch,
        prefix_len=stage.prefix_len,
        filler_len=stage.filler_len,
        key_length=stage.key_length,
        vocab_size=vocab_size,
    )


def build_model(vocab_size: int, max_seq_len: int, device: str | torch.device) -> SBCoreMiniLM:
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


def train_schedule(
    strategy: str,
    schedule: List[CurriculumStage],
    *,
    seed: int,
    device: str | torch.device,
    vocab_size: int,
    max_seq_len: int,
    batch_size: int,
    lr: float,
) -> CurriculumEvaluation:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = build_model(vocab_size=vocab_size, max_seq_len=max_seq_len, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    final_loss = 0.0

    for stage in schedule:
        batch_factory = stage_factory(stage, vocab_size=vocab_size)
        for _ in range(stage.steps):
            batch = make_batch(batch_factory, batch_size=batch_size, device=device)
            inputs = batch.tokens[:, :-1]
            targets = batch.tokens[:, 1:]

            optimizer.zero_grad(set_to_none=True)
            forward = model(inputs)
            loss = masked_next_token_loss(forward["logits"], targets, batch.focus_mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            final_loss = float(loss.detach().cpu())

    in_factory = partial(sample_passkey_batch, prefix_len=6, filler_len=8, key_length=2, vocab_size=vocab_size)
    long_factory = partial(sample_passkey_batch, prefix_len=20, filler_len=32, key_length=2, vocab_size=vocab_size)
    extra_long_factory = partial(sample_passkey_batch, prefix_len=28, filler_len=48, key_length=2, vocab_size=vocab_size)

    in_token, in_exact = evaluate(model, in_factory, device=device, batch_size=64, repeats=8)
    long_token, long_exact = evaluate(model, long_factory, device=device, batch_size=64, repeats=8)
    extra_token, extra_exact = evaluate(model, extra_long_factory, device=device, batch_size=64, repeats=8)

    return CurriculumEvaluation(
        strategy=strategy,
        total_steps=sum(stage.steps for stage in schedule),
        train_loss=final_loss,
        in_distribution_token_acc=in_token,
        in_distribution_exact_match=in_exact,
        long_context_token_acc=long_token,
        long_context_exact_match=long_exact,
        extra_long_token_acc=extra_token,
        extra_long_exact_match=extra_exact,
    )


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    plan = SBCoreTrainingPlan()
    curriculum = plan.passkey_curriculum()
    fixed_template = plan.fixed_passkey_baseline()
    fixed = [
        CurriculumStage(
            stage_id=f"{fixed_template.stage_id}-{index}",
            prefix_len=fixed_template.prefix_len,
            filler_len=fixed_template.filler_len,
            key_length=fixed_template.key_length,
            steps=120,
        )
        for index in range(4)
    ]

    report = {
        "device": device,
        "vocab_size": 40,
        "results": [
            asdict(
                train_schedule(
                    "fixed_length",
                    fixed,
                    seed=777,
                    device=device,
                    vocab_size=40,
                    max_seq_len=160,
                    batch_size=48,
                    lr=6e-3,
                )
            ),
            asdict(
                train_schedule(
                    "curriculum",
                    curriculum,
                    seed=777,
                    device=device,
                    vocab_size=40,
                    max_seq_len=160,
                    batch_size=48,
                    lr=6e-3,
                )
            ),
        ],
        "fixed_schedule": [asdict(stage) for stage in fixed],
        "curriculum_schedule": [asdict(stage) for stage in curriculum],
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
