from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import torch

from examples.v02_long_context_compare import (
    build_sb_model,
    build_tasks,
    evaluate_model,
    make_batch,
    masked_next_token_loss,
)
from sb.core_lm_torch import SBRuntimeGates, SBCoreMiniLM, runtime_device_report, staged_runtime_gates


OUTPUT_PATH = Path("data/processed/experiments/stage_ablation_long.json")


def evaluate_checkpoint(model, task, device: str | torch.device, batch_size: int, repeats: int) -> dict[str, float]:
    in_token, in_exact, in_ms = evaluate_model(
        model,
        task.eval_batch,
        device=device,
        batch_size=batch_size,
        repeats=repeats,
    )
    long_token, long_exact, long_ms = evaluate_model(
        model,
        task.long_eval_batch,
        device=device,
        batch_size=batch_size,
        repeats=repeats,
    )
    return {
        "in_distribution_token_acc": in_token,
        "in_distribution_exact_match": in_exact,
        "in_distribution_ms_per_batch": in_ms,
        "long_context_token_acc": long_token,
        "long_context_exact_match": long_exact,
        "long_context_ms_per_batch": long_ms,
    }


def train_to_checkpoints(
    model: torch.nn.Module,
    task,
    *,
    device: str | torch.device,
    checkpoint_steps: list[int],
    train_batch_size: int,
    eval_batch_size: int,
    eval_repeats: int,
    learning_rate: float,
    use_staged_gates: bool,
) -> list[dict[str, float]]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    total_steps = checkpoint_steps[-1]
    sb_warmup_steps = max(20, total_steps // 6) if isinstance(model, SBCoreMiniLM) else 0
    checkpoints: list[dict[str, float]] = []
    completed_steps = 0
    train_loss = 0.0

    for step_index in range(total_steps):
        model.train()
        if isinstance(model, SBCoreMiniLM):
            if use_staged_gates:
                _, stage_gates = staged_runtime_gates(step_index=step_index, total_steps=total_steps)
            else:
                stage_gates = SBRuntimeGates()
            model.set_runtime_gates(stage_gates)
        if sb_warmup_steps > 0:
            lr_scale = min(1.0, float(step_index + 1) / float(sb_warmup_steps))
            for group in optimizer.param_groups:
                group["lr"] = learning_rate * lr_scale

        batch = make_batch(task.train_batch, batch_size=train_batch_size, device=device)
        inputs = batch.tokens[:, :-1]
        targets = batch.tokens[:, 1:]
        optimizer.zero_grad(set_to_none=True)
        forward = model(inputs)
        logits = forward["logits"]
        loss = masked_next_token_loss(logits, targets, batch.focus_mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        train_loss = float(loss.detach().cpu())

        if step_index + 1 in checkpoint_steps:
            completed_steps = step_index + 1
            metrics = evaluate_checkpoint(
                model,
                task,
                device=device,
                batch_size=eval_batch_size,
                repeats=eval_repeats,
            )
            checkpoints.append({"step": completed_steps, "train_loss": train_loss, **metrics})

    return checkpoints


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vocab_size = 40
    tasks = [task for task in build_tasks(vocab_size=vocab_size) if task.name in {"passkey_retrieval", "needle_in_haystack"}]
    train_steps = 60
    checkpoint_steps = [20, 40, 60]
    train_batch_size = 4
    eval_batch_size = 4
    eval_repeats = 1
    learning_rate = 6e-3

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    report: dict[str, object] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "runtime": runtime_device_report(device),
        "train_steps": train_steps,
        "checkpoint_steps": checkpoint_steps,
        "train_batch_size": train_batch_size,
        "eval_batch_size": eval_batch_size,
        "eval_repeats": eval_repeats,
        "tasks": [],
    }

    for task in tasks:
        task_report = {"task": task.name, "modes": []}
        for mode_name, staged in [("full_on", False), ("staged", True)]:
            torch.manual_seed(task.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(task.seed)
            model = build_sb_model(vocab_size=vocab_size, max_seq_len=160, device=device)
            checkpoints = train_to_checkpoints(
                model,
                task,
                device=device,
                checkpoint_steps=checkpoint_steps,
                train_batch_size=train_batch_size,
                eval_batch_size=eval_batch_size,
                eval_repeats=eval_repeats,
                learning_rate=learning_rate,
                use_staged_gates=staged,
            )
            task_report["modes"].append({"mode": mode_name, "checkpoints": checkpoints})
        report["tasks"].append(task_report)

    OUTPUT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
