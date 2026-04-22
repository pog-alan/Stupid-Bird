from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import torch
import torch.nn.functional as F

from .core_lm_torch import SBCoreMiniLM, SBRuntimeGates
from .text_corpus import _format_longbench, _read_jsonl


DEFAULT_TASKS = ("passage_retrieval_zh", "multifieldqa_zh", "dureader")
DEFAULT_CARRY_POLICY = "uniform"


@dataclass(frozen=True)
class LongBenchTaskCarryPolicy:
    name: str
    carry_memory: bool
    prompt_char_limit: int
    runtime_gates: SBRuntimeGates

    def as_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "carry_memory": bool(self.carry_memory),
            "prompt_char_limit": int(self.prompt_char_limit),
            "runtime_gates": self.runtime_gates.as_dict(),
        }


def resolve_task_carry_policy(
    task_name: str,
    *,
    requested_carry_memory: bool,
    requested_prompt_char_limit: int,
    carry_policy: str = DEFAULT_CARRY_POLICY,
) -> LongBenchTaskCarryPolicy:
    normalized = str(carry_policy or DEFAULT_CARRY_POLICY).strip().lower()
    prompt_limit = max(int(requested_prompt_char_limit), 0)
    default_gates = SBRuntimeGates()

    if normalized == "uniform":
        return LongBenchTaskCarryPolicy(
            name="uniform",
            carry_memory=bool(requested_carry_memory),
            prompt_char_limit=prompt_limit,
            runtime_gates=default_gates,
        )

    if normalized != "task_adaptive":
        raise ValueError(f"unsupported carry_policy: {carry_policy}")

    lowered_task = task_name.lower()
    if not requested_carry_memory:
        return LongBenchTaskCarryPolicy(
            name="task_adaptive_no_carry_override",
            carry_memory=False,
            prompt_char_limit=prompt_limit,
            runtime_gates=default_gates,
        )

    if "retrieval" in lowered_task:
        return LongBenchTaskCarryPolicy(
            name="retrieval_strong",
            carry_memory=True,
            prompt_char_limit=prompt_limit if prompt_limit > 0 else 256,
            runtime_gates=SBRuntimeGates(
                summary_read=1.0,
                summary_write=1.0,
                scene_read=1.0,
                scene_write=1.0,
                drill=1.0,
                forgetting=1.0,
            ),
        )

    if lowered_task in {"multifieldqa_zh", "dureader"} or "qa" in lowered_task:
        default_limit = 512 if lowered_task == "multifieldqa_zh" else 256
        return LongBenchTaskCarryPolicy(
            name="qa_safe_no_carry",
            carry_memory=False,
            prompt_char_limit=prompt_limit if prompt_limit > 0 else default_limit,
            runtime_gates=default_gates,
        )

    return LongBenchTaskCarryPolicy(
        name="generic_balanced",
        carry_memory=True,
        prompt_char_limit=prompt_limit if prompt_limit > 0 else 512,
        runtime_gates=SBRuntimeGates(
            summary_read=0.6,
            summary_write=1.0,
            scene_read=0.4,
            scene_write=1.0,
            drill=0.25,
            forgetting=0.85,
        ),
    )


def _schema_metrics_from_aux(aux: Dict[str, float] | None) -> Dict[str, float]:
    aux = aux or {}
    episodic = float(aux.get("episodic_replay_schema_alignment_mean", 0.0))
    summary = float(aux.get("summary_schema_alignment_mean", 0.0))
    scene = float(aux.get("scene_schema_alignment_mean", 0.0))
    return {
        "episodic_replay_schema_alignment_mean": episodic,
        "summary_schema_alignment_mean": summary,
        "scene_schema_alignment_mean": scene,
        "schema_chain_activated": 1.0 if summary > 0.0 and scene > 0.0 else 0.0,
    }


def score_answer_continuation(
    model: SBCoreMiniLM,
    tokenizer,
    *,
    prompt: str,
    answer: str,
    device: str | torch.device,
    carry_memory: bool = True,
    prompt_char_limit: int = 0,
) -> Dict[str, float]:
    if prompt_char_limit > 0 and len(prompt) > prompt_char_limit:
        prompt = prompt[-int(prompt_char_limit) :]
    prompt_ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
    answer_ids = tokenizer.encode(answer, add_bos=False, add_eos=True)
    if carry_memory:
        if not prompt_ids:
            prompt_ids = tokenizer.encode("", add_bos=True, add_eos=False)
        warmup_ids = prompt_ids[:-1]
        scoring_input_ids = prompt_ids[-1:] + answer_ids[:-1]
        scoring_target_ids = answer_ids
        max_chunk = model.config.max_seq_len
        total_loss = 0.0
        total_correct = 0.0
        total_tokens = 0
        state = None
        schema_metric_totals = _schema_metrics_from_aux(None)
        schema_metric_steps = 0

        with torch.no_grad():
            for start in range(0, len(warmup_ids), max_chunk):
                chunk = warmup_ids[start : start + max_chunk]
                if not chunk:
                    continue
                input_ids = torch.tensor([chunk], dtype=torch.long, device=device)
                forward = model(
                    input_ids,
                    return_aux=True,
                    memory_state=state,
                    return_state=True,
                )
                state = forward.get("state")
                aux_metrics = _schema_metrics_from_aux(forward.get("aux"))
                for key, value in aux_metrics.items():
                    schema_metric_totals[key] += value
                schema_metric_steps += 1

            for start in range(0, len(scoring_target_ids), max_chunk):
                chunk_inputs = scoring_input_ids[start : start + max_chunk]
                chunk_targets = scoring_target_ids[start : start + max_chunk]
                if not chunk_inputs or not chunk_targets:
                    continue
                input_ids = torch.tensor([chunk_inputs], dtype=torch.long, device=device)
                target_ids = torch.tensor([chunk_targets], dtype=torch.long, device=device)
                forward = model(
                    input_ids,
                    return_aux=True,
                    memory_state=state,
                    return_state=True,
                )
                state = forward.get("state")
                logits = forward["logits"]
                aux_metrics = _schema_metrics_from_aux(forward.get("aux"))
                for key, value in aux_metrics.items():
                    schema_metric_totals[key] += value
                schema_metric_steps += 1
                vocab_size = logits.shape[-1]
                losses = F.cross_entropy(
                    logits.reshape(-1, vocab_size),
                    target_ids.reshape(-1),
                    reduction="none",
                ).reshape_as(target_ids)
                predictions = logits.argmax(dim=-1)
                total_loss += float(losses.sum().cpu())
                total_correct += float((predictions == target_ids).float().sum().cpu())
                total_tokens += int(target_ids.numel())

        if total_tokens <= 0:
            return {
                "answer_loss": 0.0,
                "answer_token_acc": 0.0,
                "prompt_retained_ratio": 1.0,
                "answer_retained_ratio": 0.0,
                **_schema_metrics_from_aux(None),
            }
        schema_divisor = float(max(schema_metric_steps, 1))
        return {
            "answer_loss": total_loss / float(total_tokens),
            "answer_token_acc": total_correct / float(total_tokens),
            "prompt_retained_ratio": 1.0,
            "answer_retained_ratio": 1.0,
            **{key: value / schema_divisor for key, value in schema_metric_totals.items()},
        }

    full_ids = prompt_ids + answer_ids
    max_window = model.config.max_seq_len + 1
    start_index = max(0, len(full_ids) - max_window)
    kept_ids = full_ids[start_index:]

    input_ids = torch.tensor([kept_ids[:-1]], dtype=torch.long, device=device)
    target_ids = torch.tensor([kept_ids[1:]], dtype=torch.long, device=device)
    original_target_positions = torch.arange(start_index + 1, start_index + len(kept_ids), device=device)
    focus_mask = original_target_positions >= len(prompt_ids)
    kept_prompt_tokens = max(0, min(len(prompt_ids), len(full_ids)) - start_index)

    with torch.no_grad():
        forward = model(input_ids, return_aux=True)
        logits = forward["logits"]
        vocab_size = logits.shape[-1]
        losses = F.cross_entropy(
            logits.reshape(-1, vocab_size),
            target_ids.reshape(-1),
            reduction="none",
        ).reshape_as(target_ids)
        predictions = logits.argmax(dim=-1)
        masked_losses = losses[0][focus_mask]
        masked_targets = target_ids[0][focus_mask]
        masked_predictions = predictions[0][focus_mask]

    if masked_losses.numel() == 0:
        return {
            "answer_loss": 0.0,
            "answer_token_acc": 0.0,
            "prompt_retained_ratio": 0.0,
            "answer_retained_ratio": 0.0,
            **_schema_metrics_from_aux(forward.get("aux")),
        }

    return {
        "answer_loss": float(masked_losses.mean().cpu()),
        "answer_token_acc": float((masked_predictions == masked_targets).float().mean().cpu()),
        "prompt_retained_ratio": kept_prompt_tokens / max(len(prompt_ids), 1),
        "answer_retained_ratio": float(masked_losses.numel()) / max(len(answer_ids), 1),
        **_schema_metrics_from_aux(forward.get("aux")),
    }


def evaluate_longbench_local(
    model: SBCoreMiniLM,
    tokenizer,
    *,
    device: str | torch.device,
    tasks: Sequence[str] = DEFAULT_TASKS,
    max_samples: int = 24,
    data_root: str | Path = "data/raw/longbench",
    carry_memory: bool = True,
    prompt_char_limit: int = 0,
    carry_policy: str = DEFAULT_CARRY_POLICY,
) -> Dict[str, object]:
    data_root_path = Path(data_root)
    previous_gates = model.get_runtime_gates()
    model.set_runtime_gates(SBRuntimeGates())
    model.eval()

    task_reports: List[Dict[str, float | int | str]] = []
    all_answer_losses: List[float] = []
    all_answer_token_accs: List[float] = []

    try:
        for task_name in tasks:
            task_policy = resolve_task_carry_policy(
                task_name,
                requested_carry_memory=carry_memory,
                requested_prompt_char_limit=prompt_char_limit,
                carry_policy=carry_policy,
            )
            model.set_runtime_gates(task_policy.runtime_gates)
            task_path = data_root_path / f"{task_name}.jsonl"
            rows = _read_jsonl(task_path, limit=max_samples)
            metrics: List[Dict[str, float]] = []
            for row in rows:
                prompt = _format_longbench(task_name, {**row, "answers": []}) + "\nanswer:"
                answers = row.get("answers") or []
                answer = str(answers[0]) if answers else ""
                metrics.append(
                    score_answer_continuation(
                        model,
                        tokenizer,
                        prompt=prompt,
                        answer=answer,
                        device=device,
                        carry_memory=task_policy.carry_memory,
                        prompt_char_limit=task_policy.prompt_char_limit,
                    )
                )

            task_report = {
                "task": task_name,
                "samples": len(metrics),
                "carry_policy": task_policy.name,
                "requested_carry_memory": bool(carry_memory),
                "effective_carry_memory": bool(task_policy.carry_memory),
                "effective_prompt_char_limit": int(task_policy.prompt_char_limit),
                "runtime_gates": task_policy.runtime_gates.as_dict(),
                "mean_answer_loss": sum(item["answer_loss"] for item in metrics) / max(len(metrics), 1),
                "mean_answer_token_acc": sum(item["answer_token_acc"] for item in metrics) / max(len(metrics), 1),
                "mean_prompt_retained_ratio": sum(item["prompt_retained_ratio"] for item in metrics) / max(len(metrics), 1),
                "mean_answer_retained_ratio": sum(item["answer_retained_ratio"] for item in metrics) / max(len(metrics), 1),
                "mean_episodic_replay_schema_alignment": sum(
                    item["episodic_replay_schema_alignment_mean"] for item in metrics
                )
                / max(len(metrics), 1),
                "mean_summary_schema_alignment": sum(
                    item["summary_schema_alignment_mean"] for item in metrics
                )
                / max(len(metrics), 1),
                "mean_scene_schema_alignment": sum(item["scene_schema_alignment_mean"] for item in metrics)
                / max(len(metrics), 1),
                "schema_chain_activated_ratio": sum(item["schema_chain_activated"] for item in metrics)
                / max(len(metrics), 1),
            }
            task_reports.append(task_report)
            all_answer_losses.append(float(task_report["mean_answer_loss"]))
            all_answer_token_accs.append(float(task_report["mean_answer_token_acc"]))
    finally:
        model.set_runtime_gates(SBRuntimeGates(**previous_gates))
        model.train()

    mean_answer_loss = sum(all_answer_losses) / max(len(all_answer_losses), 1)
    mean_answer_token_acc = sum(all_answer_token_accs) / max(len(all_answer_token_accs), 1)

    return {
        "tasks": task_reports,
        "aggregate": {
            "mean_answer_loss": mean_answer_loss,
            "mean_answer_token_acc": mean_answer_token_acc,
            "mean_episodic_replay_schema_alignment": sum(
                float(item["mean_episodic_replay_schema_alignment"]) for item in task_reports
            )
            / max(len(task_reports), 1),
            "mean_summary_schema_alignment": sum(
                float(item["mean_summary_schema_alignment"]) for item in task_reports
            )
            / max(len(task_reports), 1),
            "mean_scene_schema_alignment": sum(float(item["mean_scene_schema_alignment"]) for item in task_reports)
            / max(len(task_reports), 1),
            "schema_chain_activated": bool(
                task_reports
                and all(
                    float(item["mean_summary_schema_alignment"]) > 0.0
                    and float(item["mean_scene_schema_alignment"]) > 0.0
                    for item in task_reports
                )
            ),
            "schema_chain_activated_ratio": sum(
                float(item["schema_chain_activated_ratio"]) for item in task_reports
            )
            / max(len(task_reports), 1),
            "selection_score": mean_answer_token_acc * 100.0 - mean_answer_loss,
            "carry_memory": bool(carry_memory),
            "prompt_char_limit": int(prompt_char_limit),
            "carry_policy": str(carry_policy),
        },
    }
