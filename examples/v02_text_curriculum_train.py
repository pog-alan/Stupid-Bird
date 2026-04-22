from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import torch

from sb.core_lm_torch import (
    SBCoreMiniLM,
    SBCoreMiniTorchConfig,
    SBCoreMemoryState,
    next_token_loss,
    runtime_device_report,
    staged_runtime_gates,
)
from sb.longbench_local_eval import DEFAULT_CARRY_POLICY, DEFAULT_TASKS, evaluate_longbench_local
from sb.text_corpus import (
    TextBatch,
    load_stage_corpus,
    load_longbench_rows,
    load_prepared_corpus_paths,
    load_text_tokenizer,
    prepare_local_text_corpus,
    sample_longbench_answer_batch,
    sample_stage_batch,
    summarize_stage_corpus,
)


OUTPUT_PATH = Path("data/processed/experiments/text_curriculum_train.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run SB-Core text curriculum training on local Chinese corpora.")
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--foundation-steps", type=int, default=10)
    parser.add_argument("--structured-steps", type=int, default=8)
    parser.add_argument("--long-steps", type=int, default=6)
    parser.add_argument("--foundation-seq-len", type=int, default=80)
    parser.add_argument("--structured-seq-len", type=int, default=96)
    parser.add_argument("--long-seq-len", type=int, default=128)
    parser.add_argument("--foundation-batch-size", type=int, default=4)
    parser.add_argument("--structured-batch-size", type=int, default=4)
    parser.add_argument("--long-batch-size", type=int, default=2)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--state-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--signal-schema-slots", type=int, default=7)
    parser.add_argument("--semantic-slots", type=int, default=12)
    parser.add_argument("--working-slots", type=int, default=12)
    parser.add_argument("--router-top-k", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=4e-3)
    parser.add_argument("--prepared-corpus-manifest", default="")
    parser.add_argument("--wikipedia-limit", type=int, default=1000)
    parser.add_argument("--clue-limit-per-subset", type=int, default=1024)
    parser.add_argument("--longbench-limit-per-task", type=int, default=128)
    parser.add_argument("--max-vocab-size", type=int, default=4096)
    parser.add_argument("--sample-preview-len", type=int, default=120)
    parser.add_argument("--tokenizer-kind", choices=["char", "subword"], default="subword")
    parser.add_argument("--checkpoint-dir", default="data/processed/checkpoints/text_curriculum")
    parser.add_argument("--checkpoint-every", type=int, default=0)
    parser.add_argument("--resume-from", default="")
    parser.add_argument("--stop-after-steps", type=int, default=0)
    parser.add_argument("--eval-every", type=int, default=0)
    parser.add_argument("--eval-max-samples", type=int, default=4)
    parser.add_argument("--eval-tasks", default=",".join(DEFAULT_TASKS))
    parser.add_argument(
        "--eval-carry-policy",
        choices=["uniform", "task_adaptive"],
        default="task_adaptive",
    )
    parser.add_argument("--stage-val-batches", type=int, default=4)
    parser.add_argument("--train-log-every", type=int, default=10)
    parser.add_argument("--metrics-jsonl", default="")
    parser.add_argument("--longbench-answer-supervision", action="store_true")
    parser.add_argument("--longbench-answer-ratio", type=float, default=0.75)
    parser.add_argument("--carry-memory-stages", default="long_context")
    parser.add_argument("--carry-memory-reset-every", type=int, default=0)
    parser.add_argument("--experiment-tag", default="")
    parser.add_argument("--output-path", default="")
    return parser


def build_stage_plan(args: argparse.Namespace) -> List[Dict[str, int | str]]:
    return [
        {
            "name": "foundation",
            "steps": args.foundation_steps,
            "seq_len": args.foundation_seq_len,
            "batch_size": args.foundation_batch_size,
        },
        {
            "name": "structured",
            "steps": args.structured_steps,
            "seq_len": args.structured_seq_len,
            "batch_size": args.structured_batch_size,
        },
        {
            "name": "long_context",
            "steps": args.long_steps,
            "seq_len": args.long_seq_len,
            "batch_size": args.long_batch_size,
        },
    ]


def _stage_step_schedule(stages: List[Dict[str, int | str]]) -> List[Dict[str, int | str]]:
    schedule: List[Dict[str, int | str]] = []
    for stage in stages:
        schedule.extend([stage] * int(stage["steps"]))
    return schedule


def _set_seeds(seed: int) -> random.Random:
    random_gen = random.Random(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return random_gen


def _save_checkpoint(
    *,
    checkpoint_dir: Path,
    checkpoint_name: str,
    model: SBCoreMiniLM,
    optimizer: torch.optim.Optimizer,
    random_gen: random.Random,
    global_step: int,
    stage_reports: List[Dict[str, Any]],
    stage_losses: Dict[str, List[float]],
    eval_reports: List[Dict[str, Any]],
    best_eval: Dict[str, Any],
    corpus_paths: Dict[str, str],
    stages: List[Dict[str, int | str]],
    args: argparse.Namespace,
    carry_state: SBCoreMemoryState | None = None,
    carry_stage_name: str = "",
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "runtime_gates": model.get_runtime_gates(),
        "global_step": global_step,
        "stage_reports": stage_reports,
        "stage_losses": stage_losses,
        "eval_reports": eval_reports,
        "best_eval": best_eval,
        "corpus_paths": corpus_paths,
        "stages": stages,
        "args": vars(args),
        "carry_state": carry_state.detached() if carry_state is not None else None,
        "carry_stage_name": carry_stage_name,
        "python_random_state": random.getstate(),
        "random_gen_state": random_gen.getstate(),
        "torch_rng_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        payload["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    torch.save(payload, checkpoint_dir / checkpoint_name)


def _load_checkpoint(path: Path) -> Dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def _resolve_resume_path(path_str: str, checkpoint_dir: Path) -> Path | None:
    if not path_str:
        return None
    path = Path(path_str)
    if not path.is_absolute():
        path = Path.cwd() / path
    return path


def _resolve_metrics_path(path_str: str, checkpoint_dir: Path) -> Path:
    if not path_str:
        return checkpoint_dir / "events.jsonl"
    path = Path(path_str)
    if not path.is_absolute():
        path = Path.cwd() / path
    return path


def _resolve_output_path(path_str: str, experiment_tag: str) -> Path:
    if path_str:
        path = Path(path_str)
    elif experiment_tag:
        path = OUTPUT_PATH.with_name(f"{OUTPUT_PATH.stem}_{experiment_tag}{OUTPUT_PATH.suffix}")
    else:
        path = OUTPUT_PATH
    if not path.is_absolute():
        path = Path.cwd() / path
    return path


def _append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _effective_longbench_answer_ratio(args: argparse.Namespace) -> float:
    if float(args.longbench_answer_ratio) >= 0.0:
        return float(min(1.0, max(0.0, args.longbench_answer_ratio)))
    return 1.0 if bool(args.longbench_answer_supervision) else 0.0


def _schema_chain_summary_from_aux(aux: Dict[str, Any] | None) -> Dict[str, Any]:
    aux = aux or {}
    episodic = float(aux.get("episodic_replay_schema_alignment_mean", 0.0))
    summary = float(aux.get("summary_schema_alignment_mean", 0.0))
    scene = float(aux.get("scene_schema_alignment_mean", 0.0))
    return {
        "episodic_replay_schema_alignment_mean": episodic,
        "summary_schema_alignment_mean": summary,
        "scene_schema_alignment_mean": scene,
        "schema_chain_activated": bool(summary > 0.0 and scene > 0.0),
    }


def _aggregate_schema_chain_summaries(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not items:
        return {
            "episodic_replay_schema_alignment_mean": 0.0,
            "summary_schema_alignment_mean": 0.0,
            "scene_schema_alignment_mean": 0.0,
            "schema_chain_activated_ratio": 0.0,
            "schema_chain_activated": False,
        }
    episodic = sum(float(item.get("episodic_replay_schema_alignment_mean", 0.0)) for item in items) / len(items)
    summary = sum(float(item.get("summary_schema_alignment_mean", 0.0)) for item in items) / len(items)
    scene = sum(float(item.get("scene_schema_alignment_mean", 0.0)) for item in items) / len(items)
    active_ratio = sum(1.0 for item in items if bool(item.get("schema_chain_activated", False))) / len(items)
    return {
        "episodic_replay_schema_alignment_mean": episodic,
        "summary_schema_alignment_mean": summary,
        "scene_schema_alignment_mean": scene,
        "schema_chain_activated_ratio": active_ratio,
        "schema_chain_activated": bool(active_ratio > 0.0),
    }


def _merge_batches(left: TextBatch | None, right: TextBatch | None) -> TextBatch:
    if left is None:
        if right is None:
            raise ValueError("at least one batch must be provided")
        return right
    if right is None:
        return left
    return TextBatch(
        input_ids=torch.cat([left.input_ids, right.input_ids], dim=0),
        target_ids=torch.cat([left.target_ids, right.target_ids], dim=0),
        focus_mask=torch.cat([left.focus_mask, right.focus_mask], dim=0),
    )


def _evaluate_stage_validation(
    *,
    model: SBCoreMiniLM,
    stage_name: str,
    stage: Dict[str, int | str],
    stage_val_corpora: Dict[str, List[str]],
    tokenizer,
    device: str | torch.device,
    rng: random.Random,
    validation_batches: int,
) -> Dict[str, Any]:
    validation_rows = stage_val_corpora.get(stage_name) or []
    if not validation_rows or validation_batches <= 0:
        return {
            "available": False,
            "validation_rows": len(validation_rows),
            "validation_batches": 0,
            "mean_loss": 0.0,
            "schema_chain": _schema_chain_summary_from_aux(None),
        }

    was_training = model.training
    losses: List[float] = []
    schema_items: List[Dict[str, Any]] = []
    model.eval()
    with torch.no_grad():
        for _ in range(validation_batches):
            batch = sample_stage_batch(
                stage_val_corpora,
                tokenizer,
                stage_name=stage_name,
                batch_size=int(stage["batch_size"]),
                seq_len=int(stage["seq_len"]),
                device=device,
                rng=rng,
            )
            forward = model(batch.input_ids)
            loss = next_token_loss(forward["logits"], batch.target_ids, focus_mask=batch.focus_mask)
            losses.append(float(loss.detach().cpu()))
            schema_items.append(_schema_chain_summary_from_aux(forward.get("aux")))
    if was_training:
        model.train()

    return {
        "available": True,
        "validation_rows": len(validation_rows),
        "validation_batches": validation_batches,
        "mean_loss": sum(losses) / max(len(losses), 1),
        "schema_chain": _aggregate_schema_chain_summaries(schema_items),
    }


def main() -> None:
    args = build_parser().parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_dir = Path(args.checkpoint_dir)
    resume_path = _resolve_resume_path(args.resume_from, checkpoint_dir)
    metrics_path = _resolve_metrics_path(args.metrics_jsonl, checkpoint_dir)
    output_path = _resolve_output_path(args.output_path, args.experiment_tag)
    prepared_corpus_manifest = Path(args.prepared_corpus_manifest) if args.prepared_corpus_manifest else None
    if prepared_corpus_manifest is not None and not prepared_corpus_manifest.is_absolute():
        prepared_corpus_manifest = Path.cwd() / prepared_corpus_manifest

    stages = build_stage_plan(args)
    random_gen = _set_seeds(args.seed)

    if resume_path is not None and resume_path.exists():
        checkpoint = _load_checkpoint(resume_path)
        corpus_paths = checkpoint["corpus_paths"]
        stages = checkpoint["stages"]
        config = SBCoreMiniTorchConfig(**checkpoint["args"]["model_config"])
        tokenizer = load_text_tokenizer(corpus_paths["tokenizer_path"])
        stage_corpora = {
            "foundation": load_stage_corpus(corpus_paths["foundation_path"]),
            "structured": load_stage_corpus(corpus_paths["structured_path"]),
            "long_context": load_stage_corpus(corpus_paths["long_context_path"]),
        }
        stage_val_corpora = {
            "foundation": load_stage_corpus(corpus_paths["foundation_val_path"]) if corpus_paths.get("foundation_val_path") else [],
            "structured": load_stage_corpus(corpus_paths["structured_val_path"]) if corpus_paths.get("structured_val_path") else [],
            "long_context": load_stage_corpus(corpus_paths["long_context_val_path"]) if corpus_paths.get("long_context_val_path") else [],
        }
        model = SBCoreMiniLM(config).to(device)
        model.load_state_dict(checkpoint["model_state"])
        model.set_runtime_gates(type(model.runtime_gates)(**checkpoint["runtime_gates"]))
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        global_step = int(checkpoint["global_step"])
        stage_reports = list(checkpoint["stage_reports"])
        stage_losses = {key: list(value) for key, value in checkpoint["stage_losses"].items()}
        eval_reports = list(checkpoint.get("eval_reports", []))
        best_eval = dict(checkpoint.get("best_eval", {}))
        carry_state = checkpoint.get("carry_state")
        if carry_state is not None:
            carry_state = carry_state.moved_to(device)
        carry_stage_name = str(checkpoint.get("carry_stage_name", ""))
        random.setstate(checkpoint["python_random_state"])
        if "random_gen_state" in checkpoint:
            random_gen.setstate(checkpoint["random_gen_state"])
        torch.set_rng_state(checkpoint["torch_rng_state"])
        if torch.cuda.is_available() and "cuda_rng_state_all" in checkpoint:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state_all"])
    else:
        if prepared_corpus_manifest is not None and prepared_corpus_manifest.exists():
            corpus = load_prepared_corpus_paths(prepared_corpus_manifest)
        else:
            corpus = prepare_local_text_corpus(
                "data/manifest.json",
                wikipedia_limit=args.wikipedia_limit,
                clue_limit_per_subset=args.clue_limit_per_subset,
                longbench_limit_per_task=args.longbench_limit_per_task,
                max_vocab_size=args.max_vocab_size,
                tokenizer_kind=args.tokenizer_kind,
                profile_name=args.experiment_tag or "text_curriculum",
            )
        corpus_paths = asdict(corpus)
        tokenizer = load_text_tokenizer(corpus.tokenizer_path)
        stage_corpora = {
            "foundation": load_stage_corpus(corpus.foundation_path),
            "structured": load_stage_corpus(corpus.structured_path),
            "long_context": load_stage_corpus(corpus.long_context_path),
        }
        stage_val_corpora = {
            "foundation": load_stage_corpus(corpus.foundation_val_path) if corpus.foundation_val_path else [],
            "structured": load_stage_corpus(corpus.structured_val_path) if corpus.structured_val_path else [],
            "long_context": load_stage_corpus(corpus.long_context_val_path) if corpus.long_context_val_path else [],
        }
        config = SBCoreMiniTorchConfig(
            vocab_size=tokenizer.vocab_size,
            d_model=args.d_model,
            state_dim=args.state_dim,
            num_layers=args.num_layers,
            signal_schema_slots=args.signal_schema_slots,
            semantic_memory_slots=args.semantic_slots,
            working_memory_slots=args.working_slots,
            router_top_k=args.router_top_k,
            dropout=0.0,
            max_seq_len=max(args.foundation_seq_len, args.structured_seq_len, args.long_seq_len) + 16,
        )
        model = SBCoreMiniLM(config).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
        global_step = 0
        stage_reports = []
        stage_losses = {str(stage["name"]): [] for stage in stages}
        eval_reports = []
        best_eval = {}
        carry_state = None
        carry_stage_name = ""

    longbench_answer_rows = load_longbench_rows(
        "data/manifest.json",
        longbench_limit_per_task=args.longbench_limit_per_task,
    )
    longbench_answer_ratio = _effective_longbench_answer_ratio(args)

    total_steps = sum(int(stage["steps"]) for stage in stages)
    target_total_steps = total_steps
    if args.stop_after_steps > 0:
        target_total_steps = min(total_steps, global_step + args.stop_after_steps)
    warmup_steps = max(20, total_steps // 6)
    schedule = _stage_step_schedule(stages)
    eval_task_names = [item.strip() for item in args.eval_tasks.split(",") if item.strip()]
    carry_memory_stage_names = {
        item.strip() for item in str(args.carry_memory_stages).split(",") if item.strip()
    }
    carry_memory_reset_every = max(int(args.carry_memory_reset_every), 0)
    resume_carry_reset_reason = ""
    if carry_state is not None:
        next_stage_name = str(schedule[global_step]["name"]) if global_step < len(schedule) else ""
        if not carry_stage_name:
            carry_state = None
            resume_carry_reset_reason = "resume_missing_stage_name"
        elif carry_stage_name not in carry_memory_stage_names:
            carry_state = None
            carry_stage_name = ""
            resume_carry_reset_reason = "resume_stage_not_carry_enabled"
        elif next_stage_name != carry_stage_name:
            carry_state = None
            carry_stage_name = ""
            resume_carry_reset_reason = "resume_stage_mismatch"
    for stage in stages:
        stage_losses.setdefault(str(stage["name"]), [])
    completed_stage_names = {str(report["stage"]) for report in stage_reports}

    for step_index in range(global_step, target_total_steps):
        stage = schedule[step_index]
        stage_name = str(stage["name"])
        use_memory_carry = stage_name in carry_memory_stage_names
        scheduler_stage_name, stage_gates = staged_runtime_gates(step_index=step_index, total_steps=total_steps)
        model.set_runtime_gates(stage_gates)
        lr_scale = min(1.0, float(step_index + 1) / float(warmup_steps))
        for group in optimizer.param_groups:
            group["lr"] = args.learning_rate * lr_scale
        carry_reset_reason = resume_carry_reset_reason
        resume_carry_reset_reason = ""
        if not use_memory_carry:
            if carry_state is not None:
                carry_reset_reason = carry_reset_reason or "stage_not_carry_enabled"
            carry_state = None
            carry_stage_name = ""
        elif carry_stage_name and carry_stage_name != stage_name:
            carry_reset_reason = carry_reset_reason or "stage_mismatch"
            carry_state = None
            carry_stage_name = ""

        long_context_answer_fraction = 0.0
        if str(stage["name"]) == "long_context" and longbench_answer_ratio > 0.0:
            total_batch_size = int(stage["batch_size"])
            answer_batch_size = sum(1 for _ in range(total_batch_size) if random_gen.random() < longbench_answer_ratio)
            answer_batch: TextBatch | None = None
            continuation_batch: TextBatch | None = None
            if answer_batch_size > 0:
                answer_batch = sample_longbench_answer_batch(
                    longbench_answer_rows,
                    tokenizer,
                    batch_size=answer_batch_size,
                    seq_len=int(stage["seq_len"]),
                    device=device,
                    rng=random_gen,
                )
            continuation_batch_size = total_batch_size - answer_batch_size
            if continuation_batch_size > 0:
                continuation_batch = sample_stage_batch(
                    stage_corpora,
                    tokenizer,
                    stage_name=str(stage["name"]),
                    batch_size=continuation_batch_size,
                    seq_len=int(stage["seq_len"]),
                    device=device,
                    rng=random_gen,
                )
            batch = _merge_batches(answer_batch, continuation_batch)
            long_context_answer_fraction = answer_batch_size / max(total_batch_size, 1)
        else:
            batch = sample_stage_batch(
                stage_corpora,
                tokenizer,
                stage_name=str(stage["name"]),
                batch_size=int(stage["batch_size"]),
                seq_len=int(stage["seq_len"]),
                device=device,
                rng=random_gen,
            )

        optimizer.zero_grad(set_to_none=True)
        stage_step_index = len(stage_losses[stage_name])
        if use_memory_carry and carry_memory_reset_every > 0 and stage_step_index > 0:
            if stage_step_index % carry_memory_reset_every == 0:
                carry_reset_reason = carry_reset_reason or "periodic_reset"
                carry_state = None
                carry_stage_name = ""

        forward = model(
            batch.input_ids,
            memory_state=carry_state,
            return_state=use_memory_carry,
        )
        if use_memory_carry:
            carry_state = forward.get("state")  # type: ignore[assignment]
            carry_stage_name = stage_name
        else:
            carry_state = None
            carry_stage_name = ""
        loss = next_token_loss(forward["logits"], batch.target_ids, focus_mask=batch.focus_mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        stage_losses[stage_name].append(float(loss.detach().cpu()))
        global_step = step_index + 1

        if args.train_log_every > 0 and (global_step == 1 or global_step % args.train_log_every == 0):
            _append_jsonl(
                metrics_path,
                {
                    "type": "train_step",
                    "step": global_step,
                    "stage": stage_name,
                    "scheduler_stage": scheduler_stage_name,
                    "loss": float(loss.detach().cpu()),
                    "lr": float(optimizer.param_groups[0]["lr"]),
                    "long_context_answer_fraction": long_context_answer_fraction,
                    "use_memory_carry": use_memory_carry,
                    "carry_state_active": carry_state is not None,
                    "carry_stage_name": carry_stage_name,
                    "carry_reset_reason": carry_reset_reason,
                    "runtime_gates": model.get_runtime_gates(),
                },
            )

        natural_stage_boundary = (
            step_index == total_steps - 1
            or schedule[step_index + 1]["name"] != stage_name
        )
        if natural_stage_boundary and stage_name not in completed_stage_names:
            stage_validation = _evaluate_stage_validation(
                model=model,
                stage_name=stage_name,
                stage=stage,
                stage_val_corpora=stage_val_corpora,
                tokenizer=tokenizer,
                device=device,
                rng=random_gen,
                validation_batches=int(args.stage_val_batches),
            )
            stage_report = {
                "stage": stage_name,
                "steps": int(stage["steps"]),
                "seq_len": int(stage["seq_len"]),
                "batch_size": int(stage["batch_size"]),
                "final_loss": stage_losses[stage_name][-1],
                "mean_loss": sum(stage_losses[stage_name]) / max(len(stage_losses[stage_name]), 1),
                "runtime_gates": model.get_runtime_gates(),
                "aux": forward["aux"],
                "schema_chain": _schema_chain_summary_from_aux(forward["aux"]),
                "scheduler_stage": scheduler_stage_name,
                "long_context_answer_ratio": longbench_answer_ratio if stage_name == "long_context" else 0.0,
                "use_memory_carry": use_memory_carry,
                "carry_state_active": carry_state is not None,
                "carry_stage_name": carry_stage_name,
                "carry_reset_reason": carry_reset_reason,
                "validation": stage_validation,
            }
            stage_reports.append(stage_report)
            _append_jsonl(
                metrics_path,
                {
                    "type": "stage_complete",
                    "step": global_step,
                    **stage_report,
                },
            )
            completed_stage_names.add(stage_name)
            carry_state = None
            carry_stage_name = ""
            _save_checkpoint(
                checkpoint_dir=checkpoint_dir,
                checkpoint_name=f"stage_{stage_name}.pt",
                model=model,
                optimizer=optimizer,
                random_gen=random_gen,
                global_step=global_step,
                stage_reports=stage_reports,
                stage_losses=stage_losses,
                eval_reports=eval_reports,
                best_eval=best_eval,
                corpus_paths=corpus_paths,
                stages=stages,
                args=argparse.Namespace(**{**vars(args), "model_config": asdict(config)}),
                carry_state=carry_state,
                carry_stage_name=carry_stage_name,
            )

        if args.eval_every > 0 and global_step % args.eval_every == 0:
            eval_report = evaluate_longbench_local(
                model,
                tokenizer,
                device=device,
                tasks=eval_task_names,
                max_samples=args.eval_max_samples,
                carry_memory=True,
                carry_policy=args.eval_carry_policy,
            )
            aggregate = dict(eval_report["aggregate"])
            eval_entry = {
                "step": global_step,
                "scheduler_stage": scheduler_stage_name,
                "carry_policy": str(aggregate.get("carry_policy", args.eval_carry_policy)),
                "aggregate": aggregate,
                "tasks": eval_report["tasks"],
                "schema_chain": {
                    "episodic_replay_schema_alignment_mean": float(
                        aggregate.get("mean_episodic_replay_schema_alignment", 0.0)
                    ),
                    "summary_schema_alignment_mean": float(aggregate.get("mean_summary_schema_alignment", 0.0)),
                    "scene_schema_alignment_mean": float(aggregate.get("mean_scene_schema_alignment", 0.0)),
                    "schema_chain_activated": bool(aggregate.get("schema_chain_activated", False)),
                },
            }
            eval_reports.append(eval_entry)
            _append_jsonl(
                metrics_path,
                {
                    "type": "eval",
                    **eval_entry,
                },
            )
            current_score = float(aggregate["selection_score"])
            best_score = float(best_eval.get("selection_score", float("-inf")))
            if current_score > best_score:
                best_eval = {
                    "step": global_step,
                    **aggregate,
                }
                _append_jsonl(
                    metrics_path,
                    {
                        "type": "best_eval",
                        **best_eval,
                    },
                )
                _save_checkpoint(
                    checkpoint_dir=checkpoint_dir,
                    checkpoint_name="best.pt",
                    model=model,
                    optimizer=optimizer,
                    random_gen=random_gen,
                    global_step=global_step,
                    stage_reports=stage_reports,
                    stage_losses=stage_losses,
                    eval_reports=eval_reports,
                    best_eval=best_eval,
                    corpus_paths=corpus_paths,
                    stages=stages,
                    args=argparse.Namespace(**{**vars(args), "model_config": asdict(config)}),
                    carry_state=carry_state,
                    carry_stage_name=carry_stage_name,
                )

        if args.checkpoint_every > 0 and global_step % args.checkpoint_every == 0:
            _save_checkpoint(
                checkpoint_dir=checkpoint_dir,
                checkpoint_name=f"step_{global_step}.pt",
                model=model,
                optimizer=optimizer,
                random_gen=random_gen,
                global_step=global_step,
                stage_reports=stage_reports,
                stage_losses=stage_losses,
                eval_reports=eval_reports,
                best_eval=best_eval,
                corpus_paths=corpus_paths,
                stages=stages,
                args=argparse.Namespace(**{**vars(args), "model_config": asdict(config)}),
                carry_state=carry_state,
                carry_stage_name=carry_stage_name,
            )
        _save_checkpoint(
            checkpoint_dir=checkpoint_dir,
            checkpoint_name="last.pt",
            model=model,
            optimizer=optimizer,
            random_gen=random_gen,
            global_step=global_step,
            stage_reports=stage_reports,
            stage_losses=stage_losses,
            eval_reports=eval_reports,
            best_eval=best_eval,
            corpus_paths=corpus_paths,
            stages=stages,
            args=argparse.Namespace(**{**vars(args), "model_config": asdict(config)}),
            carry_state=carry_state,
            carry_stage_name=carry_stage_name,
        )

    final_scheduler_stage, final_gates = staged_runtime_gates(
        step_index=max(global_step - 1, 0),
        total_steps=max(total_steps, 1),
    )
    model.set_runtime_gates(final_gates)
    if args.eval_every > 0 and eval_task_names and (not eval_reports or int(eval_reports[-1]["step"]) != global_step):
        eval_report = evaluate_longbench_local(
            model,
            tokenizer,
            device=device,
            tasks=eval_task_names,
            max_samples=args.eval_max_samples,
            carry_memory=True,
            carry_policy=args.eval_carry_policy,
        )
        aggregate = dict(eval_report["aggregate"])
        eval_entry = {
            "step": global_step,
            "scheduler_stage": final_scheduler_stage,
            "carry_policy": str(aggregate.get("carry_policy", args.eval_carry_policy)),
            "aggregate": aggregate,
            "tasks": eval_report["tasks"],
            "schema_chain": {
                "episodic_replay_schema_alignment_mean": float(
                    aggregate.get("mean_episodic_replay_schema_alignment", 0.0)
                ),
                "summary_schema_alignment_mean": float(aggregate.get("mean_summary_schema_alignment", 0.0)),
                "scene_schema_alignment_mean": float(aggregate.get("mean_scene_schema_alignment", 0.0)),
                "schema_chain_activated": bool(aggregate.get("schema_chain_activated", False)),
            },
        }
        eval_reports.append(eval_entry)
        _append_jsonl(
            metrics_path,
            {
                "type": "eval",
                **eval_entry,
            },
        )
        current_score = float(aggregate["selection_score"])
        best_score = float(best_eval.get("selection_score", float("-inf")))
        if current_score > best_score:
            best_eval = {
                "step": global_step,
                **aggregate,
            }
            _append_jsonl(
                metrics_path,
                {
                    "type": "best_eval",
                    **best_eval,
                },
            )
            _save_checkpoint(
                checkpoint_dir=checkpoint_dir,
                checkpoint_name="best.pt",
                model=model,
                optimizer=optimizer,
                random_gen=random_gen,
                global_step=global_step,
                stage_reports=stage_reports,
                stage_losses=stage_losses,
                eval_reports=eval_reports,
                best_eval=best_eval,
                corpus_paths=corpus_paths,
                stages=stages,
                args=argparse.Namespace(**{**vars(args), "model_config": asdict(config)}),
                carry_state=carry_state,
                carry_stage_name=carry_stage_name,
            )
        _save_checkpoint(
            checkpoint_dir=checkpoint_dir,
            checkpoint_name="last.pt",
            model=model,
            optimizer=optimizer,
            random_gen=random_gen,
            global_step=global_step,
            stage_reports=stage_reports,
            stage_losses=stage_losses,
            eval_reports=eval_reports,
            best_eval=best_eval,
            corpus_paths=corpus_paths,
            stages=stages,
            args=argparse.Namespace(**{**vars(args), "model_config": asdict(config)}),
            carry_state=carry_state,
            carry_stage_name=carry_stage_name,
        )

    sample_batch = sample_stage_batch(
        stage_corpora,
        tokenizer,
        stage_name="long_context",
        batch_size=1,
        seq_len=args.long_seq_len,
        device=device,
        rng=random_gen,
    )
    model.set_runtime_gates(staged_runtime_gates(step_index=max(total_steps - 1, 0), total_steps=max(total_steps, 1))[1])
    with torch.no_grad():
        sample_forward = model(sample_batch.input_ids)
        predictions = sample_forward["logits"].argmax(dim=-1)
        sample_loss = next_token_loss(sample_forward["logits"], sample_batch.target_ids, focus_mask=sample_batch.focus_mask)

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment_tag": args.experiment_tag,
        "runtime": runtime_device_report(device),
        "resume_from": str(resume_path.resolve()) if resume_path is not None and resume_path.exists() else "",
        "corpus": summarize_stage_corpus(corpus_paths),
        "model_config": asdict(config),
        "tokenizer_vocab_size": tokenizer.vocab_size,
        "longbench_answer_supervision": bool(args.longbench_answer_supervision),
        "longbench_answer_ratio": longbench_answer_ratio,
        "eval_carry_policy": str(args.eval_carry_policy or DEFAULT_CARRY_POLICY),
        "carry_memory_stages": sorted(carry_memory_stage_names),
        "carry_memory_reset_every": carry_memory_reset_every,
        "training_plan": stages,
        "completed_steps": global_step,
        "target_total_steps": total_steps,
        "stopped_early": global_step < total_steps,
        "checkpoint_dir": str(checkpoint_dir.resolve()),
        "metrics_path": str(metrics_path.resolve()),
        "best_checkpoint_path": str((checkpoint_dir / "best.pt").resolve()) if (checkpoint_dir / "best.pt").exists() else "",
        "stage_reports": stage_reports,
        "eval_reports": eval_reports,
        "best_eval": best_eval,
        "schema_chain_summary": {
            "training": _aggregate_schema_chain_summaries(
                [dict(report.get("schema_chain", {})) for report in stage_reports]
            ),
            "evaluation": _aggregate_schema_chain_summaries(
                [dict(report.get("schema_chain", {})) for report in eval_reports]
            ),
            "validation": _aggregate_schema_chain_summaries(
                [
                    dict(report.get("validation", {}).get("schema_chain", {}))
                    for report in stage_reports
                    if bool(report.get("validation", {}).get("available", False))
                ]
            ),
        },
        "sample_eval": {
            "loss": float(sample_loss.detach().cpu()),
            "input_preview": tokenizer.decode(sample_batch.input_ids[0][: args.sample_preview_len].tolist()),
            "target_preview": tokenizer.decode(sample_batch.target_ids[0][: args.sample_preview_len].tolist()),
            "prediction_preview": tokenizer.decode(predictions[0][: args.sample_preview_len].tolist()),
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True, indent=2))
    print(f"report_path={output_path}")


if __name__ == "__main__":
    main()
