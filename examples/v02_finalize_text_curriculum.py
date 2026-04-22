from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

import torch

from sb.core_lm_torch import (
    SBCoreMiniLM,
    SBCoreMiniTorchConfig,
    next_token_loss,
    runtime_device_report,
    staged_runtime_gates,
)
from sb.longbench_local_eval import DEFAULT_CARRY_POLICY, DEFAULT_TASKS, evaluate_longbench_local
from sb.text_corpus import (
    load_prepared_corpus_paths,
    load_stage_corpus,
    load_text_tokenizer,
    sample_stage_batch,
    summarize_stage_corpus,
)


DEFAULT_OUTPUT_PATH = Path("data/processed/experiments/text_curriculum_train.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Finalize an SB-Core text curriculum checkpoint into a report JSON.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prepared-corpus-manifest", default="")
    parser.add_argument("--experiment-tag", default="")
    parser.add_argument("--output-path", default="")
    parser.add_argument("--sample-preview-len", type=int, default=120)
    parser.add_argument("--refresh-eval", action="store_true")
    parser.add_argument("--eval-max-samples", type=int, default=0)
    parser.add_argument("--eval-tasks", default=",".join(DEFAULT_TASKS))
    parser.add_argument(
        "--eval-carry-policy",
        choices=["uniform", "task_adaptive"],
        default="",
    )
    parser.add_argument("--no-carry-memory", action="store_true")
    return parser


def _resolve_output_path(path_str: str, experiment_tag: str) -> Path:
    if path_str:
        return Path(path_str)
    if experiment_tag:
        return DEFAULT_OUTPUT_PATH.with_name(f"{DEFAULT_OUTPUT_PATH.stem}_{experiment_tag}{DEFAULT_OUTPUT_PATH.suffix}")
    return DEFAULT_OUTPUT_PATH


def _schema_chain_summary_from_aux(aux: Dict[str, Any] | None) -> Dict[str, Any]:
    aux = aux or {}
    summary = {
        "episodic_replay_schema_alignment_mean": float(aux.get("episodic_replay_schema_alignment_mean", 0.0)),
        "summary_schema_alignment_mean": float(aux.get("summary_schema_alignment_mean", 0.0)),
        "scene_schema_alignment_mean": float(aux.get("scene_schema_alignment_mean", 0.0)),
    }
    summary["schema_chain_activated"] = bool(
        summary["summary_schema_alignment_mean"] > 0.0 or summary["scene_schema_alignment_mean"] > 0.0
    )
    return summary


def _aggregate_schema_chain_summaries(items: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    rows = [item for item in items if item]
    if not rows:
        return {
            "episodic_replay_schema_alignment_mean": 0.0,
            "summary_schema_alignment_mean": 0.0,
            "scene_schema_alignment_mean": 0.0,
            "schema_chain_activated_ratio": 0.0,
            "schema_chain_activated": False,
        }
    count = float(len(rows))
    episodic = sum(float(item.get("episodic_replay_schema_alignment_mean", 0.0)) for item in rows) / count
    summary = sum(float(item.get("summary_schema_alignment_mean", 0.0)) for item in rows) / count
    scene = sum(float(item.get("scene_schema_alignment_mean", 0.0)) for item in rows) / count
    activated_ratio = sum(1.0 for item in rows if bool(item.get("schema_chain_activated", False))) / count
    return {
        "episodic_replay_schema_alignment_mean": episodic,
        "summary_schema_alignment_mean": summary,
        "scene_schema_alignment_mean": scene,
        "schema_chain_activated_ratio": activated_ratio,
        "schema_chain_activated": bool(activated_ratio > 0.0),
    }


def _load_corpus_paths(checkpoint: Dict[str, Any], prepared_manifest: Path | None) -> Dict[str, str]:
    if prepared_manifest is not None and prepared_manifest.exists():
        return asdict(load_prepared_corpus_paths(prepared_manifest))
    return dict(checkpoint["corpus_paths"])


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _merge_events_into_reports(
    stage_reports: List[Dict[str, Any]],
    eval_reports: List[Dict[str, Any]],
    best_eval: Dict[str, Any],
    metrics_path: Path,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    rows = _read_jsonl(metrics_path)
    if not rows:
        return stage_reports, eval_reports, best_eval

    stage_by_name = {str(item.get("stage", "")): dict(item) for item in stage_reports}
    eval_by_step = {int(item.get("step", -1)): dict(item) for item in eval_reports}
    latest_best = dict(best_eval)

    for row in rows:
        row_type = str(row.get("type", ""))
        if row_type == "stage_complete":
            stage_name = str(row.get("stage", ""))
            if stage_name:
                stage_payload = dict(row)
                stage_payload.pop("type", None)
                stage_payload.pop("step", None)
                stage_by_name[stage_name] = stage_payload
        elif row_type == "eval":
            step = int(row.get("step", -1))
            if step >= 0:
                eval_payload = dict(row)
                eval_payload.pop("type", None)
                eval_by_step[step] = eval_payload
        elif row_type == "best_eval":
            best_payload = dict(row)
            best_payload.pop("type", None)
            latest_best = best_payload

    stage_order = [str(item.get("stage", "")) for item in stage_reports]
    merged_stage_reports = [stage_by_name[name] for name in stage_order if name in stage_by_name]
    for name, payload in stage_by_name.items():
        if name not in stage_order:
            merged_stage_reports.append(payload)
    merged_eval_reports = [eval_by_step[step] for step in sorted(eval_by_step)]
    return merged_stage_reports, merged_eval_reports, latest_best


def _build_model(checkpoint: Dict[str, Any], device: torch.device) -> SBCoreMiniLM:
    args_dict = dict(checkpoint.get("args", {}))
    model_config = dict(args_dict.get("model_config", {}))
    config = SBCoreMiniTorchConfig(**model_config)
    model = SBCoreMiniLM(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def _resolve_eval_carry_policy(args: argparse.Namespace, checkpoint_args: Dict[str, Any]) -> str:
    requested = str(args.eval_carry_policy or "").strip()
    if requested:
        return requested
    checkpoint_value = str(checkpoint_args.get("eval_carry_policy", "")).strip()
    if checkpoint_value:
        return checkpoint_value
    return "task_adaptive" if DEFAULT_CARRY_POLICY == "uniform" else DEFAULT_CARRY_POLICY


def main() -> None:
    args = build_parser().parse_args()
    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    args_dict = dict(checkpoint.get("args", {}))
    eval_carry_policy = _resolve_eval_carry_policy(args, args_dict)
    experiment_tag = str(args.experiment_tag or args_dict.get("experiment_tag", "")).strip()
    output_path = _resolve_output_path(str(args.output_path), experiment_tag)
    prepared_manifest = Path(args.prepared_corpus_manifest) if args.prepared_corpus_manifest else None
    corpus_paths = _load_corpus_paths(checkpoint, prepared_manifest)
    checkpoint_dir = Path(str(args_dict.get("checkpoint_dir", checkpoint_path.parent)))
    metrics_path = checkpoint_dir / "events.jsonl"
    tokenizer = load_text_tokenizer(corpus_paths["tokenizer_path"])
    stage_corpora = {
        "foundation": load_stage_corpus(corpus_paths["foundation_path"]),
        "structured": load_stage_corpus(corpus_paths["structured_path"]),
        "long_context": load_stage_corpus(corpus_paths["long_context_path"]),
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _build_model(checkpoint, device)
    stages = list(checkpoint.get("stages", []))
    total_steps = sum(int(stage["steps"]) for stage in stages)
    global_step = int(checkpoint.get("global_step", 0))
    model.set_runtime_gates(staged_runtime_gates(step_index=max(global_step - 1, 0), total_steps=max(total_steps, 1))[1])

    random_gen = random.Random(23)
    if "random_gen_state" in checkpoint:
        random_gen.setstate(checkpoint["random_gen_state"])

    sample_batch = sample_stage_batch(
        stage_corpora,
        tokenizer,
        stage_name="long_context",
        batch_size=1,
        seq_len=int(next((stage["seq_len"] for stage in stages if str(stage["name"]) == "long_context"), 64)),
        device=device,
        rng=random_gen,
    )
    with torch.no_grad():
        sample_forward = model(sample_batch.input_ids)
        predictions = sample_forward["logits"].argmax(dim=-1)
        sample_loss = next_token_loss(sample_forward["logits"], sample_batch.target_ids, focus_mask=sample_batch.focus_mask)

    eval_reports = list(checkpoint.get("eval_reports", []))
    if args.refresh_eval or (args.eval_max_samples > 0 and not eval_reports):
        eval_report = evaluate_longbench_local(
            model,
            tokenizer,
            device=device,
            tasks=[item.strip() for item in str(args.eval_tasks).split(",") if item.strip()],
            max_samples=int(args.eval_max_samples or 2),
            carry_memory=not args.no_carry_memory,
            carry_policy=eval_carry_policy,
        )
        aggregate = dict(eval_report["aggregate"])
        eval_reports.append(
            {
                "step": global_step,
                "scheduler_stage": staged_runtime_gates(
                    step_index=max(global_step - 1, 0),
                    total_steps=max(total_steps, 1),
                )[0],
                "carry_policy": str(aggregate.get("carry_policy", eval_carry_policy)),
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
        )

    stage_reports = list(checkpoint.get("stage_reports", []))
    best_eval = dict(checkpoint.get("best_eval", {}))
    if not best_eval and eval_reports:
        best_eval = dict(eval_reports[-1].get("aggregate", {}))
        best_eval["step"] = int(eval_reports[-1].get("step", global_step))
    stage_reports, eval_reports, best_eval = _merge_events_into_reports(
        stage_reports=stage_reports,
        eval_reports=eval_reports,
        best_eval=best_eval,
        metrics_path=metrics_path,
    )

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment_tag": experiment_tag,
        "runtime": runtime_device_report(device),
        "resume_from": str(checkpoint_path.resolve()),
        "corpus": summarize_stage_corpus(corpus_paths),
        "model_config": dict(args_dict.get("model_config", {})),
        "tokenizer_vocab_size": tokenizer.vocab_size,
        "longbench_answer_supervision": bool(args_dict.get("longbench_answer_supervision", False)),
        "longbench_answer_ratio": float(args_dict.get("longbench_answer_ratio", 0.0)),
        "eval_carry_policy": eval_carry_policy,
        "carry_memory_stages": sorted(
            item.strip() for item in str(args_dict.get("carry_memory_stages", "")).split(",") if item.strip()
        ),
        "carry_memory_reset_every": int(args_dict.get("carry_memory_reset_every", 0)),
        "training_plan": stages,
        "completed_steps": global_step,
        "target_total_steps": total_steps,
        "stopped_early": global_step < total_steps,
        "checkpoint_dir": str(checkpoint_dir.resolve()),
        "metrics_path": str(metrics_path.resolve()),
        "best_checkpoint_path": str((checkpoint_path.parent / "best.pt").resolve()) if (checkpoint_path.parent / "best.pt").exists() else "",
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
            "schema_chain": _schema_chain_summary_from_aux(sample_forward.get("aux")),
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True, indent=2))
    print(f"report_path={output_path}")


if __name__ == "__main__":
    main()
