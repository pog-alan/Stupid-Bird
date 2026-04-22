from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

from sb.core_lm_torch import SBCoreMiniLM, SBCoreMiniTorchConfig, SBRuntimeGates, runtime_device_report
from sb.longbench_local_eval import DEFAULT_TASKS, evaluate_longbench_local
from sb.text_corpus import load_text_tokenizer


DEFAULT_CHECKPOINT = Path("data/processed/checkpoints/text_curriculum/last.pt")
OUTPUT_PATH = Path("data/processed/experiments/longbench_matrix_eval.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a systematic carry/no-carry x prompt-limit LongBench matrix.")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--tasks", default=",".join(DEFAULT_TASKS))
    parser.add_argument("--prompt-limits", default="256,512")
    parser.add_argument("--carry-policies", default="uniform,task_adaptive")
    parser.add_argument("--max-samples", type=int, default=1)
    parser.add_argument("--experiment-tag", default="")
    parser.add_argument("--output-path", default="")
    return parser


def resolve_output_path(path_str: str, experiment_tag: str) -> Path:
    if path_str:
        path = Path(path_str)
    elif experiment_tag:
        path = OUTPUT_PATH.with_name(f"{OUTPUT_PATH.stem}_{experiment_tag}{OUTPUT_PATH.suffix}")
    else:
        path = OUTPUT_PATH
    if not path.is_absolute():
        path = Path.cwd() / path
    return path


def parse_prompt_limits(value: str) -> List[int]:
    limits: List[int] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        limits.append(max(int(item), 0))
    if not limits:
        limits = [256, 512]
    return limits


def parse_carry_policies(value: str) -> List[str]:
    policies: List[str] = []
    for item in value.split(","):
        item = item.strip()
        if item:
            policies.append(item)
    return policies or ["uniform", "task_adaptive"]


def load_model_and_tokenizer(checkpoint_path: Path, device: str) -> Tuple[SBCoreMiniLM, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = SBCoreMiniTorchConfig(**checkpoint["args"]["model_config"])
    tokenizer = load_text_tokenizer(checkpoint["corpus_paths"]["tokenizer_path"])
    model = SBCoreMiniLM(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.set_runtime_gates(SBRuntimeGates())
    model.eval()
    return model, tokenizer


def _selection_score(report: Dict[str, Any]) -> float:
    return float(report.get("selection_score", 0.0))


def main() -> None:
    args = build_parser().parse_args()
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    task_names = [item.strip() for item in args.tasks.split(",") if item.strip()]
    prompt_limits = parse_prompt_limits(args.prompt_limits)
    carry_policies = parse_carry_policies(args.carry_policies)
    output_path = resolve_output_path(args.output_path, args.experiment_tag)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, tokenizer = load_model_and_tokenizer(checkpoint_path, device)

    rows: List[Dict[str, Any]] = []
    by_key: Dict[Tuple[str, int, bool, str], Dict[str, Any]] = {}
    for task_name in task_names:
        for prompt_limit in prompt_limits:
            eval_report = evaluate_longbench_local(
                model,
                tokenizer,
                device=device,
                tasks=[task_name],
                max_samples=args.max_samples,
                carry_memory=False,
                prompt_char_limit=prompt_limit,
                carry_policy="uniform",
            )
            aggregate = dict(eval_report["aggregate"])
            task_report = dict(eval_report["tasks"][0]) if eval_report["tasks"] else {}
            baseline_row = {
                "task": task_name,
                "prompt_char_limit": int(prompt_limit),
                "carry_memory": False,
                "carry_policy": "uniform",
                "selection_score": _selection_score(aggregate),
                "aggregate": aggregate,
                "task_report": task_report,
            }
            rows.append(baseline_row)
            by_key[(task_name, int(prompt_limit), False, "uniform")] = baseline_row

            for carry_policy in carry_policies:
                eval_report = evaluate_longbench_local(
                    model,
                    tokenizer,
                    device=device,
                    tasks=[task_name],
                    max_samples=args.max_samples,
                    carry_memory=True,
                    prompt_char_limit=prompt_limit,
                    carry_policy=carry_policy,
                )
                aggregate = dict(eval_report["aggregate"])
                task_report = dict(eval_report["tasks"][0]) if eval_report["tasks"] else {}
                row = {
                    "task": task_name,
                    "prompt_char_limit": int(prompt_limit),
                    "carry_memory": True,
                    "carry_policy": carry_policy,
                    "selection_score": _selection_score(aggregate),
                    "aggregate": aggregate,
                    "task_report": task_report,
                }
                rows.append(row)
                by_key[(task_name, int(prompt_limit), True, carry_policy)] = row

    deltas: List[Dict[str, Any]] = []
    best_carry_by_task: List[Dict[str, Any]] = []
    for task_name in task_names:
        carry_candidates: List[Dict[str, Any]] = []
        for prompt_limit in prompt_limits:
            no_carry = by_key.get((task_name, int(prompt_limit), False, "uniform"))
            if no_carry is None:
                continue
            no_carry_task = no_carry["task_report"]
            for carry_policy in carry_policies:
                carry = by_key.get((task_name, int(prompt_limit), True, carry_policy))
                if carry is None:
                    continue
                carry_candidates.append(carry)
                carry_task = carry["task_report"]
                deltas.append(
                    {
                        "task": task_name,
                        "prompt_char_limit": int(prompt_limit),
                        "carry_policy": carry_policy,
                        "carry_gain_selection_score": float(carry["selection_score"]) - float(no_carry["selection_score"]),
                        "carry_gain_answer_loss": float(carry_task.get("mean_answer_loss", 0.0))
                        - float(no_carry_task.get("mean_answer_loss", 0.0)),
                        "carry_gain_token_acc": float(carry_task.get("mean_answer_token_acc", 0.0))
                        - float(no_carry_task.get("mean_answer_token_acc", 0.0)),
                        "carry_gain_prompt_retained_ratio": float(carry_task.get("mean_prompt_retained_ratio", 0.0))
                        - float(no_carry_task.get("mean_prompt_retained_ratio", 0.0)),
                        "carry_gain_episodic_schema": float(
                            carry_task.get("mean_episodic_replay_schema_alignment", 0.0)
                        )
                        - float(no_carry_task.get("mean_episodic_replay_schema_alignment", 0.0)),
                        "carry_gain_summary_schema": float(carry_task.get("mean_summary_schema_alignment", 0.0))
                        - float(no_carry_task.get("mean_summary_schema_alignment", 0.0)),
                        "carry_gain_scene_schema": float(carry_task.get("mean_scene_schema_alignment", 0.0))
                        - float(no_carry_task.get("mean_scene_schema_alignment", 0.0)),
                    }
                )
        if carry_candidates:
            best = max(carry_candidates, key=lambda item: float(item["selection_score"]))
            best_carry_by_task.append(
                {
                    "task": task_name,
                    "best_prompt_char_limit": int(best["prompt_char_limit"]),
                    "best_carry_policy": str(best["carry_policy"]),
                    "best_selection_score": float(best["selection_score"]),
                    "best_answer_loss": float(best["task_report"].get("mean_answer_loss", 0.0)),
                    "best_summary_schema_alignment": float(
                        best["task_report"].get("mean_summary_schema_alignment", 0.0)
                    ),
                    "best_scene_schema_alignment": float(
                        best["task_report"].get("mean_scene_schema_alignment", 0.0)
                    ),
                }
            )

    report: Dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment_tag": args.experiment_tag,
        "runtime": runtime_device_report(device),
        "checkpoint": str(checkpoint_path.resolve()),
        "tasks": task_names,
        "prompt_limits": prompt_limits,
        "carry_policies": carry_policies,
        "max_samples": int(args.max_samples),
        "rows": rows,
        "carry_deltas": deltas,
        "best_carry_by_task": best_carry_by_task,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True, indent=2))
    print(f"report_path={output_path}")


if __name__ == "__main__":
    main()
