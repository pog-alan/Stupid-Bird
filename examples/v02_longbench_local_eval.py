from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import torch

from sb.core_lm_torch import SBCoreMiniLM, SBCoreMiniTorchConfig, SBRuntimeGates, runtime_device_report
from sb.longbench_local_eval import DEFAULT_TASKS, evaluate_longbench_local
from sb.text_corpus import load_text_tokenizer


DEFAULT_CHECKPOINT = Path("data/processed/checkpoints/text_curriculum/last.pt")
OUTPUT_PATH = Path("data/processed/experiments/longbench_local_eval.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate SB-Core checkpoints on local LongBench continuation scoring.")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--tasks", default=",".join(DEFAULT_TASKS))
    parser.add_argument("--max-samples", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--no-carry-memory", action="store_true")
    parser.add_argument("--carry-policy", choices=["uniform", "task_adaptive"], default="uniform")
    parser.add_argument("--prompt-char-limit", type=int, default=0)
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


def main() -> None:
    args = build_parser().parse_args()
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = SBCoreMiniTorchConfig(**checkpoint["args"]["model_config"])
    tokenizer = load_text_tokenizer(checkpoint["corpus_paths"]["tokenizer_path"])
    output_path = resolve_output_path(args.output_path, args.experiment_tag)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SBCoreMiniLM(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.set_runtime_gates(SBRuntimeGates())
    model.eval()

    task_names = [item.strip() for item in args.tasks.split(",") if item.strip()]
    eval_report = evaluate_longbench_local(
        model,
        tokenizer,
        device=device,
        tasks=task_names,
        max_samples=args.max_samples,
        carry_memory=not bool(args.no_carry_memory),
        prompt_char_limit=args.prompt_char_limit,
        carry_policy=args.carry_policy,
    )
    report: Dict[str, object] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment_tag": args.experiment_tag,
        "runtime": runtime_device_report(device),
        "checkpoint": str(checkpoint_path.resolve()),
        **eval_report,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True, indent=2))
    print(f"report_path={output_path}")


if __name__ == "__main__":
    main()
