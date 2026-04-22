from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List


OUTPUT_PATH = Path("data/processed/experiments/long_context_ratio_sweep.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a small ratio sweep for SB-Core long-context QA supervision.")
    parser.add_argument("--ratios", default="0,0.5,1.0")
    parser.add_argument("--base-checkpoint-dir", default="data/processed/checkpoints/ratio_sweep")
    parser.add_argument("--wikipedia-limit", type=int, default=120)
    parser.add_argument("--clue-limit-per-subset", type=int, default=64)
    parser.add_argument("--longbench-limit-per-task", type=int, default=12)
    parser.add_argument("--foundation-steps", type=int, default=4)
    parser.add_argument("--structured-steps", type=int, default=3)
    parser.add_argument("--long-steps", type=int, default=3)
    parser.add_argument("--d-model", type=int, default=40)
    parser.add_argument("--state-dim", type=int, default=40)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--semantic-slots", type=int, default=8)
    parser.add_argument("--working-slots", type=int, default=8)
    parser.add_argument("--router-top-k", type=int, default=2)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--eval-max-samples", type=int, default=1)
    parser.add_argument("--train-log-every", type=int, default=5)
    return parser


def _ratio_label(ratio: float) -> str:
    return str(ratio).replace(".", "p")


def _run_ratio(args: argparse.Namespace, ratio: float) -> Dict[str, object]:
    checkpoint_dir = Path(args.base_checkpoint_dir) / f"ratio_{_ratio_label(ratio)}"
    command = [
        sys.executable,
        "-m",
        "examples.v02_text_curriculum_train",
        "--checkpoint-dir",
        str(checkpoint_dir),
        "--tokenizer-kind",
        "subword",
        "--longbench-answer-ratio",
        str(ratio),
        "--wikipedia-limit",
        str(args.wikipedia_limit),
        "--clue-limit-per-subset",
        str(args.clue_limit_per_subset),
        "--longbench-limit-per-task",
        str(args.longbench_limit_per_task),
        "--foundation-steps",
        str(args.foundation_steps),
        "--structured-steps",
        str(args.structured_steps),
        "--long-steps",
        str(args.long_steps),
        "--d-model",
        str(args.d_model),
        "--state-dim",
        str(args.state_dim),
        "--num-layers",
        str(args.num_layers),
        "--semantic-slots",
        str(args.semantic_slots),
        "--working-slots",
        str(args.working_slots),
        "--router-top-k",
        str(args.router_top_k),
        "--checkpoint-every",
        str(args.checkpoint_every),
        "--eval-every",
        str(args.eval_every),
        "--eval-max-samples",
        str(args.eval_max_samples),
        "--train-log-every",
        str(args.train_log_every),
    ]
    subprocess.run(command, check=True, cwd=Path.cwd())

    report_path = Path("data/processed/experiments/text_curriculum_train.json")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    best_eval = dict(report.get("best_eval", {}))
    latest_eval = report.get("eval_reports", [])[-1] if report.get("eval_reports") else {}
    return {
        "ratio": ratio,
        "checkpoint_dir": str(checkpoint_dir.resolve()),
        "best_checkpoint_path": report.get("best_checkpoint_path", ""),
        "best_eval": best_eval,
        "latest_eval": latest_eval,
        "completed_steps": report.get("completed_steps", 0),
    }


def main() -> None:
    args = build_parser().parse_args()
    ratios = [float(item.strip()) for item in args.ratios.split(",") if item.strip()]

    results: List[Dict[str, object]] = []
    for ratio in ratios:
        results.append(_run_ratio(args, ratio))

    sorted_results = sorted(
        results,
        key=lambda item: float(item.get("best_eval", {}).get("selection_score", float("-inf"))),
        reverse=True,
    )
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "ratios": ratios,
        "results": results,
        "best_ratio": sorted_results[0]["ratio"] if sorted_results else None,
        "best_result": sorted_results[0] if sorted_results else {},
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
