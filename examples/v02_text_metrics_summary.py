from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize SB-Core text curriculum JSONL event logs.")
    parser.add_argument("--events", required=True)
    return parser


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    args = build_parser().parse_args()
    rows = _read_jsonl(Path(args.events))

    train_rows = [row for row in rows if row.get("type") == "train_step"]
    eval_rows = [row for row in rows if row.get("type") == "eval"]
    best_rows = [row for row in rows if row.get("type") == "best_eval"]
    stage_rows = [row for row in rows if row.get("type") == "stage_complete"]

    report = {
        "events_path": str(Path(args.events).resolve()),
        "counts": {
            "train_step": len(train_rows),
            "eval": len(eval_rows),
            "best_eval": len(best_rows),
            "stage_complete": len(stage_rows),
        },
        "latest_train": train_rows[-1] if train_rows else {},
        "latest_eval": eval_rows[-1] if eval_rows else {},
        "best_eval": best_rows[-1] if best_rows else {},
        "stage_sequence": [row.get("stage", "") for row in stage_rows],
        "loss_trend": {
            "first_train_loss": float(train_rows[0]["loss"]) if train_rows else 0.0,
            "last_train_loss": float(train_rows[-1]["loss"]) if train_rows else 0.0,
            "min_train_loss": min(float(row["loss"]) for row in train_rows) if train_rows else 0.0,
        },
    }

    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
