from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

from examples.v02_acmm_chinese_c4_eval import build_model
from sb.acmm_text import build_text_observation, find_manifest_dataset_path, iter_chinese_c4_texts


LABELS = {
    "0": "普通文本",
    "1": "弱风险线索",
    "2": "高风险/需复核",
    "u": "不确定",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_label_candidates(
    *,
    dataset_path: Path,
    limit: int,
    min_chars: int,
    selection: str,
    seed: int,
) -> List[Dict[str, object]]:
    model = build_model()
    rows = list(iter_chinese_c4_texts(dataset_path, limit=limit, min_chars=min_chars))
    candidates: List[Dict[str, object]] = []
    for row in rows:
        text_observation = build_text_observation(row)
        result = model.cognitive_step(text_observation.observation)
        review_score = max(result.gates.request_review, result.gates.trigger_alert, result.gates.update_rule)
        candidates.append(
            {
                "row_id": text_observation.row_id,
                "text": text_observation.text,
                "weak_label": text_observation.weak_label,
                "weak_high_risk": text_observation.weak_high_risk,
                "matched_rules": list(text_observation.matched_rules),
                "observation": text_observation.observation,
                "acmm": {
                    "review_score": review_score,
                    "emotion": result.emotion.as_dict(),
                    "gates": result.gates.as_dict(),
                    "memory_writes": [item.memory_type for item in result.memory_writes],
                },
                "human_label": "",
                "human_high_risk": None,
                "annotator": "",
                "notes": "",
                "created_at": _now(),
                "updated_at": "",
            }
        )
    return _select_candidates(candidates, selection=selection, seed=seed)


def load_existing(path: Path) -> Dict[str, Dict[str, object]]:
    if not path.exists():
        return {}
    records: Dict[str, Dict[str, object]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            row_id = str(item.get("row_id", ""))
            if row_id:
                records[row_id] = item
    return records


def merge_existing(
    candidates: Sequence[Mapping[str, object]],
    existing: Mapping[str, Mapping[str, object]],
) -> List[Dict[str, object]]:
    merged: List[Dict[str, object]] = []
    seen = set()
    for candidate in candidates:
        row_id = str(candidate.get("row_id", ""))
        seen.add(row_id)
        if row_id in existing:
            kept = dict(candidate)
            old = dict(existing[row_id])
            for key in ("human_label", "human_high_risk", "annotator", "notes", "created_at", "updated_at"):
                if key in old:
                    kept[key] = old[key]
            merged.append(kept)
        else:
            merged.append(dict(candidate))
    for row_id, item in existing.items():
        if row_id not in seen:
            merged.append(dict(item))
    return merged


def write_jsonl(path: Path, records: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for item in records:
            handle.write(json.dumps(dict(item), ensure_ascii=False) + "\n")


def interactive_label(
    records: List[Dict[str, object]],
    *,
    output_path: Path,
    annotator: str,
    max_preview_chars: int,
) -> None:
    print("标签：0=普通文本, 1=弱风险线索, 2=高风险/需复核, u=不确定, s=跳过, q=退出")
    for index, record in enumerate(records, start=1):
        if record.get("human_label"):
            continue
        print("\n" + "=" * 80)
        print(f"[{index}/{len(records)}] row_id={record.get('row_id')}")
        print(f"weak_label={record.get('weak_label')} weak_high_risk={record.get('weak_high_risk')}")
        print(f"matched_rules={record.get('matched_rules')}")
        acmm = record.get("acmm", {})
        if isinstance(acmm, Mapping):
            print(f"acmm_review_score={acmm.get('review_score')}")
        print("-" * 80)
        print(str(record.get("text", ""))[:max_preview_chars])
        choice = input("请选择标签: ").strip().lower()
        if choice == "q":
            break
        if choice == "s" or not choice:
            continue
        if choice not in LABELS:
            print("无效输入，已跳过。")
            continue
        label = LABELS[choice]
        record["human_label"] = label
        record["human_high_risk"] = label in {"弱风险线索", "高风险/需复核"}
        record["annotator"] = annotator
        record["updated_at"] = _now()
        note = input("备注，可留空: ").strip()
        if note:
            record["notes"] = note
        write_jsonl(output_path, records)
        print("已保存。")


def _select_candidates(candidates: List[Dict[str, object]], *, selection: str, seed: int) -> List[Dict[str, object]]:
    if selection == "sequential":
        return candidates
    rng = random.Random(seed)
    if selection == "random":
        shuffled = list(candidates)
        rng.shuffle(shuffled)
        return shuffled
    ranked = sorted(
        candidates,
        key=lambda item: float(_nested_get(item, ("acmm", "review_score"), 0.0)),
        reverse=True,
    )
    if selection == "acmm":
        return ranked
    if selection == "mixed":
        half = max(1, len(candidates) // 2)
        top = ranked[:half]
        top_ids = {item["row_id"] for item in top}
        rest = [item for item in candidates if item["row_id"] not in top_ids]
        rng.shuffle(rest)
        return top + rest
    raise ValueError(f"未知 selection：{selection}")


def _nested_get(source: Mapping[str, object], path: Sequence[str], default: object) -> object:
    value: object = source
    for key in path:
        if not isinstance(value, Mapping):
            return default
        value = value.get(key, default)
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create or interactively label Chinese-C4 ACMM review samples.")
    parser.add_argument("--manifest-path", default="data/manifest.json")
    parser.add_argument("--dataset-name", default="chinese_c4_sample")
    parser.add_argument("--dataset-path", default="")
    parser.add_argument("--output-path", type=Path, default=Path("data/processed/labels/acmm_chinese_c4_labels.jsonl"))
    parser.add_argument("--limit", type=int, default=120)
    parser.add_argument("--min-chars", type=int, default=80)
    parser.add_argument("--selection", choices=("mixed", "acmm", "random", "sequential"), default="mixed")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--annotator", default="manual")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--max-preview-chars", type=int, default=900)
    return parser


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = build_parser().parse_args()
    dataset_path = (
        Path(args.dataset_path)
        if args.dataset_path
        else find_manifest_dataset_path(args.manifest_path, dataset_name=args.dataset_name)
    )
    candidates = build_label_candidates(
        dataset_path=dataset_path,
        limit=args.limit,
        min_chars=args.min_chars,
        selection=args.selection,
        seed=args.seed,
    )
    records = merge_existing(candidates, load_existing(args.output_path))
    write_jsonl(args.output_path, records)
    labeled_count = sum(1 for item in records if item.get("human_label"))
    summary = {
        "dataset_path": str(dataset_path),
        "output_path": str(args.output_path),
        "records": len(records),
        "labeled": labeled_count,
        "unlabeled": len(records) - labeled_count,
        "selection": args.selection,
        "interactive": args.interactive,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.interactive:
        interactive_label(
            records,
            output_path=args.output_path,
            annotator=args.annotator,
            max_preview_chars=args.max_preview_chars,
        )


if __name__ == "__main__":
    main()
