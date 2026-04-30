from __future__ import annotations

import argparse
import io
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List

from huggingface_hub import HfApi, hf_hub_download


DATA_ROOT = Path("data")
CACHE_ROOT = DATA_ROOT / "cache" / "huggingface"
DEFAULT_REPO_ID = "shjwudp/chinese-c4"
DEFAULT_OUTPUT_DIR = DATA_ROOT / "raw" / "chinese_c4"
DEFAULT_MANIFEST_PATH = DATA_ROOT / "manifest.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download a reproducible Chinese-C4 sample for SB-Core training.")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--manifest-path", default=str(DEFAULT_MANIFEST_PATH))
    parser.add_argument("--max-rows", type=int, default=5000)
    parser.add_argument("--min-chars", type=int, default=80)
    parser.add_argument("--max-shards", type=int, default=2)
    parser.add_argument("--split-name", default="train")
    parser.add_argument("--dataset-name", default="chinese_c4_sample")
    parser.add_argument("--no-update-manifest", action="store_true")
    return parser


def _require_zstandard():
    try:
        import zstandard as zstd
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Chinese-C4 使用 .zst 压缩分片。请先运行：python -m pip install zstandard"
        ) from exc
    return zstd


def _repo_shards(repo_id: str) -> List[str]:
    info = HfApi().dataset_info(repo_id)
    shards = [
        item.rfilename
        for item in (info.siblings or [])
        if item.rfilename.startswith("data/") and item.rfilename.endswith(".jsonl.zst")
    ]
    return sorted(shards)


def _iter_zst_jsonl(path: Path) -> Iterable[Dict[str, object]]:
    zstd = _require_zstandard()
    with path.open("rb") as raw:
        reader = zstd.ZstdDecompressor().stream_reader(raw)
        with io.TextIOWrapper(reader, encoding="utf-8") as text_stream:
            for line in text_stream:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)


def _text_from_row(row: Dict[str, object]) -> str:
    for key in ("text", "content", "正文"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return " ".join(value.split())
    return ""


def _write_rows(
    *,
    repo_id: str,
    shards: List[str],
    output_path: Path,
    max_rows: int,
    min_chars: int,
) -> Dict[str, object]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    row_count = 0
    scanned_rows = 0
    used_shards: List[Dict[str, object]] = []

    with output_path.open("w", encoding="utf-8") as handle:
        for shard_name in shards:
            local_path = Path(
                hf_hub_download(
                    repo_id=repo_id,
                    filename=shard_name,
                    repo_type="dataset",
                    cache_dir=str(CACHE_ROOT),
                )
            )
            shard_written = 0
            for row in _iter_zst_jsonl(local_path):
                scanned_rows += 1
                text = _text_from_row(row)
                if len(text) < min_chars:
                    continue
                record = {
                    "id": f"chinese-c4-{row_count:08d}",
                    "text": text,
                    "url": row.get("url", ""),
                    "timestamp": row.get("timestamp", ""),
                    "source_shard": shard_name,
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                row_count += 1
                shard_written += 1
                if row_count >= max_rows:
                    break
            used_shards.append(
                {
                    "name": shard_name,
                    "local_path": str(local_path.resolve()),
                    "written_rows": shard_written,
                }
            )
            if row_count >= max_rows:
                break

    return {
        "rows": row_count,
        "scanned_rows": scanned_rows,
        "size_mb": round(output_path.stat().st_size / (1024 * 1024), 2) if output_path.exists() else 0.0,
        "used_shards": used_shards,
    }


def _load_manifest(path: Path) -> Dict[str, object]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "root": str(DATA_ROOT.resolve()),
        "datasets": [],
    }


def _update_manifest(path: Path, dataset_entry: Dict[str, object]) -> None:
    manifest = _load_manifest(path)
    datasets = [item for item in manifest.get("datasets", []) if item.get("name") != dataset_entry["name"]]
    datasets.append(dataset_entry)
    manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
    manifest["root"] = str(DATA_ROOT.resolve())
    manifest["datasets"] = datasets
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    if args.max_rows <= 0:
        raise ValueError("max_rows 必须大于 0。")
    if args.max_shards <= 0:
        raise ValueError("max_shards 必须大于 0。")
    _require_zstandard()

    output_dir = Path(args.output_dir)
    output_path = output_dir / f"{args.dataset_name}_{args.max_rows}.jsonl"
    shards = _repo_shards(args.repo_id)
    if not shards:
        raise RuntimeError(f"未找到 Chinese-C4 .jsonl.zst 分片：{args.repo_id}")
    selected_shards = shards[: int(args.max_shards)]
    stats = _write_rows(
        repo_id=args.repo_id,
        shards=selected_shards,
        output_path=output_path,
        max_rows=int(args.max_rows),
        min_chars=int(args.min_chars),
    )
    dataset_entry = {
        "name": args.dataset_name,
        "repo_id": args.repo_id,
        "split": args.split_name,
        "path": str(output_path.resolve()),
        "rows": int(stats["rows"]),
        "scanned_rows": int(stats["scanned_rows"]),
        "size_mb": float(stats["size_mb"]),
        "min_chars": int(args.min_chars),
        "selected_shards": selected_shards,
        "used_shards": stats["used_shards"],
        "source": f"https://huggingface.co/datasets/{args.repo_id}",
    }
    if not args.no_update_manifest:
        _update_manifest(Path(args.manifest_path), dataset_entry)
    print(
        json.dumps(
            {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "manifest_updated": not args.no_update_manifest,
                "manifest_path": str(Path(args.manifest_path).resolve()),
                "dataset": dataset_entry,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
