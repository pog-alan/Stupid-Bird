from __future__ import annotations

import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import pyarrow.parquet as pq
from datasets import load_dataset
from huggingface_hub import hf_hub_download, snapshot_download


DATA_ROOT = Path("data")
RAW_ROOT = DATA_ROOT / "raw"
PROCESSED_ROOT = DATA_ROOT / "processed"
CACHE_ROOT = DATA_ROOT / "cache" / "huggingface"
MANIFEST_PATH = DATA_ROOT / "manifest.json"

CLUE_SUBSETS = ["afqmc", "tnews", "cmnli"]
LONGBENCH_TASKS = ["passage_retrieval_zh", "multifieldqa_zh", "dureader"]
WIKIPEDIA_SAMPLE_ROWS = 5000


def ensure_dirs() -> None:
    for path in [RAW_ROOT, PROCESSED_ROOT, CACHE_ROOT]:
        path.mkdir(parents=True, exist_ok=True)


def file_size_mb(path: Path) -> float:
    return round(path.stat().st_size / (1024 * 1024), 2)


def parquet_row_count(path: Path) -> int:
    return int(pq.ParquetFile(path).metadata.num_rows)


def download_clue() -> Dict[str, object]:
    target_root = RAW_ROOT / "clue"
    target_root.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id="clue",
        repo_type="dataset",
        local_dir=str(target_root),
        cache_dir=str(CACHE_ROOT),
        allow_patterns=[f"{subset}/*.parquet" for subset in CLUE_SUBSETS],
    )

    subsets: List[Dict[str, object]] = []
    for subset in CLUE_SUBSETS:
        subset_dir = target_root / subset
        split_info: Dict[str, Dict[str, object]] = {}
        for parquet_path in sorted(subset_dir.glob("*.parquet")):
            split_name = parquet_path.stem.split("-")[0]
            split_info[split_name] = {
                "path": str(parquet_path.resolve()),
                "rows": parquet_row_count(parquet_path),
                "size_mb": file_size_mb(parquet_path),
            }
        subsets.append({"name": subset, "splits": split_info})
    return {
        "name": "clue",
        "local_root": str(target_root.resolve()),
        "subsets": subsets,
        "source": "https://huggingface.co/datasets/clue",
    }


def download_longbench() -> Dict[str, object]:
    target_root = RAW_ROOT / "longbench"
    target_root.mkdir(parents=True, exist_ok=True)
    archive_path = Path(
        hf_hub_download(
            repo_id="THUDM/LongBench",
            filename="data.zip",
            repo_type="dataset",
            cache_dir=str(CACHE_ROOT),
        )
    )

    extracted: List[Dict[str, object]] = []
    with zipfile.ZipFile(archive_path) as zf:
        members = zf.namelist()
        for task in LONGBENCH_TASKS:
            matched = None
            for member in members:
                stem = Path(member).stem
                if stem == task and member.endswith((".jsonl", ".json")):
                    matched = member
                    break
            if matched is None:
                continue
            output_path = target_root / Path(matched).name
            if not output_path.exists():
                output_path.write_bytes(zf.read(matched))
            row_count = 0
            with output_path.open("r", encoding="utf-8") as handle:
                for row_count, _ in enumerate(handle, start=1):
                    pass
            extracted.append(
                {
                    "task": task,
                    "path": str(output_path.resolve()),
                    "rows": row_count,
                    "size_mb": file_size_mb(output_path),
                }
            )
    return {
        "name": "longbench",
        "local_root": str(target_root.resolve()),
        "tasks": extracted,
        "archive_path": str(archive_path.resolve()),
        "source": "https://huggingface.co/datasets/THUDM/LongBench",
    }


def sample_wikipedia_zh() -> Dict[str, object]:
    target_root = PROCESSED_ROOT / "wikipedia_zh_sample"
    target_root.mkdir(parents=True, exist_ok=True)
    output_path = target_root / f"train_{WIKIPEDIA_SAMPLE_ROWS}.jsonl"

    if not output_path.exists():
        dataset = load_dataset(
            "wikimedia/wikipedia",
            "20231101.zh",
            split="train",
            streaming=True,
        )
        with output_path.open("w", encoding="utf-8") as handle:
            for index, row in enumerate(dataset):
                record = {
                    "id": row.get("id"),
                    "url": row.get("url"),
                    "title": row.get("title"),
                    "text": row.get("text"),
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                if index + 1 >= WIKIPEDIA_SAMPLE_ROWS:
                    break

    row_count = 0
    with output_path.open("r", encoding="utf-8") as handle:
        for row_count, _ in enumerate(handle, start=1):
            pass
    return {
        "name": "wikipedia_zh_sample",
        "path": str(output_path.resolve()),
        "rows": row_count,
        "size_mb": file_size_mb(output_path),
        "source": "https://huggingface.co/datasets/wikimedia/wikipedia",
    }


def build_manifest() -> Dict[str, object]:
    ensure_dirs()
    clue_info = download_clue()
    longbench_info = download_longbench()
    wikipedia_info = sample_wikipedia_zh()
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "root": str(DATA_ROOT.resolve()),
        "datasets": [clue_info, longbench_info, wikipedia_info],
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    manifest = build_manifest()
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
