from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path


DEFAULT_OUTPUT_PATH = Path("data/processed/experiments/sb_visual_real_dataset_prepare.json")
REPO_ROOT = Path(__file__).resolve().parents[1]
VISUAL_DATA_PATH = REPO_ROOT / "sb" / "sb-visual" / "sb_visual_data.py"


def _load_module(module_name: str, path: Path) -> object:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare a real SB-Visual scene dataset from rural/urban image folders.")
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--destination-root", required=True)
    parser.add_argument("--scene-names", default="rural,urban")
    parser.add_argument("--source-annotation-manifest", default="")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--copy-files", action="store_true")
    parser.add_argument("--no-copy-files", action="store_true")
    parser.add_argument("--include-classification-qa", action="store_true")
    parser.add_argument("--no-classification-qa", action="store_true")
    parser.add_argument("--output-path", default="")
    return parser


def _resolve_output_path(path_str: str) -> Path:
    path = Path(path_str) if path_str else DEFAULT_OUTPUT_PATH
    if not path.is_absolute():
        path = Path.cwd() / path
    return path


def main() -> None:
    args = _build_parser().parse_args()
    visual_data = _load_module("sb_visual_data", VISUAL_DATA_PATH)
    copy_files = True
    if args.copy_files:
        copy_files = True
    if args.no_copy_files:
        copy_files = False
    include_classification_qa = True
    if args.no_classification_qa:
        include_classification_qa = False
    if args.include_classification_qa:
        include_classification_qa = True

    scene_names = tuple(item.strip() for item in args.scene_names.split(",") if item.strip())
    prepared = visual_data.prepare_real_scene_dataset(
        args.source_root,
        args.destination_root,
        scene_names=scene_names,
        source_annotation_manifest=args.source_annotation_manifest or None,
        train_ratio=float(args.train_ratio),
        seed=int(args.seed),
        copy_files=copy_files,
        include_classification_qa=include_classification_qa,
    )
    output_path = _resolve_output_path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(prepared, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"report_path={output_path.resolve()}")


if __name__ == "__main__":
    main()
