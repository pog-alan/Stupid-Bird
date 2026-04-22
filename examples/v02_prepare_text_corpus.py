from __future__ import annotations

import argparse
import json
from pathlib import Path

from sb.text_corpus import prepare_local_text_corpus, summarize_stage_corpus


DEFAULT_PROFILE_PATH = Path("configs/sb_core_stage3_data_profile.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare a reproducible SB-Core text corpus from local raw datasets.")
    parser.add_argument("--profile-path", default=str(DEFAULT_PROFILE_PATH))
    parser.add_argument("--manifest-path", default="data/manifest.json")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    profile_path = Path(args.profile_path)
    if not profile_path.is_absolute():
        profile_path = Path.cwd() / profile_path
    profile = json.loads(profile_path.read_text(encoding="utf-8"))

    corpus_paths = prepare_local_text_corpus(
        args.manifest_path,
        output_dir=profile["output_dir"],
        wikipedia_limit=int(profile["wikipedia_limit"]),
        clue_limit_per_subset=int(profile["clue_limit_per_subset"]),
        longbench_limit_per_task=int(profile["longbench_limit_per_task"]),
        max_vocab_size=int(profile["max_vocab_size"]),
        min_freq=int(profile["min_freq"]),
        tokenizer_kind=str(profile["tokenizer_kind"]),
        validation_ratio=float(profile.get("validation_ratio", 0.05)),
        seed=int(profile.get("seed", 23)),
        profile_name=str(profile.get("profile_name", profile_path.stem)),
    )

    report = {
        "profile_path": str(profile_path.resolve()),
        "profile": profile,
        "corpus": summarize_stage_corpus(corpus_paths),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
