from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from sb.acmm_formal import acmm_formal_markdown, acmm_formal_spec_dict, build_acmm_formal_spec


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Export the formal mathematical object spec for ACMM.")
    parser.add_argument("--format", choices=("json", "markdown"), default="json")
    parser.add_argument("--output-path", type=Path, default=None)
    args = parser.parse_args()

    spec = build_acmm_formal_spec()
    spec.validate()
    if args.format == "markdown":
        text = acmm_formal_markdown()
    else:
        text = json.dumps(acmm_formal_spec_dict(), ensure_ascii=False, indent=2)
    if args.output_path is not None:
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        args.output_path.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
