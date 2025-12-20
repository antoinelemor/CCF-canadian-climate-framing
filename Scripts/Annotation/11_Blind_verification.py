"""
PROJECT:
--------
CCF-Canadian-Climate-Framing

TITLE:
------
11_Blind_verification.py

MAIN OBJECTIVE:
---------------
Create a “blind-verification” copy of the manual-annotation file so that
sentences can be re-labelled without bias.  

MAIN FEATURES:
--------------
1) Robust, line-by-line streaming read/write → constant memory usage.
2) Automatic creation of the output directory if it does not exist.
3) Optional custom input/output paths via CLI arguments.
4) Informative progress indicator (tqdm or simple counter).

Author:
-------
[Anonymous]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterator

# ════════════════════════════════════════════════════════════════════════
# 1. Helpers
# ════════════════════════════════════════════════════════════════════════
def stream_jsonl(path: Path) -> Iterator[dict]:
    """Yield JSON objects one by one from a .jsonl file."""
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue  # skip blank lines
            try:
                yield json.loads(line)
            except json.JSONDecodeError as err:
                msg = f"[ERROR] Invalid JSON at line {line_no} of {path}: {err}"
                raise ValueError(msg) from err


def dump_jsonl(objs: Iterator[dict], path: Path, total: int | None = None) -> None:
    """Write an iterator of JSON objects to a .jsonl file."""
    try:
        from tqdm.auto import tqdm  # pylint: disable=import-error
        iterator = tqdm(objs, total=total, unit="sent", colour="green")
    except ModuleNotFoundError:  # tqdm not installed
        iterator = objs

    with path.open("w", encoding="utf-8") as outfile:
        for idx, obj in enumerate(iterator, start=1):
            json_str = json.dumps(obj, ensure_ascii=False)
            outfile.write(json_str + "\n")

            # Show a minimal progress message if tqdm is unavailable
            if "tqdm" not in sys.modules and idx % 10_000 == 0:
                print(f"[INFO] Processed {idx:,} sentences…", file=sys.stderr, flush=True)


# ════════════════════════════════════════════════════════════════════════
# 2. Core logic
# ════════════════════════════════════════════════════════════════════════
def create_blind_file(src: Path, dst: Path) -> None:
    """Generate a blind-verification JSONL with empty labels."""
    if not src.exists():
        raise FileNotFoundError(f"Input file not found: {src}")

    # Make sure the destination directory exists
    dst.parent.mkdir(parents=True, exist_ok=True)

    # First pass to count lines for the progress bar (cheap: read as bytes)
    total_lines = sum(1 for _ in src.open("rb"))

    # Second pass: stream-process, wipe labels, write out
    def gen_clean_objs() -> Iterator[dict]:
        for obj in stream_jsonl(src):
            if "label" in obj:
                obj["label"] = []
            yield obj

    dump_jsonl(gen_clean_objs(), dst, total=total_lines)
    print(f"[SUCCESS] Blind file written to {dst}.")


# ════════════════════════════════════════════════════════════════════════
# 3. CLI
# ════════════════════════════════════════════════════════════════════════
def parse_args() -> argparse.Namespace:
    default_dir = (
        Path(__file__).resolve()
        .parents[2]
        / "Database/Training_data/manual_annotations_JSONL"
    )

    parser = argparse.ArgumentParser(
        description="Create a blank-label copy of the manual-annotation JSONL."
    )
    parser.add_argument(
        "-i", "--input",
        type=Path,
        default=default_dir / "all.jsonl",
        help=f"Path to the source all.jsonl file (default: {default_dir/'all.jsonl'})"
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=default_dir / "blind_verification.jsonl",
        help=(
            "Path to the output blind_verification.jsonl "
            f"(default: {default_dir/'blind_verification.jsonl'})"
        )
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        create_blind_file(args.input, args.output)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        print(f"[FAIL] {exc}", file=sys.stderr)
        sys.exit(1)
