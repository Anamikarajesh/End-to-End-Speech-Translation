#!/usr/bin/env python3
"""Merge multiple Fairseq-style TSV manifests into a single file.

The script keeps the header from the first input and skips subsequent
headers so that the combined output maintains a valid manifest format.
"""
import argparse
from pathlib import Path
from typing import Iterable, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Concatenate Fairseq TSV manifests while keeping a single header.",
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="Manifest paths to merge in the provided order.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Destination path for the merged manifest.",
    )
    return parser.parse_args()


def iter_manifest_lines(paths: Iterable[Path]) -> Iterable[str]:
    for idx, manifest in enumerate(paths):
        with manifest.open("r", encoding="utf-8") as src:
            for line_number, line in enumerate(src):
                if idx > 0 and line_number == 0 and line.strip().startswith("id"):
                    continue
                yield line


def merge_manifests(inputs: List[Path], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as dest:
        for line in iter_manifest_lines(inputs):
            dest.write(line)


def main() -> None:
    args = parse_args()
    missing = [str(path) for path in args.inputs if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing input manifests: " + ", ".join(missing))
    merge_manifests(args.inputs, args.output)


if __name__ == "__main__":
    main()
