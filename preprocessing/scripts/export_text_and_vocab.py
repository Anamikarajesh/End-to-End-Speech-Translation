#!/usr/bin/env python3
"""Export transcripts from manifests for downstream tokenizer work.

The script collects the ``text`` column from train and validation
manifests into plain-text files that can be consumed by SentencePiece
or other subword trainers. The earlier word-level dictionary export has
been removed because the speech-to-text pipeline relies on a shared
SentencePiece model instead.
"""
import argparse
import csv
from pathlib import Path
from typing import Iterable


def _iter_text(tsv_path: Path) -> Iterable[str]:
    with tsv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if "text" not in reader.fieldnames:
            raise ValueError(f"Manifest missing 'text' column: {tsv_path}")
        for row in reader:
            yield row["text"].strip()


def export_texts(train_manifest: Path, valid_manifest: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    train_txt = output_dir / "combined_train.txt"
    valid_txt = output_dir / "combined_valid.txt"

    with train_txt.open("w", encoding="utf-8") as train_out:
        for text in _iter_text(train_manifest):
            train_out.write(text + "\n")

    with valid_txt.open("w", encoding="utf-8") as valid_out:
        for text in _iter_text(valid_manifest):
            valid_out.write(text + "\n")

    print(f"Wrote transcripts to {train_txt} and {valid_txt}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-manifest",
        required=True,
        type=Path,
        help="Path to the training manifest TSV (with tgt_text column)",
    )
    parser.add_argument(
        "--valid-manifest",
        required=True,
        type=Path,
        help="Path to the validation manifest TSV (with tgt_text column)",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory for exported text files and dictionary",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    export_texts(args.train_manifest, args.valid_manifest, args.output_dir)


if __name__ == "__main__":
    main()
