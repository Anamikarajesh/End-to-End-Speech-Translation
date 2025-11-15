#!/usr/bin/env python3
"""Generate Fairseq S2T manifests for IWSLT mr-hi data splits.

The dataset layout assumed by this script is the one distributed for
IWSLT 2023 mr-hi, where each split (``train``/``dev``/``test``) contains a
``stamped.tsv`` with segment metadata, a ``wav/`` directory with the
waveforms, and a ``txt/<split>.hi`` file with Hindi target text.

The output manifest follows Fairseq's speech-to-text TSV schema with the
columns ``id``, ``audio``, ``n_frames``, and ``text``.
"""
from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
from typing import List, Tuple

import soundfile as sf


logger = logging.getLogger(__name__)

STAMPED_FILENAME = "stamped.tsv"
TXT_SUBDIR = "txt"
AUDIO_SUBDIR = "wav"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Root directory of the iwslt2023_mr-hi dataset.",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "dev", "test"],
        required=True,
        help="Split to process.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination path for the generated manifest TSV.",
    )
    parser.add_argument(
        "--target-lang",
        type=str,
        default="hi",
        help="Language suffix of the translation text (e.g., 'hi' to read txt/<split>.hi).",
    )
    return parser.parse_args()


def read_stamped(split_dir: Path) -> List[str]:
    stamped_path = split_dir / STAMPED_FILENAME
    if not stamped_path.is_file():
        raise FileNotFoundError(f"Missing {stamped_path}")

    audio_rel_paths: List[str] = []
    with stamped_path.open("r", encoding="utf-8") as stamped_file:
        for line_no, line in enumerate(stamped_file, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 1:
                raise ValueError(f"Malformed stamped.tsv line {line_no}: {line!r}")
            audio_rel_paths.append(parts[0])
    logger.info("Loaded %d audio entries from %s", len(audio_rel_paths), stamped_path)
    return audio_rel_paths


def read_transcripts(split_dir: Path, split: str, target_lang: str) -> List[str]:
    transcript_path = split_dir / TXT_SUBDIR / f"{split}.{target_lang}"
    if not transcript_path.is_file():
        raise FileNotFoundError(f"Missing transcript file {transcript_path}")

    with transcript_path.open("r", encoding="utf-8") as txt_file:
        transcripts = [line.rstrip("\n") for line in txt_file]
    logger.info("Loaded %d transcripts from %s", len(transcripts), transcript_path)
    return transcripts


def collect_metadata(
    split_dir: Path, audio_rel_paths: List[str]
) -> List[Tuple[str, Path, int]]:
    metadata: List[Tuple[str, Path, int]] = []
    for idx, rel_path in enumerate(audio_rel_paths):
        rel_path = rel_path.strip()
        if not rel_path:
            raise ValueError(f"Empty audio path at index {idx}")

        audio_path = (split_dir / rel_path).resolve()
        if not audio_path.is_file():
            raise FileNotFoundError(f"Missing audio file: {audio_path}")

        try:
            info = sf.info(audio_path)
        except Exception as exc:  # pragma: no cover - sanity safeguard
            raise RuntimeError(f"Failed to read metadata for {audio_path}") from exc

        sample_id = Path(rel_path).with_suffix("").name
        metadata.append((sample_id, audio_path, info.frames))
    return metadata


def write_manifest(
    metadata: List[Tuple[str, Path, int]], transcripts: List[str], output_path: Path
) -> None:
    if len(metadata) != len(transcripts):
        raise ValueError(
            "Mismatch between number of audio files (%d) and transcripts (%d)"
            % (len(metadata), len(transcripts))
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as out_tsv:
        writer = csv.writer(out_tsv, delimiter="\t")
        writer.writerow(["id", "audio", "n_frames", "tgt_text"])
        for (sample_id, audio_path, n_frames), text in zip(metadata, transcripts):
            writer.writerow([sample_id, str(audio_path), n_frames, text.strip()])

    logger.info("Wrote manifest with %d rows to %s", len(metadata), output_path)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    split_dir = args.dataset_root / args.split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Split directory does not exist: {split_dir}")

    audio_rel_paths = read_stamped(split_dir)
    transcripts = read_transcripts(split_dir, args.split, args.target_lang)
    metadata = collect_metadata(split_dir, audio_rel_paths)
    write_manifest(metadata, transcripts, args.output)


if __name__ == "__main__":
    main()
