#!/usr/bin/env python3
"""Utility for preparing the Panlingua IWSLT2023 Marathi→Hindi speech translation corpus.

Steps performed per split (train/dev/test/test-2024):
 1. Optionally resample audio to the requested sample rate (default 16 kHz) and ensure mono channel.
 2. Mirror the dataset layout under the destination directory.
 3. Emit a UTF-8 JSONL manifest capturing absolute audio paths, durations, and Hindi translations when available.

Example usage:
    python scripts/prepare_iwslt_dataset.py \
        --source-root datasets/iwslt2023_mr-hi \
        --output-root processed_data/iwslt_16khz \
        --target-sr 16000

The script is idempotent: already-processed files are skipped unless --force is set.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import soundfile as sf
import soxr

# Default split ordering to keep manifests reproducible.
SPLITS_WITH_TEXT = ["train", "dev", "test"]
SPLITS_AUDIO_ONLY = ["test-2024"]


@dataclass
class Segment:
    """Metadata for an audio segment extracted from stamped.tsv."""

    rel_audio_path: Path
    start_sec: float
    duration_sec: float
    translation: Optional[str] = None

    @property
    def end_sec(self) -> float:
        return self.start_sec + self.duration_sec


def load_segments(split_dir: Path, split_name: str, has_transcripts: bool) -> List[Segment]:
    stamped_path = split_dir / "stamped.tsv"
    if not stamped_path.is_file():
        raise FileNotFoundError(f"Missing stamped.tsv for split '{split_name}' at {stamped_path}")

    with stamped_path.open("r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    translations: List[str] = []
    if has_transcripts:
        txt_path = split_dir / "txt" / f"{split_name}.hi"
        if not txt_path.is_file():
            raise FileNotFoundError(f"Missing Hindi translation file for split '{split_name}' at {txt_path}")
        with txt_path.open("r", encoding="utf-8") as f:
            translations = [line.rstrip("\n") for line in f]
        if len(translations) != len(lines):
            raise RuntimeError(
                f"Mismatch for split '{split_name}': {len(translations)} translations vs {len(lines)} stamped entries"
            )

    segments: List[Segment] = []
    for idx, line in enumerate(lines):
        parts = line.split("\t")
        if len(parts) != 3:
            raise ValueError(
                f"Expected 3 columns in stamped.tsv but got {len(parts)} at split '{split_name}' line {idx + 1}: {line}"
            )
        rel_audio, start_str, duration_str = parts
        rel_path = Path(rel_audio)
        try:
            start = float(start_str)
            duration = float(duration_str)
        except ValueError as exc:
            raise ValueError(
                f"Could not parse start/duration in split '{split_name}' line {idx + 1}: {line}"
            ) from exc

        translation = translations[idx] if has_transcripts else None
        segments.append(Segment(rel_audio_path=rel_path, start_sec=start, duration_sec=duration, translation=translation))

    return segments


def ensure_audio(
    src_audio: Path,
    dst_audio: Path,
    target_sr: int,
    force: bool,
) -> float:
    """Convert/copy audio to the destination path and return its duration in seconds."""

    if dst_audio.exists() and not force:
        info = sf.info(str(dst_audio))
        return info.frames / float(info.samplerate)

    dst_audio.parent.mkdir(parents=True, exist_ok=True)

    info = sf.info(str(src_audio))
    needs_resample = info.samplerate != target_sr
    needs_mono = info.channels != 1

    if not needs_resample and not needs_mono:
        shutil.copy2(src_audio, dst_audio)
        return info.frames / float(info.samplerate)

    audio, sr = sf.read(str(src_audio), dtype="float32")
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if sr != target_sr:
        audio = soxr.resample(audio, sr, target_sr)
        sr = target_sr

    sf.write(dst_audio, audio, sr, subtype="PCM_16")
    return len(audio) / float(sr)


def write_manifest(manifest_path: Path, records: Iterable[dict]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_tsv(tsv_path: Path, rows: Iterable[dict]) -> None:
    import csv

    tsv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["id", "audio", "n_frames", "tgt_text", "speaker"]
    with tsv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def human_time(seconds: float) -> str:
    seconds = int(round(seconds))
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def process_split(
    split_name: str,
    source_root: Path,
    output_root: Path,
    target_sr: int,
    has_transcripts: bool,
    force: bool,
    max_items: int,
    emit_tsv: bool,
) -> dict:
    split_source = source_root / split_name
    split_output = output_root / split_name

    segments = load_segments(split_source, split_name, has_transcripts)
    if max_items and max_items > 0:
        segments = segments[:max_items]

    manifest_records = []
    tsv_rows = []
    total_duration = 0.0

    for segment in segments:
        src_audio = split_source / segment.rel_audio_path
        if not src_audio.is_file():
            raise FileNotFoundError(f"Missing audio file referenced in stamped.tsv: {src_audio}")

        dst_audio = split_output / segment.rel_audio_path
        duration = ensure_audio(src_audio, dst_audio, target_sr, force=force)

        total_duration += duration

        record = {
            "id": segment.rel_audio_path.stem,
            "audio_filepath": str(dst_audio.resolve()),
            "duration_sec": round(duration, 6),
            "lang": {"source": "mr", "target": "hi"},
            "split": split_name,
            "start_sec": segment.start_sec,
            "end_sec": segment.end_sec,
        }
        if segment.translation is not None:
            record["translation"] = segment.translation
        manifest_records.append(record)

        if emit_tsv:
            rel_audio = Path(split_name) / segment.rel_audio_path
            tsv_rows.append(
                {
                    "id": segment.rel_audio_path.stem,
                    "audio": rel_audio.as_posix(),
                    "n_frames": int(round(duration * target_sr)),
                    "tgt_text": segment.translation if segment.translation is not None else "",
                    "speaker": "",
                }
            )

    manifest_path = output_root / "manifests" / f"{split_name}.jsonl"
    write_manifest(manifest_path, manifest_records)

    if emit_tsv:
        tsv_path = output_root / "manifests" / f"{split_name}.tsv"
        write_tsv(tsv_path, tsv_rows)

    return {
        "split": split_name,
        "num_segments": len(segments),
        "total_duration_sec": total_duration,
        "manifest": manifest_path,
        "tsv": output_root / "manifests" / f"{split_name}.tsv" if emit_tsv else None,
        "audio_root": split_output,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare the Panlingua IWSLT2023 Marathi→Hindi dataset.")
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("datasets/iwslt2023_mr-hi"),
        help="Path to the cloned iwslt2023_mr-hi repository",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("processed_data/iwslt_16khz"),
        help="Destination directory for resampled audio and manifests",
    )
    parser.add_argument(
        "--target-sr",
        type=int,
        default=16000,
        help="Desired audio sampling rate (Hz)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-process audio files even if the destination already exists",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=0,
        help="Optionally limit processing to the first N items per split (useful for smoke tests)",
    )
    parser.add_argument(
        "--no-tsv",
        action="store_true",
        help="Skip writing Fairseq-style TSV manifests",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    source_root: Path = args.source_root
    output_root: Path = args.output_root
    target_sr: int = args.target_sr
    force: bool = args.force
    max_items: int = args.max_items
    emit_tsv: bool = not args.no_tsv

    if not source_root.exists():
        raise FileNotFoundError(f"Source root does not exist: {source_root}")

    output_root.mkdir(parents=True, exist_ok=True)

    summary = []

    for split in SPLITS_WITH_TEXT:
        result = process_split(
            split_name=split,
            source_root=source_root,
            output_root=output_root,
            target_sr=target_sr,
            has_transcripts=True,
            force=force,
            max_items=max_items,
            emit_tsv=emit_tsv,
        )
        summary.append(result)

    for split in SPLITS_AUDIO_ONLY:
        result = process_split(
            split_name=split,
            source_root=source_root,
            output_root=output_root,
            target_sr=target_sr,
            has_transcripts=False,
            force=force,
            max_items=max_items,
            emit_tsv=emit_tsv,
        )
        summary.append(result)

    print("\nPreparation complete:\n")
    for item in summary:
        duration_h = human_time(item["total_duration_sec"])
        print(
            f"  • {item['split']}: {item['num_segments']} segments, {duration_h} (hh:mm:ss) total, "
            f"manifest → {item['manifest']}"
        )
    print(f"\nResampled audio stored under: {output_root}\n")


if __name__ == "__main__":
    main()
