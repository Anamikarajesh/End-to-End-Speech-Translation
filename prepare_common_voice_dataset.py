#!/usr/bin/env python3
"""Normalize Common Voice Marathi ASR data to 16 kHz and build Fairseq manifests."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import torchaudio
from torchaudio import functional as F


# Allow very large TSV fields (Common Voice metadata can exceed the default limit).
try:
    csv.field_size_limit(sys.maxsize)
except OverflowError:
    csv.field_size_limit(10**9)


@dataclass
class Utterance:
    clip_rel_path: Path
    sentence: str
    client_id: str
    id: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Common Voice Marathi dataset")
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("datasets/common_voice_mr"),
        help="Directory containing Common Voice split TSVs and clips/ folder",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("processed_data/common_voice_16khz"),
        help="Destination for normalized audio and manifests",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "dev", "test"],
        help="List of Common Voice splits to process (expects <split>.tsv)",
    )
    parser.add_argument(
        "--target-sr",
        type=int,
        default=16000,
        help="Target sample rate for output audio",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=0,
        help="Limit the number of utterances per split (for smoke tests)",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=0.5,
        help="Minimum duration (seconds) to keep an utterance",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=20.0,
        help="Maximum duration (seconds) to keep an utterance",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-create audio files even if they already exist",
    )
    parser.add_argument(
        "--no-tsv",
        action="store_true",
        help="Skip writing Fairseq TSV manifests (JSONL is always written)",
    )
    return parser.parse_args()


def load_split(split_tsv: Path, max_items: int) -> List[Utterance]:
    if not split_tsv.is_file():
        raise FileNotFoundError(f"Missing Common Voice TSV: {split_tsv}")

    utterances: List[Utterance] = []
    with split_tsv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            sentence = (row.get("sentence") or "").strip()
            clip_rel = row.get("path")
            client_id = row.get("client_id", "")
            if not sentence or not clip_rel:
                continue
            stem = Path(clip_rel).stem
            utt_id = row.get("id") or stem
            utterances.append(
                Utterance(
                    clip_rel_path=Path(clip_rel),
                    sentence=sentence,
                    client_id=client_id,
                    id=utt_id,
                )
            )
            if max_items and len(utterances) >= max_items:
                break
    return utterances


def ensure_audio(
    src: Path,
    dst: Path,
    target_sr: int,
    force: bool,
) -> float:
    if dst.exists() and not force:
        info = torchaudio.info(str(dst))
        return info.num_frames / float(info.sample_rate)

    waveform, sample_rate = torchaudio.load(str(src))
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sample_rate != target_sr:
        waveform = F.resample(waveform, sample_rate, target_sr)
        sample_rate = target_sr

    dst.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(dst), waveform, sample_rate, encoding="PCM_S", bits_per_sample=16)
    return waveform.size(1) / float(sample_rate)


def write_jsonl(path: Path, records: Iterable[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_tsv(path: Path, rows: Iterable[Dict]) -> None:
    fieldnames = ["id", "audio", "n_frames", "tgt_text", "speaker"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_text(path: Path, sentences: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for sentence in sentences:
            f.write(sentence + "\n")


def process_split(
    split: str,
    args: argparse.Namespace,
) -> Dict:
    split_tsv = args.source_root / f"{split}.tsv"
    utterances = load_split(split_tsv, args.max_items)
    clips_root = args.source_root / "clips"
    output_split = args.output_root / split / "wav"

    jsonl_records: List[Dict] = []
    tsv_rows: List[Dict] = []
    text_lines: List[str] = []
    total_duration = 0.0
    kept = 0

    for utt in utterances:
        src_audio = clips_root / utt.clip_rel_path
        if not src_audio.is_file():
            raise FileNotFoundError(f"Audio file referenced in {split_tsv} not found: {src_audio}")

        dst_name = utt.id.replace("/", "_") + ".wav"
        dst_audio = output_split / dst_name
        duration = ensure_audio(src_audio, dst_audio, args.target_sr, args.force)

        if duration < args.min_duration or duration > args.max_duration:
            continue

        kept += 1
        total_duration += duration
        text_lines.append(utt.sentence)

        jsonl_records.append(
            {
                "id": utt.id,
                "audio_filepath": str(dst_audio.resolve()),
                "duration_sec": round(duration, 6),
                "text": utt.sentence,
                "lang": "mr",
                "speaker": utt.client_id,
                "split": split,
            }
        )

        rel_audio = Path(split) / "wav" / dst_name
        tsv_rows.append(
            {
                "id": utt.id,
                "audio": rel_audio.as_posix(),
                "n_frames": int(round(duration * args.target_sr)),
                "tgt_text": utt.sentence,
                "speaker": utt.client_id,
            }
        )

    manifests_root = args.output_root / "manifests"
    write_jsonl(manifests_root / f"{split}.jsonl", jsonl_records)
    if not args.no_tsv:
        write_tsv(manifests_root / f"{split}.tsv", tsv_rows)
    write_text(args.output_root / split / "txt" / f"{split}.txt", text_lines)

    return {
        "split": split,
        "num_utterances": kept,
        "duration_sec": total_duration,
        "jsonl": manifests_root / f"{split}.jsonl",
        "tsv": manifests_root / f"{split}.tsv" if not args.no_tsv else None,
    }


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    summary = []
    for split in args.splits:
        result = process_split(split, args)
        summary.append(result)

    print("\nCommon Voice preparation summary:\n")
    for item in summary:
        hours = int(item["duration_sec"] // 3600)
        minutes = int((item["duration_sec"] % 3600) // 60)
        seconds = int(item["duration_sec"] % 60)
        print(
            f"  • {item['split']}: {item['num_utterances']} utterances, "
            f"{hours:02d}:{minutes:02d}:{seconds:02d} (hh:mm:ss), "
            f"manifest → {item['jsonl']}"
        )


if __name__ == "__main__":
    main()
