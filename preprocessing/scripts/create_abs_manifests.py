#!/usr/bin/env python3
"""Create Fairseq-style TSV manifests with absolute audio paths.

This is a small utility adapted from create_manifests.py that always
writes absolute audio paths. If soundfile cannot read a file (for
example .3gp containers), the script will set n_frames to 0 and log a
warning. This lets you create usable manifests without installing
ffmpeg/libsndfile variants that support every container.
"""
import argparse
import csv
import logging
from datetime import datetime
from pathlib import Path
from random import Random
from typing import List, Tuple

import soundfile as sf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create Fairseq-style speech manifests with absolute audio paths.",
    )
    parser.add_argument("--dataset-dir", type=Path, required=True, help="Path that contains the waveform files and index.")
    parser.add_argument("--line-index", type=Path, default=Path("line_index.tsv"), help="Relative (from --dataset-dir) or absolute path to the index TSV.")
    parser.add_argument("--train-output", type=Path, default=Path("train_abs.tsv"), help="Output path for the train manifest.")
    parser.add_argument("--valid-output", type=Path, default=Path("valid_abs.tsv"), help="Output path for the valid manifest.")
    parser.add_argument("--train-ratio", type=float, default=0.95, help="Fraction of examples assigned to the train split.")
    parser.add_argument("--seed", type=int, default=20240229, help="Random seed used for shuffling before split.")
    parser.add_argument("--log-dir", type=Path, default=Path("log/preprocessing"), help="Directory where an execution log will be stored.")
    parser.add_argument(
        "--scan-if-missing",
        action="store_true",
        help="Discover audio/text pairs by scanning when the line index is absent.",
    )
    parser.add_argument(
        "--scan-extensions",
        default=".wav,.flac,.mp3,.3gp",
        help="Comma separated audio extensions to consider while scanning (only used with --scan-if-missing).",
    )
    return parser.parse_args()


def configure_logging(log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"create_abs_manifests_{timestamp}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler()],
    )
    logging.info("Logging to %s", log_path)


def load_index(index_path: Path) -> List[Tuple[str, str]]:
    rows: List[Tuple[str, str]] = []
    with index_path.open("r", encoding="utf-8") as tsv_in:
        reader = csv.reader(tsv_in, delimiter="\t")
        for line_num, row in enumerate(reader, start=1):
            if len(row) != 2:
                raise ValueError(f"Line {line_num} in {index_path} does not have 2 columns: {row}")
            utt_id, text = row
            rows.append((utt_id.strip(), text.strip()))
    logging.info("Loaded %d labelled utterances from %s", len(rows), index_path)
    return rows


def discover_examples(dataset_dir: Path, extensions: List[str]) -> List[Tuple[str, str]]:
    ext_set = {ext.strip().lower() for ext in extensions if ext.strip()}
    rows: List[Tuple[str, str]] = []
    missing_transcripts = []
    seen_ids = set()
    for audio_path in sorted(dataset_dir.rglob("*")):
        if not audio_path.is_file():
            continue
        if audio_path.suffix.lower() not in ext_set:
            continue
        rel_path = audio_path.relative_to(dataset_dir)
        utt_id = rel_path.with_suffix("").as_posix()
        text_path = audio_path.with_suffix(".txt")
        if not text_path.is_file():
            missing_transcripts.append(text_path)
            continue
        text = text_path.read_text(encoding="utf-8").strip()
        if not text:
            logging.warning("Empty transcript for %s", text_path)
        if utt_id in seen_ids:
            logging.warning("Duplicate utterance id detected during scan: %s", utt_id)
            continue
        seen_ids.add(utt_id)
        rows.append((utt_id, text))
    if missing_transcripts:
        logging.warning("Skipped %d audio files without transcripts; sample: %s", len(missing_transcripts), missing_transcripts[:3])
    logging.info("Discovered %d audio/text pairs by scanning %s", len(rows), dataset_dir)
    return rows


def collect_audio_metadata(dataset_dir: Path, examples: List[Tuple[str, str]], extensions: List[str]):
    metadata = []
    missing = []
    ext_candidates = []
    for ext in extensions:
        ext = ext.strip()
        if not ext:
            continue
        if not ext.startswith("."):
            ext = f".{ext}"
        ext_candidates.append(ext.lower())
    for utt_id, text in examples:
        # Look for something like utterance.wav, utterance.3gp, etc.
        rel_path = Path(utt_id)
        base_path = dataset_dir / rel_path
        candidates = [base_path.with_suffix(ext) for ext in ext_candidates]
        candidates.append(base_path)
        audio_path = None
        for c in candidates:
            if c.is_file():
                audio_path = c
                break
        if audio_path is None:
            missing.append(dataset_dir / f"{utt_id}.wav")
            continue

        try:
            info = sf.info(audio_path)
            n_frames = info.frames
        except Exception:
            logging.warning("Could not read audio metadata for %s; setting n_frames=0", audio_path)
            n_frames = 0

        metadata.append((utt_id, audio_path.resolve(), n_frames, text))
    if missing:
        error_preview = "\n".join(str(p) for p in missing[:5])
        logging.error("Missing %d audio files. Sample: %s", len(missing), error_preview)
        raise FileNotFoundError("Some waveform files listed in the index are missing.")
    logging.info("Collected metadata for %d audio files", len(metadata))
    return metadata


def split_examples(examples, train_ratio: float, seed: int):
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be between 0 and 1")
    rng = Random(seed)
    sorted_examples = sorted(examples, key=lambda item: item[0])
    rng.shuffle(sorted_examples)
    cutoff = int(len(sorted_examples) * train_ratio)
    cutoff = min(max(cutoff, 1), len(sorted_examples) - 1)
    train_split = sorted_examples[:cutoff]
    valid_split = sorted_examples[cutoff:]
    logging.info("Split %d samples into %d train and %d validation", len(examples), len(train_split), len(valid_split))
    return train_split, valid_split


def write_manifest(rows, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as tsv_out:
        writer = csv.writer(tsv_out, delimiter="\t")
        writer.writerow(["id", "audio", "n_frames", "tgt_text"])
        for utt_id, audio_path, n_frames, text in rows:
            writer.writerow([utt_id, str(audio_path), n_frames, text])
    logging.info("Wrote manifest with %d rows to %s", len(rows), output_path)


def describe_lengths(rows, split_name: str) -> None:
    if not rows:
        logging.warning("No rows present in %s split; skipping stats", split_name)
        return
    frame_counts = [n_frames for _, _, n_frames, _ in rows]
    min_frames = min(frame_counts)
    max_frames = max(frame_counts)
    avg_frames = sum(frame_counts) / len(frame_counts)
    logging.info(
        "%s length stats (frames): min=%d max=%d avg=%.1f",
        split_name,
        min_frames,
        max_frames,
        avg_frames,
    )


def main():
    args = parse_args()
    dataset_dir = args.dataset_dir
    index_path = args.line_index if args.line_index.is_absolute() else dataset_dir / args.line_index

    configure_logging(args.log_dir)
    logging.info("Dataset directory: %s", dataset_dir)
    logging.info("Index file: %s", index_path)

    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")

    extensions = args.scan_extensions.split(",")

    if index_path.is_file():
        labelled_utts = load_index(index_path)
    else:
        if not args.scan_if_missing:
            raise FileNotFoundError(f"Index file does not exist: {index_path}")
        logging.info("Index missing; scanning for audio/text pairs")
        labelled_utts = discover_examples(dataset_dir, extensions)
        if not labelled_utts:
            raise FileNotFoundError("No audio/text pairs discovered during scanning")

    metadata = collect_audio_metadata(dataset_dir, labelled_utts, extensions)
    train_rows, valid_rows = split_examples(metadata, args.train_ratio, args.seed)

    write_manifest(train_rows, args.train_output)
    write_manifest(valid_rows, args.valid_output)

    describe_lengths(train_rows, "train")
    describe_lengths(valid_rows, "valid")

    logging.info("Done")


if __name__ == "__main__":
    main()
