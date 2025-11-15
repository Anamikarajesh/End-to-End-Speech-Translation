#!/usr/bin/env python3
"""Utility to create Fairseq speech manifests with deterministic train/valid split."""
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
        description="Create Fairseq-style speech manifests with a train/valid split.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("dataset/mr_in_female"),
        help="Path that contains line_index.tsv and the waveform files.",
    )
    parser.add_argument(
        "--line-index",
        type=Path,
        default=Path("line_index.tsv"),
        help="Relative path (from --dataset-dir) or absolute path to the index TSV.",
    )
    parser.add_argument(
        "--train-output",
        type=Path,
        default=Path("train.tsv"),
        help="Output path (relative to --dataset-dir) for the train manifest.",
    )
    parser.add_argument(
        "--valid-output",
        type=Path,
        default=Path("valid.tsv"),
        help="Output path (relative to --dataset-dir) for the validation manifest.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.95,
        help="Fraction of examples assigned to the train split.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20240229,
        help="Random seed used for shuffling before the split.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("log/preprocessing"),
        help="Directory where an execution log will be stored.",
    )
    return parser.parse_args()


def configure_logging(log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"create_manifests_{timestamp}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
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


def collect_audio_metadata(dataset_dir: Path, examples: List[Tuple[str, str]]):
    metadata = []
    missing = []
    for utt_id, text in examples:
        audio_path = dataset_dir / f"{utt_id}.wav"
        if not audio_path.is_file():
            missing.append(audio_path)
            continue
        info = sf.info(audio_path)
        # We rely on the frame count instead of duration to avoid rounding errors.
        metadata.append((utt_id, audio_path, info.frames, text))
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
    logging.info(
        "Split %d samples into %d train and %d validation", len(examples), len(train_split), len(valid_split)
    )
    return train_split, valid_split


def write_manifest(dataset_dir: Path, rows, output_path: Path) -> None:
    output_path = output_path if output_path.is_absolute() else dataset_dir / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as tsv_out:
        writer = csv.writer(tsv_out, delimiter="\t")
        writer.writerow(["id", "audio", "n_frames", "text"])
        for utt_id, audio_path, n_frames, text in rows:
            rel_audio = audio_path.relative_to(dataset_dir)
            writer.writerow([utt_id, rel_audio.as_posix(), n_frames, text])
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
    if not index_path.is_file():
        raise FileNotFoundError(f"Index file does not exist: {index_path}")

    labelled_utts = load_index(index_path)
    metadata = collect_audio_metadata(dataset_dir, labelled_utts)
    train_rows, valid_rows = split_examples(metadata, args.train_ratio, args.seed)

    write_manifest(dataset_dir, train_rows, args.train_output)
    write_manifest(dataset_dir, valid_rows, args.valid_output)

    describe_lengths(train_rows, "train")
    describe_lengths(valid_rows, "valid")

    logging.info("Done")


if __name__ == "__main__":
    main()
