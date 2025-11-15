#!/usr/bin/env python3
"""Utility to assemble a Fairseq speech-to-text data directory.

This script takes existing manifest TSV files that list absolute audio paths
plus SentencePiece assets and writes them into the directory structure that
``fairseq-train`` expects when using the ``speech_to_text`` task with
``--config-yaml``. By default the generated config enables on-the-fly feature
extraction (Fbank + utterance CMVN, with optional SpecAugment).
"""
from __future__ import annotations

import argparse
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import csv

import soundfile as sf

try:
    import yaml
except ImportError as exc:  # pragma: no cover - dependency guard
    raise SystemExit("PyYAML is required: pip install pyyaml") from exc


def _parse_manifest_arg(value: str) -> Tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            f"Manifest specification '{value}' is missing '='. Expected format split=/path/to.tsv"
        )
    split, path = value.split("=", 1)
    split = split.strip()
    if not split:
        raise argparse.ArgumentTypeError(f"Invalid split name in '{value}'")
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"Manifest file not found: {path}")
    return split, path


@dataclass
class ManifestStats:
    split: str
    num_samples: int
    total_seconds: float

    @property
    def total_hours(self) -> float:
        return self.total_seconds / 3600


def _ensure_required_columns(reader: csv.DictReader, path: Path) -> None:
    required_cols = {"id", "audio", "n_frames", "tgt_text"}
    if not required_cols.issubset(reader.fieldnames or set()):
        raise ValueError(
            f"Manifest {path} is missing required columns {required_cols}. Found {reader.fieldnames}."
        )


def _duration_to_frames(duration_s: float) -> int:
    return max(int(round(duration_s * 100)), 1)


def _compute_duration(audio_path: str) -> float:
    info = sf.info(audio_path)
    if info.samplerate == 0:
        raise ValueError(f"Invalid sample rate for {audio_path}")
    return info.frames / info.samplerate


def _copy_manifest(src: Path, dst: Path) -> ManifestStats:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open("r", encoding="utf-8") as in_tsv, dst.open(
        "w", encoding="utf-8", newline=""
    ) as out_tsv:
        reader = csv.DictReader(in_tsv, delimiter="\t")
        _ensure_required_columns(reader, src)
        writer = csv.DictWriter(out_tsv, fieldnames=reader.fieldnames, delimiter="\t")
        writer.writeheader()

        num_samples = 0
        total_seconds = 0.0
        for row in reader:
            audio_path = row["audio"]
            duration_s = _compute_duration(audio_path)
            row["n_frames"] = str(_duration_to_frames(duration_s))
            writer.writerow(row)
            num_samples += 1
            total_seconds += duration_s

    split = dst.stem
    return ManifestStats(split=split, num_samples=num_samples, total_seconds=total_seconds)


SPECIAL_TOKENS = {"<pad>", "<s>", "</s>", "<unk>"}


def _copy_asset(src: Path, dst_dir: Path) -> Path:
    target = dst_dir / src.name
    shutil.copy(src, target)
    return target


def _write_vocab_with_overwrite(src: Path, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open("r", encoding="utf-8") as inp, dst.open("w", encoding="utf-8") as outp:
        for line in inp:
            stripped = line.rstrip("\n")
            if not stripped:
                outp.write("\n")
                continue
            parts = stripped.split(" ")
            token = parts[0]
            if token in SPECIAL_TOKENS and "#fairseq:overwrite" not in parts:
                stripped = f"{stripped} #fairseq:overwrite"
            outp.write(stripped + "\n")
    return dst


def _build_feature_transforms(sample_rate: int, include_specaugment: bool) -> Dict:
    transforms = [
        {
            "_target_": "fairseq.data.audio.feature_transforms.FbankFeat",
            "num_mel_bins": 80,
            "sample_rate": sample_rate,
            "dither": 0.0,
        },
        {
            "_target_": "fairseq.data.audio.feature_transforms.UtteranceCMVN",
        },
    ]
    if include_specaugment:
        transforms.append(
            {
                "_target_": "fairseq.data.audio.feature_transforms.SpecAugment",
                "freq_mask_n": 1,
                "freq_mask_max_f": 27,
                "time_mask_n": 1,
                "time_mask_max_t": 100,
                "time_mask_max_p": 1.0,
            }
        )
    return {
        "_target_": "fairseq.data.audio.feature_transforms.CompositeAudioFeatureTransform",
        "transforms": transforms,
    }


def _write_config(
    output_dir: Path,
    vocab_filename: str,
    spm_model_filename: str,
    sample_rate: int,
    use_audio_input: bool,
    audio_root: str,
    extra_config: Dict,
    config_name: str,
    include_specaugment: bool,
) -> Path:
    config = {
        "sample_rate": sample_rate,
        "use_audio_input": use_audio_input,
        "audio_root": audio_root,
        "vocab_filename": vocab_filename,
        "bpe_tokenizer": {
            "bpe": "sentencepiece",
            "sentencepiece_model": spm_model_filename,
        },
        "feature_transforms": _build_feature_transforms(sample_rate, include_specaugment),
    }
    config.update(extra_config)
    config_path = output_dir / config_name
    with config_path.open("w", encoding="utf-8") as yaml_out:
        yaml.safe_dump(config, yaml_out, sort_keys=False)
    return config_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        action="append",
        type=_parse_manifest_arg,
        required=True,
        help="Manifest specification in the form split=/absolute/path.tsv. Repeatable.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Destination directory for the assembled data bin.",
    )
    parser.add_argument(
        "--spm-model",
        type=Path,
        required=True,
        help="Path to the SentencePiece .model file used for targets.",
    )
    parser.add_argument(
        "--spm-vocab",
        type=Path,
        required=False,
        help="Optional path to the SentencePiece vocabulary (.txt) file. If omitted the script looks for a sibling .txt next to the model.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Sample rate of the audio recordings in Hz.",
    )
    parser.add_argument(
        "--audio-root",
        type=str,
        default="",
        help="Optional root to prepend to relative audio paths. Leave empty when manifests use absolute paths.",
    )
    parser.add_argument(
        "--config-name",
        type=str,
        default="config.yaml",
        help="Filename of the generated YAML config.",
    )
    parser.add_argument(
        "--use-audio-input",
        action="store_true",
        help="Store configs that feed raw waveforms to the model (skip on-the-fly feature extraction).",
    )
    parser.add_argument(
        "--disable-specaugment",
        action="store_true",
        help="Omit SpecAugment from the generated feature pipeline (useful for fine-tuning/eval).",
    )
    parser.add_argument(
        "--extra-config",
        type=Path,
        help="Optional YAML/JSON file whose key-value pairs will be merged into the generated config.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    return parser.parse_args()


def _load_extra_config(extra_path: Path | None) -> Dict:
    if extra_path is None:
        return {}
    if not extra_path.is_file():
        raise FileNotFoundError(f"Extra config file not found: {extra_path}")
    suffix = extra_path.suffix.lower()
    if suffix in {".yml", ".yaml"}:
        with extra_path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}
    if suffix == ".json":
        import json

        with extra_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    raise ValueError("Unsupported extra config format; expected YAML or JSON")


def _ensure_required_splits(manifests: Iterable[Tuple[str, Path]]):
    splits = {name for name, _ in manifests}
    missing = {"train", "valid"} - splits
    if missing:
        raise ValueError(f"Missing required manifest splits: {', '.join(sorted(missing))}")


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s %(message)s")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifests: Tuple[Tuple[str, Path], ...] = tuple(args.manifest)
    _ensure_required_splits(manifests)

    copied_stats = []
    for split, src_path in manifests:
        dest_path = output_dir / f"{split}.tsv"
        stats = _copy_manifest(src_path, dest_path)
        copied_stats.append(stats)
        logging.info("Copied %s -> %s (%d samples, %.2f h)", src_path, dest_path, stats.num_samples, stats.total_hours)

    spm_model_path = args.spm_model.expanduser().resolve()
    if not spm_model_path.is_file():
        raise FileNotFoundError(f"SentencePiece model not found: {spm_model_path}")
    copied_model = _copy_asset(spm_model_path, output_dir)

    if args.spm_vocab is not None:
        spm_vocab_path = args.spm_vocab.expanduser().resolve()
        if not spm_vocab_path.is_file():
            raise FileNotFoundError(f"SentencePiece vocab not found: {spm_vocab_path}")
    else:
        spm_vocab_path = spm_model_path.with_suffix(".txt")
        if not spm_vocab_path.is_file():
            raise FileNotFoundError(
                "--spm-vocab was omitted and the inferred vocab file does not exist: "
                f"{spm_vocab_path}"
            )
    copied_vocab = _write_vocab_with_overwrite(spm_vocab_path, output_dir / spm_vocab_path.name)

    extra_config = _load_extra_config(args.extra_config)
    config_path = _write_config(
        output_dir=output_dir,
        vocab_filename=copied_vocab.name,
        spm_model_filename=copied_model.name,
        sample_rate=args.sample_rate,
    use_audio_input=args.use_audio_input,
        audio_root=args.audio_root,
        extra_config=extra_config,
        config_name=args.config_name,
        include_specaugment=not args.disable_specaugment,
    )
    logging.info("Wrote config to %s", config_path)

    logging.info("Data bin ready at %s", output_dir)


if __name__ == "__main__":
    main()
