#!/usr/bin/env python3
"""Generate Fairseq S2T data config YAML for the IWSLT Marathi→Hindi corpus."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create Fairseq S2T data config YAML")
    parser.add_argument(
        "--manifest-root",
        type=Path,
        default=Path("processed_data/iwslt_16khz/manifests"),
        help="Directory containing Fairseq TSV manifests",
    )
    parser.add_argument(
        "--spm",
        type=str,
        default="spm/mr_hi.model",
        help="SentencePiece model path relative to manifest root",
    )
    parser.add_argument(
        "--vocab",
        type=str,
        default="dict.txt",
        help="Dictionary filename relative to manifest root",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="config.yaml",
        help="Name of the YAML file to generate inside manifest root",
    )
    parser.add_argument(
        "--specaugment",
        choices=["lb", "ld", "sm", "ss", "none"],
        default="lb",
        help="SpecAugment policy to embed in the config",
    )
    parser.add_argument(
        "--audio-root",
        type=str,
        default="",
        help="Audio root to set in config (relative paths recommended)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = {
        "sample_rate": 16000,
        "input_channels": 1,
        "input_feat_per_channel": 80,
        "use_audio_input": False,
        "standardize_audio": False,
        "audio_root": args.audio_root,
        "vocab_filename": args.vocab,
        "bpe_tokenizer": {
            "bpe": "sentencepiece",
            "sentencepiece_model": args.spm,
        },
        "transforms": {
            "*": ["utterance_cmvn"],
        },
    }

    policy_map = {
        "lb": {"time_wrap_W": 0, "freq_mask_N": 1, "freq_mask_F": 27, "time_mask_N": 1, "time_mask_T": 100, "time_mask_p": 1.0},
        "ld": {"time_wrap_W": 0, "freq_mask_N": 2, "freq_mask_F": 27, "time_mask_N": 2, "time_mask_T": 100, "time_mask_p": 1.0},
        "sm": {"time_wrap_W": 0, "freq_mask_N": 2, "freq_mask_F": 15, "time_mask_N": 2, "time_mask_T": 70, "time_mask_p": 0.2},
        "ss": {"time_wrap_W": 0, "freq_mask_N": 2, "freq_mask_F": 27, "time_mask_N": 2, "time_mask_T": 70, "time_mask_p": 0.2},
    }

    if args.specaugment != "none":
        config["specaugment"] = policy_map[args.specaugment]
        config["transforms"]["_train"] = ["utterance_cmvn", "specaugment"]
    else:
        config["transforms"]["_train"] = ["utterance_cmvn"]

    output_path = args.manifest_root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, allow_unicode=True, sort_keys=False)

    print(f"Config written to {output_path} with SpecAugment policy='{args.specaugment}'.")


if __name__ == "__main__":
    main()
