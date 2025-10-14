#!/usr/bin/env python3
"""Train a SentencePiece tokenizer for the Hindi target side and export a Fairseq-compatible dictionary."""

from __future__ import annotations

import argparse
import io
import tempfile
from pathlib import Path
from typing import Iterable, List

import sentencepiece as spm

SPECIAL_TOKENS = ["<unk>", "<pad>", "<s>", "</s>"]


def read_corpus(paths: Iterable[Path]) -> List[str]:
    lines: List[str] = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Input text file not found: {path}")
        with path.open("r", encoding="utf-8") as f:
            lines.extend(line.strip() for line in f if line.strip())
    return lines


def train_sentencepiece(lines: List[str], vocab_size: int, model_prefix: Path, model_type: str, character_coverage: float) -> Path:
    model_prefix.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as tmp:
        tmp.write("\n".join(lines))
        tmp_path = Path(tmp.name)

    spm.SentencePieceTrainer.train(
        input=str(tmp_path),
        vocab_size=vocab_size,
        model_prefix=str(model_prefix),
        character_coverage=character_coverage,
        model_type=model_type,
        pad_id=3,
        pad_piece="<pad>",
    )

    tmp_path.unlink(missing_ok=True)
    return model_prefix.with_suffix(".model")


def export_fairseq_dict(model_path: Path, dict_path: Path) -> None:
    sp = spm.SentencePieceProcessor()
    sp.load(str(model_path))

    dict_path.parent.mkdir(parents=True, exist_ok=True)

    with dict_path.open("w", encoding="utf-8") as f:
        f.write("<unk> 1\n")
        f.write("<pad> 0\n")
        f.write("<s> 0\n")
        f.write("</s> 0\n")

        for idx in range(sp.get_piece_size()):
            piece = sp.id_to_piece(idx)
            if piece in SPECIAL_TOKENS:
                continue
            f.write(f"{piece} 1\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SentencePiece model and Fairseq dictionary for Hindi targets.")
    parser.add_argument(
        "--input-text",
        nargs="+",
        type=Path,
        required=True,
        help="Paths to UTF-8 text files containing Hindi translations (one sentence per line)",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=4000,
        help="Target vocabulary size for SentencePiece",
    )
    parser.add_argument(
        "--model-prefix",
        type=Path,
        default=Path("processed_data/iwslt_16khz/spm/mr_hi"),
        help="Output prefix for SentencePiece model ('.model' and '.vocab' will be created)",
    )
    parser.add_argument(
        "--dict-path",
        type=Path,
        default=Path("processed_data/iwslt_16khz/dict.txt"),
        help="Path to write the Fairseq dictionary",
    )
    parser.add_argument(
        "--model-type",
        choices=["unigram", "bpe", "char", "word"],
        default="unigram",
        help="SentencePiece model type",
    )
    parser.add_argument(
        "--character-coverage",
        type=float,
        default=0.9995,
        help="Character coverage for SentencePiece training",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_texts = read_corpus(args.input_text)
    model_path = train_sentencepiece(
        lines=input_texts,
        vocab_size=args.vocab_size,
        model_prefix=args.model_prefix,
        model_type=args.model_type,
        character_coverage=args.character_coverage,
    )

    export_fairseq_dict(model_path, args.dict_path)

    print("SentencePiece model written to:", model_path)
    print("Dictionary written to:", args.dict_path)


if __name__ == "__main__":
    main()
