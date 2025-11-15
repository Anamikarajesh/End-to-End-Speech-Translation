import argparse
import json
from pathlib import Path
from typing import List, Tuple

import sacrebleu
import sentencepiece as spm


def parse_generate_file(path: Path) -> Tuple[List[str], List[str]]:
    """Read a fairseq generate output file and collect hypothesis/reference lines."""
    hyps: List[str] = []
    refs: List[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            if raw_line.startswith("H-"):
                parts = raw_line.rstrip("\n").split("\t")
                if len(parts) >= 3:
                    hyps.append(parts[2])
            elif raw_line.startswith("T-"):
                parts = raw_line.rstrip("\n").split("\t")
                if len(parts) >= 2:
                    refs.append(parts[1])
    return hyps, refs


def maybe_detok(tokenizer: spm.SentencePieceProcessor, text: str, always: bool = False) -> str:
    """Decode SentencePiece text when it looks tokenized."""
    cleaned = text.strip()
    if not cleaned:
        return ""
    if always or "▁" in cleaned:
        return tokenizer.decode(cleaned.split())
    return cleaned


def detokenize_pairs(
    tokenizer: spm.SentencePieceProcessor,
    hypotheses: List[str],
    references: List[str],
    decode_refs: bool,
) -> Tuple[List[str], List[str]]:
    detok_hyps = [maybe_detok(tokenizer, h, always=True) for h in hypotheses]
    detok_refs = [maybe_detok(tokenizer, r, always=decode_refs) for r in references]
    return detok_hyps, detok_refs


def ensure_lengths(hypotheses: List[str], references: List[str]) -> Tuple[List[str], List[str]]:
    if len(hypotheses) != len(references):
        n = min(len(hypotheses), len(references))
        print(f"Warning: length mismatch (hyp={len(hypotheses)}, ref={len(references)}); truncating to {n}")
        hypotheses = hypotheses[:n]
        references = references[:n]
    return hypotheses, references


def write_outputs(
    hypotheses: List[str],
    references: List[str],
    output_dir: Path,
) -> Tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    hyp_path = output_dir / "model_translations.detok"
    ref_path = output_dir / "reference_translations.detok"
    hyp_path.write_text("\n".join(hypotheses) + "\n", encoding="utf-8")
    ref_path.write_text("\n".join(references) + "\n", encoding="utf-8")
    return hyp_path, ref_path


def compute_bleu(hypotheses: List[str], references: List[str]) -> sacrebleu.metrics.bleu.BLEUScore:
    return sacrebleu.corpus_bleu(hypotheses, [references])


def main() -> None:
    parser = argparse.ArgumentParser(description="Detokenize Fairseq outputs and compute sacreBLEU.")
    parser.add_argument("generate_file", type=Path, help="Path to fairseq generate output (e.g., generate-valid.txt).")
    parser.add_argument("data_dir", type=Path, help="Path to the data-bin directory containing spm_6k.model.")
    parser.add_argument("--decode-references", action="store_true", help="Force SentencePiece detokenization on references as well.")
    parser.add_argument("--output-dir", type=Path, default=Path("inference"), help="Directory to store detokenized outputs.")
    parser.add_argument("--save-json", type=Path, default=None, help="Optional path to save BLEU details as JSON.")
    args = parser.parse_args()

    spm_model = args.data_dir / "spm_6k.model"
    if not spm_model.exists():
        raise FileNotFoundError(f"SentencePiece model not found at {spm_model}")

    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(str(spm_model))

    raw_hyps, raw_refs = parse_generate_file(args.generate_file)
    detok_hyps, detok_refs = detokenize_pairs(tokenizer, raw_hyps, raw_refs, args.decode_references)
    detok_hyps, detok_refs = ensure_lengths(detok_hyps, detok_refs)

    hyp_path, ref_path = write_outputs(detok_hyps, detok_refs, args.output_dir)
    bleu = compute_bleu(detok_hyps, detok_refs)

    print("=== sacreBLEU ===")
    print(bleu.format())
    print(f"Hypotheses saved to: {hyp_path}")
    print(f"References saved to: {ref_path}")

    if args.save_json is not None:
        bleu_payload = {
            "score": bleu.score,
            "precisions": bleu.precisions,
            "bp": bleu.bp,
            "sys_len": bleu.sys_len,
            "ref_len": bleu.ref_len,
        }
        signature = getattr(bleu, "signature", None)
        if signature is not None:
            bleu_payload["signature"] = signature
        args.save_json.write_text(json.dumps(bleu_payload, indent=2), encoding="utf-8")
        print(f"BLEU details written to: {args.save_json}")


if __name__ == "__main__":
    main()
