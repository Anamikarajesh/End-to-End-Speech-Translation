import argparse
import os
from pathlib import Path

import numpy as np
import sentencepiece as spm
import soundfile as sf
import torch
import torchaudio
from fairseq import checkpoint_utils, tasks


def validate_paths(model_path: Path, data_dir: Path, audio_file: Path) -> bool:
    """Return True when all required paths exist; otherwise print an error."""
    if not model_path.exists():
        print(f"Error: Model checkpoint not found at: {model_path}")
        return False
    if not data_dir.exists():
        print(f"Error: Data directory not found at: {data_dir}")
        return False
    if not audio_file.exists():
        print(f"Error: Audio file not found at: {audio_file}")
        return False

    spm_path = data_dir / "spm_6k.model"
    if not spm_path.exists():
        print(f"Error: SentencePiece model not found at: {spm_path}")
        return False

    return True


def load_model_and_task(model_path: Path, data_dir: Path):
    """Load the model ensemble, config, and task for inference."""
    print("--- Loading model and task... ---")
    models, cfg, task = checkpoint_utils.load_model_ensemble_and_task(
        [str(model_path)],
        arg_overrides={"data": str(data_dir)},
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models[0].to(device).eval()
    generator = task.build_generator([model], cfg.generation)
    return model, generator, task, cfg, device


def load_tokenizer(data_dir: Path) -> spm.SentencePieceProcessor:
    """Load the SentencePiece tokenizer used for detokenizing model outputs."""
    spm_path = data_dir / "spm_6k.model"
    print(f"--- Loading tokenizer: {spm_path} ---")
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(str(spm_path))
    return tokenizer


def prepare_dataset(task, cfg):
    """Load the validation split so we can reuse its processing pipeline."""
    subset = cfg.dataset.valid_subset
    task.load_dataset(subset)
    return task.dataset(subset)


def load_features(dataset, audio_file: Path, device: torch.device):
    """Fetch model-ready features for the requested audio path."""
    audio_str = str(audio_file)
    audio_abs = str(audio_file.resolve())
    if audio_str in dataset.audio_paths:
        idx = dataset.audio_paths.index(audio_str)
    elif audio_abs in dataset.audio_paths:
        idx = dataset.audio_paths.index(audio_abs)
    else:
        idx = -1

    if idx >= 0:
        item = dataset[idx]
        features = item.source
        if not torch.is_tensor(features):
            features = torch.from_numpy(features)
        return features.float().to(device)

    # Fallback: run the classic S2T transforms for external audio.
    print(f"--- Loading audio: {audio_file} ---")
    waveform_np, sample_rate = sf.read(audio_str)
    if waveform_np.ndim == 1:
        waveform = torch.from_numpy(waveform_np).unsqueeze(0)
    else:
        waveform = torch.from_numpy(np.transpose(waveform_np, (1, 0)))
    waveform = waveform.float()

    target_sample_rate = getattr(dataset.cfg, "use_sample_rate", 16000)
    if sample_rate != target_sample_rate:
        print(
            f"Warning: Resampling audio from {sample_rate}Hz to {target_sample_rate}Hz"
        )
        resampler = torchaudio.transforms.Resample(
            orig_freq=sample_rate,
            new_freq=target_sample_rate,
        )
        waveform = resampler(waveform)

    if dataset.feature_transforms is None:
        raise RuntimeError(
            "This audio is not part of the prepared manifests and no feature "
            "transform pipeline is defined. Please add the file to a manifest "
            "or update the script with a custom frontend."
        )

    features = dataset.feature_transforms(waveform.squeeze(0).numpy())
    if not torch.is_tensor(features):
        features = torch.from_numpy(features)
    return features.float().to(device)


def run_inference(model, generator, task, tokenizer, features_tensor):
    """Run beam search decoding and detokenize the resulting tokens."""
    sample = {
        "net_input": {
            "src_tokens": features_tensor.unsqueeze(0),
            "src_lengths": torch.tensor(
                [features_tensor.shape[0]], dtype=torch.long, device=features_tensor.device
            ),
        }
    }

    print("--- Translating... ---")
    with torch.no_grad():
        translation = task.inference_step(generator, [model], sample)

    token_ids = translation[0][0]["tokens"].cpu().numpy().tolist()
    return tokenizer.decode(token_ids)


def main(args):
    model_path = Path(args.model_path)
    data_dir = Path(args.data_dir)
    audio_file = Path(args.audio_file)

    if not validate_paths(model_path, data_dir, audio_file):
        return

    model, generator, task, cfg, device = load_model_and_task(model_path, data_dir)
    tokenizer = load_tokenizer(data_dir)
    dataset = prepare_dataset(task, cfg)
    print("--- Loading audio features ---")
    features_tensor = load_features(dataset, audio_file, device)
    translation_text = run_inference(model, generator, task, tokenizer, features_tensor)

    print("\n" + "=" * 30)
    print(f"Audio File: {audio_file.name}")
    print(f"Model Used: {model_path.name}")
    print("\n--- TRANSLATION (HINDI) ---")
    print(translation_text)
    print("=" * 30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Translate a single audio file using a Fairseq S2T model.")
    parser.add_argument("--model-path", required=True, help="Path to the checkpoint .pt file")
    parser.add_argument("--data-dir", required=True, help="Path to the data-bin directory")
    parser.add_argument("--audio-file", required=True, help="Path to the .wav file to translate")
    main(parser.parse_args())
