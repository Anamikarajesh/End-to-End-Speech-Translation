import os
import sys
import csv
import numpy as np
import torchaudio
import pandas as pd
from tqdm import tqdm
import torch
import soundfile as sf

DATA_ROOT = os.path.abspath(".")           
MP3_DIR   = os.path.join(DATA_ROOT, "clips")
WAV_DIR   = os.path.join(DATA_ROOT, "clips_16k")
FEAT_DIR  = os.path.join(DATA_ROOT, "features_80mel")
MANIFESTS = os.path.join(DATA_ROOT, "manifests")

os.makedirs(WAV_DIR, exist_ok=True)
os.makedirs(FEAT_DIR, exist_ok=True)
os.makedirs(MANIFESTS, exist_ok=True)  

def convert_mp3_to_wav():
    files = [f for f in os.listdir(MP3_DIR) if f.endswith(".mp3")]
    resamplers = {}

    for file in tqdm(files, desc="Converting MP3 → WAV"):
        src = os.path.join(MP3_DIR, file)
        dst = os.path.join(WAV_DIR, file.replace(".mp3", ".wav"))
        if os.path.exists(dst):
            continue
        try:
            wav, sr = torchaudio.load(src)
        except Exception:
            wav_np, sr = sf.read(src, dtype="float32", always_2d=True)
            wav = torch.from_numpy(wav_np.T)
        if sr != 16000:
            if sr not in resamplers:
                resamplers[sr] = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
            wav = resamplers[sr](wav)
            sr = 16000
        wav_np = wav.transpose(0, 1).cpu().numpy()
        if wav_np.shape[1] == 1:
            wav_np = wav_np[:, 0]
        # Prefer soundfile.write to avoid torchcodec dependency for torchaudio.save
        sf.write(dst, wav_np, sr, subtype="PCM_16")

mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=16000,
    n_fft=400,          
    win_length=400,
    hop_length=160,
    n_mels=80,
    power=2.0
)
db_transform = torchaudio.transforms.AmplitudeToDB(stype="power")


def cmvn(feat):
    mean = feat.mean(axis=1, keepdims=True)
    std  = feat.std(axis=1, keepdims=True) + 1e-9
    return (feat - mean) / std

def extract_features():
    wav_files = [f for f in os.listdir(WAV_DIR) if f.endswith(".wav")]
    for fname in tqdm(wav_files, desc="Extracting log-Mel features"):
        wav_path = os.path.join(WAV_DIR, fname)
        out_path = os.path.join(FEAT_DIR, fname.replace(".wav", ".npy"))
        if os.path.exists(out_path):
            continue
        try:
            wav_np, sr = sf.read(wav_path, dtype="float32", always_2d=True)
            wav = torch.from_numpy(wav_np.T)
        except Exception:
            wav, sr = torchaudio.load(wav_path)
        if wav.shape[0] > 1:
            wav = wav.mean(0, keepdim=True)
        mel = mel_transform(wav)
        logmel = db_transform(mel).squeeze(0).numpy()
        normed = cmvn(logmel)
        np.save(out_path, normed.astype(np.float32))

def make_manifest(split_name):
    tsv_path = os.path.join(DATA_ROOT, f"{split_name}.tsv")
    if not os.path.exists(tsv_path):
        print(f"No {split_name}.tsv found, skipping...")
        return
    try:
        df = pd.read_csv(tsv_path, sep="\t")
    except pd.errors.ParserError:
        # Fall back to the python engine with a higher field size limit for very long utterances.
        try:
            csv.field_size_limit(sys.maxsize)
        except OverflowError:
            csv.field_size_limit(int(1e9))
        df = pd.read_csv(
            tsv_path,
            sep="\t",
            engine="python",
            quoting=csv.QUOTE_NONE,
            on_bad_lines="skip",
        )
    path_col_candidates = [
        "path",
        "audio_path",
        "clip",
        "audio",
        "audio_file",
    ]
    text_col_candidates = [
        "sentence",
        "tgt_text",
        "target_text",
        "transcript",
        "normalized_text",
    ]

    path_col = next((col for col in path_col_candidates if col in df.columns), None)
    text_col = next((col for col in text_col_candidates if col in df.columns), None)

    if path_col is None:
        print(
            f"Skipping {split_name}: couldn’t find any of {path_col_candidates} in columns {list(df.columns)}"
        )
        return

    if text_col is None:
        print(
            f"Skipping {split_name}: couldn’t find any of {text_col_candidates} in columns {list(df.columns)}"
        )
        return
    out_path = os.path.join(MANIFESTS, f"{split_name}.tsv")

    with open(out_path, "w", newline='', encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["id", "n_frames", "tgt_text"])

        for _, row in df.iterrows():
            clip_value = row[path_col]
            if pd.isna(clip_value) or not isinstance(clip_value, str):
                continue
            clip_name = os.path.splitext(os.path.basename(clip_value))[0]
            feat_file = os.path.join(FEAT_DIR, clip_name + ".npy")
            if not os.path.exists(feat_file):
                continue
            feat = np.load(feat_file)
            text_value = row[text_col]
            tgt_text = "" if pd.isna(text_value) else str(text_value)
            writer.writerow([feat_file, feat.shape[1], tgt_text])
    print(f"Manifest saved: {out_path}")

if __name__ == "__main__":
    print("Step 1: Converting audio")
    convert_mp3_to_wav()
    print("Step 2: Extracting features")
    extract_features()
    print("Step 3: Generating manifests")
    for split in ["train", "dev", "test", "validated"]:
        make_manifest(split)
    print("Features and manifests ready.")
