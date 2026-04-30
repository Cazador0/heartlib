"""Vocal removal via Demucs htdemucs_ft using the residual method.

instrumental = mixture - predicted_vocals

Pushes any model error into the discarded vocal stem, preserving maximum
fidelity in the kept signal. Used by top Music Demixing Challenge entries.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

import soundfile as sf
import torch
import torchaudio
from demucs.apply import apply_model
from demucs.pretrained import get_model


def _device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def extract(input_path: Path, output_dir: Path, *, mp3: bool = True) -> Path:
    device = _device()
    model = get_model("htdemucs_ft").to(device).eval()

    wav, sr = torchaudio.load(str(input_path))
    if sr != model.samplerate:
        wav = torchaudio.functional.resample(wav, sr, model.samplerate)
        sr = model.samplerate
    if wav.shape[0] == 1:
        wav = wav.repeat(2, 1)

    with torch.no_grad():
        sources = apply_model(
            model,
            wav[None].to(device),
            split=True,
            overlap=0.25,
            progress=True,
        )[0].cpu().numpy()

    vocals_idx = model.sources.index("vocals")
    instrumental = wav.numpy() - sources[vocals_idx]

    output_dir.mkdir(parents=True, exist_ok=True)
    flac_path = output_dir / f"{input_path.stem}_instrumental.flac"
    sf.write(str(flac_path), instrumental.T, sr, format="FLAC")

    if mp3:
        if shutil.which("ffmpeg") is None:
            print("ffmpeg not found on PATH; skipping MP3 conversion")
        else:
            mp3_path = flac_path.with_suffix(".mp3")
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(flac_path), "-q:a", "2", str(mp3_path)],
                check=True,
                capture_output=True,
            )

    return flac_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Path to mixture audio file")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("research/output"),
    )
    parser.add_argument("--no-mp3", action="store_true")
    args = parser.parse_args()

    out = extract(args.input, args.output_dir, mp3=not args.no_mp3)
    print(f"Wrote: {out}")


if __name__ == "__main__":
    main()
