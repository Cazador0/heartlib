"""HeartCodec encode -> decode roundtrip on an instrumental track.

Uses HeartCodec's ScalarModel autoencoder (heartcodec.scalar_model.encode /
.decode). Reports SI-SDR between input and reconstruction so we can quantify
how transparent the 12.5 Hz codec is on real-world non-vocal audio.

Checkpoint expected at ./ckpt/HeartCodec-oss/ per the project README.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

from heartlib.heartcodec.modeling_heartcodec import HeartCodec

from research.evaluate import si_sdr


def _device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def roundtrip(input_path: Path, ckpt_path: Path, output_dir: Path) -> tuple[Path, float]:
    device = _device()
    model = HeartCodec.from_pretrained(str(ckpt_path)).to(device).eval()
    sr_target = int(model.sample_rate)

    wav, sr = torchaudio.load(str(input_path))
    if sr != sr_target:
        wav = torchaudio.functional.resample(wav, sr, sr_target)
    mono = wav.mean(dim=0, keepdim=True).unsqueeze(0).to(device)

    with torch.no_grad():
        emb = model.scalar_model.encode(mono)
        recon = model.scalar_model.decode(emb)

    recon_np = recon.squeeze(0).squeeze(0).cpu().numpy()
    src_np = mono.squeeze(0).squeeze(0).cpu().numpy()
    sdr = si_sdr(src_np, recon_np)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{input_path.stem}_codec_roundtrip.flac"
    sf.write(str(out_path), recon_np.astype(np.float32), sr_target, format="FLAC")
    return out_path, sdr


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Instrumental FLAC/WAV")
    parser.add_argument(
        "--ckpt",
        type=Path,
        default=Path("ckpt/HeartCodec-oss"),
        help="HeartCodec checkpoint dir",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("research/output"))
    args = parser.parse_args()

    out, sdr = roundtrip(args.input, args.ckpt, args.output_dir)
    print(f"Wrote: {out}")
    print(f"SI-SDR (input vs reconstruction): {sdr:.2f} dB")


if __name__ == "__main__":
    main()
