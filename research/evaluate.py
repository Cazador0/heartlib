"""Fidelity metrics for instrumental extraction.

- si_sdr: scale-invariant SDR in dB (when ground truth available)
- vocal_leakage_dbfs: re-runs separation on the output, returns RMS of the
  detected vocal stem in dBFS. Goal: <= -60 dBFS.
- spectrogram_png: writes a Mel spectrogram PNG for visual / Claude-judge input.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

EPS = 1e-9


def si_sdr(reference: np.ndarray, estimate: np.ndarray) -> float:
    """Scale-invariant SDR in dB."""
    ref = np.asarray(reference, dtype=np.float64).flatten()
    est = np.asarray(estimate, dtype=np.float64).flatten()
    n = min(ref.size, est.size)
    ref, est = ref[:n], est[:n]
    alpha = float(est @ ref) / float(ref @ ref + EPS)
    target = alpha * ref
    noise = est - target
    return float(10.0 * np.log10((target @ target + EPS) / (noise @ noise + EPS)))


def rms_dbfs(audio: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(np.asarray(audio, dtype=np.float64) ** 2)))
    return float(20.0 * np.log10(rms + EPS))


def vocal_leakage_dbfs(audio_path: Path, *, device: str | None = None) -> float:
    """Re-run Demucs on the output; return RMS of vocal stem in dBFS."""
    import torch
    import torchaudio
    from demucs.apply import apply_model
    from demucs.pretrained import get_model

    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = get_model("htdemucs_ft").to(device).eval()

    wav, sr = torchaudio.load(str(audio_path))
    if sr != model.samplerate:
        wav = torchaudio.functional.resample(wav, sr, model.samplerate)
    if wav.shape[0] == 1:
        wav = wav.repeat(2, 1)

    with torch.no_grad():
        sources = apply_model(
            model, wav[None].to(device), split=True, overlap=0.25
        )[0]
    vocals_idx = model.sources.index("vocals")
    vocals = sources[vocals_idx].cpu().numpy()
    return rms_dbfs(vocals)


def spectrogram_png(audio_path: Path, png_path: Path, *, n_mels: int = 128) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch
    import torchaudio

    wav, sr = torchaudio.load(str(audio_path))
    mono = wav.mean(dim=0, keepdim=True)
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr, n_fft=2048, hop_length=512, n_mels=n_mels,
    )(mono)
    log_mel = torch.log10(mel + EPS).squeeze(0).numpy()

    fig, ax = plt.subplots(figsize=(12, 4))
    im = ax.imshow(log_mel, aspect="auto", origin="lower", cmap="magma")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Mel bin")
    ax.set_title(audio_path.name)
    fig.colorbar(im, ax=ax, label="log10 mel power")
    fig.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(png_path), dpi=100)
    plt.close(fig)
    return png_path
