# research/ — instrumental extraction + codec roundtrip

Vocal-free audio replication research track. Two stages:

- **Option A — extract_instrumental.py** — Demucs `htdemucs_ft` residual method.
  `instrumental = mixture - predicted_vocals`. The original drums/bass/other
  waveforms are preserved verbatim; only the predicted vocals are subtracted.
- **Option C — codec_roundtrip.py** — encode the Option-A instrumental through
  HeartCodec's `ScalarModel` autoencoder and decode. Reports SI-SDR to quantify
  codec transparency.

Both stages are exercised by `evaluate.py` (SI-SDR, vocal-leakage dBFS,
spectrogram PNG) and `claude_judge.py` (Haiku 4.5 qualitative review).

## Install

```bash
pip install -e ".[research]"
```

`ffmpeg` is required for the optional MP3 export step.

## Test data — MUSDB18

MUSDB18 is the standard music-source-separation benchmark. It ships with
ground-truth stems, so we can compute true SI-SDR and not just leakage.

1. Download MUSDB18-HQ from <https://sigsep.github.io/datasets/musdb.html>.
2. Place a track's `mixture.wav` at e.g. `research/input/AlGreen_LetsStayTogether/mixture.wav`.

If you only have a single mix file (no stems), the leakage metric still works
but you won't get a true SDR number against the ground-truth instrumental.

## Run — Option A

```bash
python -m research.extract_instrumental \
    research/input/<track>/mixture.wav \
    --output-dir research/output
```

Outputs `<track>_instrumental.flac` (and `.mp3` if `ffmpeg` is available).

## Run — Option C

Requires the HeartCodec checkpoint (see project README):

```bash
hf download --local-dir './ckpt/HeartCodec-oss' HeartMuLa/HeartCodec-oss-20260123

python -m research.codec_roundtrip \
    research/output/<track>_instrumental.flac \
    --ckpt ./ckpt/HeartCodec-oss \
    --output-dir research/output
```

Prints SI-SDR (input vs reconstruction).

## Evaluate

```python
from pathlib import Path
from research.evaluate import vocal_leakage_dbfs, spectrogram_png

leakage = vocal_leakage_dbfs(Path("research/output/<track>_instrumental.flac"))
print(f"vocal leakage: {leakage:.2f} dBFS")  # goal: <= -60

spectrogram_png(
    Path("research/output/<track>_instrumental.flac"),
    Path("research/output/<track>_instrumental.png"),
)
```

## Claude judge (optional)

Cost: ~$0.001 per call on Haiku 4.5.

```bash
export ANTHROPIC_API_KEY=...
python -m research.claude_judge \
    research/output/<track>_instrumental.png \
    --leakage-dbfs -62.4 \
    --duration-s 240.0
```

Returns structured JSON: `verdict`, `vocal_residue_audible`,
`spectrogram_observations`, `recommendation`.

## Tests

```bash
pytest tests/research
```

Synthetic-audio unit tests for `si_sdr` / `rms_dbfs`. The Demucs and
matplotlib paths are exercised by integration runs, not unit tests.
