"""Synthetic-audio tests for evaluate.si_sdr and evaluate.rms_dbfs.

Avoids any model loads or audio-file IO so the suite stays fast and
hermetic. Demucs / matplotlib paths are exercised via integration runs,
not unit tests.
"""
from __future__ import annotations

import numpy as np

from research.evaluate import rms_dbfs, si_sdr


def test_si_sdr_perfect_reconstruction_is_huge() -> None:
    rng = np.random.default_rng(0)
    ref = rng.standard_normal(48000).astype(np.float32)
    assert si_sdr(ref, ref) > 100.0


def test_si_sdr_scale_invariance() -> None:
    rng = np.random.default_rng(1)
    ref = rng.standard_normal(48000).astype(np.float32)
    scaled = ref * 3.7
    assert si_sdr(ref, scaled) > 100.0


def test_si_sdr_pure_noise_is_low() -> None:
    rng = np.random.default_rng(2)
    ref = rng.standard_normal(48000).astype(np.float32)
    noise = rng.standard_normal(48000).astype(np.float32)
    assert si_sdr(ref, noise) < 5.0


def test_si_sdr_signal_plus_noise_decreases_monotonically() -> None:
    rng = np.random.default_rng(3)
    ref = rng.standard_normal(48000).astype(np.float32)
    noise = rng.standard_normal(48000).astype(np.float32)
    high_snr = ref + 0.01 * noise
    low_snr = ref + 0.5 * noise
    assert si_sdr(ref, high_snr) > si_sdr(ref, low_snr)


def test_rms_dbfs_full_scale_sine_is_near_minus_3() -> None:
    t = np.linspace(0, 1, 48000, endpoint=False)
    full_scale_sine = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    db = rms_dbfs(full_scale_sine)
    assert -3.5 < db < -2.5


def test_rms_dbfs_silence_is_very_negative() -> None:
    silence = np.zeros(48000, dtype=np.float32)
    assert rms_dbfs(silence) < -150.0
