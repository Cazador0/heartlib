"""Claude-as-judge over a Mel spectrogram + numerical metrics.

Uses Haiku 4.5 (cheapest current vision-capable Claude model: $1/$5 per 1M).
Output is constrained to a small JSON schema via output_config.format so the
caller can ingest results programmatically.

Prompt-caching note: Haiku 4.5's minimum cacheable prefix is 4096 tokens
(see Anthropic prompt-caching docs). This judge's system prompt is small
(~500 tokens), so cache_control would silently no-op. Caching is therefore
not enabled. If we later batch hundreds of tracks per run, we can pad the
system prompt with a richer rubric to clear the threshold and add caching
on the system block.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path

import anthropic

MODEL = "claude-haiku-4-5"

SYSTEM_PROMPT = (
    "You are an audio engineer evaluating whether vocals have been removed "
    "cleanly from a song while preserving the instrumental. You will receive "
    "(1) a Mel spectrogram of the candidate instrumental track and "
    "(2) numerical metrics. Return a terse, structured JSON verdict."
)

USER_TEMPLATE = (
    "Metrics for this track:\n"
    "- vocal_leakage_dbfs: {leakage:.2f}  (goal: <= -60.0; lower is better)\n"
    "- si_sdr_codec_db: {sdr}  (codec roundtrip fidelity; null if not run)\n"
    "- duration_s: {duration:.2f}\n\n"
    "Inspect the spectrogram for residual vocal formants (typically 200 Hz - 4 kHz, "
    "horizontal harmonic stacks with vibrato), pumping/dropouts where vocals were, "
    "and overall instrumental integrity. Be concise."
)

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {
            "type": "string",
            "enum": ["clean", "minor_residue", "audible_residue", "broken"],
        },
        "vocal_residue_audible": {"type": "boolean"},
        "spectrogram_observations": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 4,
        },
        "recommendation": {"type": "string"},
    },
    "required": [
        "verdict",
        "vocal_residue_audible",
        "spectrogram_observations",
        "recommendation",
    ],
    "additionalProperties": False,
}


def judge(
    spectrogram_png: Path,
    *,
    leakage_dbfs: float,
    duration_s: float,
    si_sdr_db: float | None = None,
    model: str = MODEL,
) -> dict:
    if "ANTHROPIC_API_KEY" not in os.environ:
        raise RuntimeError("ANTHROPIC_API_KEY not set in environment")

    client = anthropic.Anthropic()
    img_b64 = base64.standard_b64encode(spectrogram_png.read_bytes()).decode()

    response = client.messages.create(
        model=model,
        max_tokens=512,
        system=SYSTEM_PROMPT,
        output_config={"format": {"type": "json_schema", "schema": OUTPUT_SCHEMA}},
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": img_b64,
                        },
                    },
                    {
                        "type": "text",
                        "text": USER_TEMPLATE.format(
                            leakage=leakage_dbfs,
                            sdr="null" if si_sdr_db is None else f"{si_sdr_db:.2f}",
                            duration=duration_s,
                        ),
                    },
                ],
            }
        ],
    )

    text = next(b.text for b in response.content if b.type == "text")
    return json.loads(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spectrogram", type=Path, help="PNG written by evaluate.spectrogram_png")
    parser.add_argument("--leakage-dbfs", type=float, required=True)
    parser.add_argument("--duration-s", type=float, required=True)
    parser.add_argument("--si-sdr-db", type=float, default=None)
    args = parser.parse_args()

    verdict = judge(
        args.spectrogram,
        leakage_dbfs=args.leakage_dbfs,
        duration_s=args.duration_s,
        si_sdr_db=args.si_sdr_db,
    )
    print(json.dumps(verdict, indent=2))


if __name__ == "__main__":
    main()
