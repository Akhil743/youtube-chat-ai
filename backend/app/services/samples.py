"""Bundled transcripts for the demo's sample videos (all CC BY on YouTube).

Shipping them means the sample chips work even when YouTube blocks the
server's IP, and they're pre-indexed at startup so the first question is fast.
Attribution is in the README and in each JSON file.
"""

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from app.services.transcript import TranscriptResult, build_transcript_from_raw

SAMPLES_DIR = Path(__file__).resolve().parent.parent / "samples"


@dataclass(frozen=True)
class Sample:
    video_id: str
    title: str
    channel: str
    url: str
    duration_seconds: int
    license: str
    transcript: TranscriptResult


@lru_cache(maxsize=1)
def load_samples() -> dict[str, Sample]:
    samples = {}
    for path in sorted(SAMPLES_DIR.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        vid = data["video_id"]
        samples[vid] = Sample(
            video_id=vid,
            title=data["title"],
            channel=data["channel"],
            url=data["url"],
            duration_seconds=int(data["duration_seconds"]),
            license=data["license"],
            transcript=build_transcript_from_raw(vid, data["segments"], data.get("language", "en")),
        )
    return samples


def get_sample(video_id: str) -> Sample | None:
    return load_samples().get(video_id)
