import os
import re
import json
import logging
from dataclasses import dataclass
from typing import Optional

import requests as http_requests
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter

log = logging.getLogger(__name__)

TRANSCRIPT_WORKER_URL = os.getenv(
    "TRANSCRIPT_WORKER_URL",
    "https://yt-transcript-proxy.akhilrajsanthosh.workers.dev/",
)


@dataclass
class TranscriptChunk:
    text: str
    start_seconds: float
    end_seconds: float


@dataclass
class TranscriptResult:
    video_id: str
    language: str
    chunks: list[TranscriptChunk]
    full_text: str


_ytt = YouTubeTranscriptApi()
_formatter = TextFormatter()

_VIDEO_ID_RE = re.compile(
    r"(?:v=|/v/|youtu\.be/|/embed/|/shorts/)([a-zA-Z0-9_-]{11})"
)


def extract_video_id(url: str) -> str:
    match = _VIDEO_ID_RE.search(url)
    if not match:
        raise ValueError(f"Could not extract video ID from: {url}")
    return match.group(1)


def _fetch_via_library(video_id: str):
    try:
        return _ytt.fetch(video_id, languages=["en"])
    except Exception as e:
        log.warning(f"Library fetch (en) failed: {e}")
    try:
        return _ytt.fetch(video_id)
    except Exception as e:
        log.warning(f"Library fetch (any) failed: {e}")
    return None


def _fetch_via_worker(video_id: str) -> Optional[list[dict]]:
    """Fetch transcript via Cloudflare Worker proxy."""
    try:
        resp = http_requests.post(
            TRANSCRIPT_WORKER_URL,
            json={"videoId": video_id},
            timeout=30,
        )
        if resp.status_code != 200:
            log.warning(f"Worker returned {resp.status_code}: {resp.text[:200]}")
            return None
        data = resp.json()
        return data.get("segments")
    except Exception as e:
        log.warning(f"Worker fetch failed: {e}")
        return None


def fetch_transcript(video_id: str) -> TranscriptResult:
    # method 1: youtube-transcript-api library (works locally / non-cloud IPs)
    transcript_obj = _fetch_via_library(video_id)
    if transcript_obj is not None:
        chunks = [
            TranscriptChunk(
                text=snippet.text,
                start_seconds=snippet.start,
                end_seconds=snippet.start + snippet.duration,
            )
            for snippet in transcript_obj
        ]
        return TranscriptResult(
            video_id=video_id,
            language=transcript_obj.language_code,
            chunks=chunks,
            full_text=_formatter.format_transcript(transcript_obj),
        )

    # method 2: Cloudflare Worker proxy (for cloud deployments)
    segments = _fetch_via_worker(video_id)
    if segments:
        log.info(f"Worker method succeeded for {video_id}")
        return build_transcript_from_raw(video_id, segments)

    raise ValueError(f"No transcript available for video {video_id}")


def build_transcript_from_raw(video_id: str, segments: list[dict]) -> TranscriptResult:
    chunks = [
        TranscriptChunk(
            text=s["text"],
            start_seconds=s["start"],
            end_seconds=s["start"] + s.get("duration", 0),
        )
        for s in segments
    ]
    full_text = " ".join(s["text"] for s in segments)
    return TranscriptResult(
        video_id=video_id,
        language="en",
        chunks=chunks,
        full_text=full_text,
    )
