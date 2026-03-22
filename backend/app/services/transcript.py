import re
import json
import logging
from dataclasses import dataclass
from typing import Optional

import requests as http_requests
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter

log = logging.getLogger(__name__)


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

_BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


def extract_video_id(url: str) -> str:
    match = _VIDEO_ID_RE.search(url)
    if not match:
        raise ValueError(f"Could not extract video ID from: {url}")
    return match.group(1)


def _fetch_via_library(video_id: str):
    """Try the youtube-transcript-api library."""
    try:
        return _ytt.fetch(video_id, languages=["en"])
    except Exception as e:
        log.warning(f"Library fetch (en) failed: {e}")

    try:
        return _ytt.fetch(video_id)
    except Exception as e:
        log.warning(f"Library fetch (any) failed: {e}")

    try:
        for t in _ytt.list(video_id):
            return t.fetch()
    except Exception as e:
        log.warning(f"Library list failed: {e}")

    return None


def _fetch_via_raw_http(video_id: str) -> Optional[list[dict]]:
    """Manually fetch transcript by scraping YouTube page with browser headers."""
    try:
        resp = http_requests.get(
            f"https://www.youtube.com/watch?v={video_id}",
            headers={**_BROWSER_HEADERS, "Cookie": "CONSENT=PENDING+987;"},
            timeout=15,
        )
        resp.raise_for_status()
        html = resp.text

        match = re.search(r'"captions":.*?"captionTracks":\s*(\[.*?\])', html)
        if not match:
            log.warning("No captionTracks found in page HTML")
            return None

        tracks = json.loads(match.group(1))
        if not tracks:
            return None

        track = next((t for t in tracks if t.get("languageCode") == "en"), tracks[0])
        caption_url = track["baseUrl"]

        caption_resp = http_requests.get(
            caption_url,
            headers=_BROWSER_HEADERS,
            params={"fmt": "json3"},
            timeout=15,
        )
        caption_resp.raise_for_status()
        data = caption_resp.json()

        if "events" not in data:
            return None

        segments = []
        for event in data["events"]:
            if "segs" not in event:
                continue
            text = "".join(s.get("utf8", "") for s in event["segs"]).strip()
            if text:
                segments.append({
                    "text": text,
                    "start": (event.get("tStartMs", 0)) / 1000.0,
                    "duration": (event.get("dDurationMs", 0)) / 1000.0,
                })
        return segments if segments else None
    except Exception as e:
        log.warning(f"Raw HTTP transcript fetch failed: {e}")
        return None


def fetch_transcript(video_id: str) -> TranscriptResult:
    # method 1: youtube-transcript-api library
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

    # method 2: raw HTTP scrape with browser headers
    segments = _fetch_via_raw_http(video_id)
    if segments:
        return build_transcript_from_raw(video_id, segments)

    raise ValueError(f"No transcript available for video {video_id}")


def build_transcript_from_raw(video_id: str, segments: list[dict]) -> TranscriptResult:
    """Build TranscriptResult from raw segment dicts: {"text", "start", "duration"}"""
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
