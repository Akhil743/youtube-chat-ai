"""Fetching transcripts and metadata for a YouTube video.

All functions here are blocking (network I/O); call them via asyncio.to_thread.

Order of attempts for a transcript:
1. Bundled sample transcripts (app/samples/) - always work, even when YouTube
   blocks the server's IP.
2. youtube-transcript-api from the server (works locally; often blocked on
   cloud IPs).
3. The Cloudflare Worker in transcript-worker/ as a fallback proxy.
"""

import logging
import os
import re
from dataclasses import dataclass
from typing import Optional
from urllib.parse import parse_qs, urlparse

import requests as http_requests
from youtube_transcript_api import YouTubeTranscriptApi

log = logging.getLogger(__name__)

TRANSCRIPT_WORKER_URL = os.getenv(
    "TRANSCRIPT_WORKER_URL",
    "https://yt-transcript-proxy.akhilrajsanthosh.workers.dev/",
)
HTTP_TIMEOUT_S = float(os.getenv("TRANSCRIPT_HTTP_TIMEOUT_S", "20"))


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

    @property
    def char_count(self) -> int:
        return sum(len(c.text) for c in self.chunks)

    @property
    def end_seconds(self) -> float:
        return max((c.end_seconds for c in self.chunks), default=0.0)


class TranscriptUnavailable(Exception):
    pass


_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")
_YT_HOSTS = {
    "youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com",
    "youtube-nocookie.com", "www.youtube-nocookie.com",
}
_PATH_PREFIXES = ("embed", "shorts", "live", "v", "e")


def extract_video_id(url: str) -> str:
    """Return the 11-character video ID from a YouTube URL (or a bare ID).

    Accepts watch?v=, youtu.be/, /embed/, /shorts/, /live/, /v/ URLs on
    youtube.com, m.youtube.com, music.youtube.com and youtube-nocookie.com.
    """
    raw = (url or "").strip()
    if _ID_RE.match(raw):
        return raw
    if "://" not in raw:
        raw = "https://" + raw
    parsed = urlparse(raw)
    host = (parsed.hostname or "").lower()
    candidate = None
    if host in ("youtu.be", "www.youtu.be"):
        candidate = parsed.path.lstrip("/").split("/")[0]
    elif host in _YT_HOSTS:
        qs = parse_qs(parsed.query)
        if parsed.path in ("/watch", "/watch/") and qs.get("v"):
            candidate = qs["v"][0]
        else:
            parts = [p for p in parsed.path.split("/") if p]
            if len(parts) >= 2 and parts[0] in _PATH_PREFIXES:
                candidate = parts[1]
    if candidate and _ID_RE.match(candidate):
        return candidate
    raise ValueError("Could not find a YouTube video ID in that link")


def _fetch_via_library(video_id: str) -> Optional[TranscriptResult]:
    ytt = YouTubeTranscriptApi()
    for langs in (["en", "en-US", "en-GB"], None):
        try:
            fetched = ytt.fetch(video_id, languages=langs) if langs else _fetch_any(ytt, video_id)
            if fetched is None:
                continue
            chunks = [
                TranscriptChunk(text=s.text, start_seconds=s.start, end_seconds=s.start + s.duration)
                for s in fetched
            ]
            return TranscriptResult(video_id=video_id, language=fetched.language_code, chunks=chunks)
        except Exception as e:  # library raises many specific types; all mean "try next"
            log.warning("Transcript library fetch failed for %s (langs=%s): %s", video_id, langs, type(e).__name__)
    return None


def _fetch_any(ytt: YouTubeTranscriptApi, video_id: str):
    """First available transcript in any language (manual preferred)."""
    transcripts = list(ytt.list(video_id))
    transcripts.sort(key=lambda t: t.is_generated)
    return transcripts[0].fetch() if transcripts else None


def _fetch_via_worker(video_id: str) -> Optional[TranscriptResult]:
    try:
        resp = http_requests.post(TRANSCRIPT_WORKER_URL, json={"videoId": video_id}, timeout=HTTP_TIMEOUT_S)
        if resp.status_code != 200:
            log.warning("Transcript worker returned %s for %s", resp.status_code, video_id)
            return None
        segments = resp.json().get("segments") or []
        return build_transcript_from_raw(video_id, segments) if segments else None
    except Exception as e:
        log.warning("Transcript worker fetch failed for %s: %s", video_id, type(e).__name__)
        return None


def fetch_transcript(video_id: str) -> TranscriptResult:
    """Blocking. Library first, then the Cloudflare Worker fallback."""
    result = _fetch_via_library(video_id)
    if result is None:
        result = _fetch_via_worker(video_id)
        if result is not None:
            log.info("Transcript for %s fetched via worker fallback", video_id)
    if result is None or not result.chunks:
        raise TranscriptUnavailable(video_id)
    return result


def build_transcript_from_raw(video_id: str, segments: list[dict], language: str = "en") -> TranscriptResult:
    chunks = []
    for s in segments:
        text = str(s.get("text", "")).strip()
        if not text:
            continue
        start = float(s.get("start", 0) or 0)
        duration = float(s.get("duration", 0) or 0)
        chunks.append(TranscriptChunk(text=text, start_seconds=start, end_seconds=start + max(duration, 0)))
    return TranscriptResult(video_id=video_id, language=language, chunks=chunks)


def fetch_oembed(video_id: str) -> Optional[dict]:
    """Blocking. Title and channel name via YouTube's public oEmbed endpoint."""
    try:
        resp = http_requests.get(
            "https://www.youtube.com/oembed",
            params={"url": f"https://www.youtube.com/watch?v={video_id}", "format": "json"},
            timeout=5,
        )
        if resp.status_code != 200:
            return None
        data = resp.json()
        return {"title": str(data.get("title", ""))[:200], "channel": str(data.get("author_name", ""))[:120]}
    except Exception as e:
        log.info("oEmbed lookup failed for %s: %s", video_id, type(e).__name__)
        return None
