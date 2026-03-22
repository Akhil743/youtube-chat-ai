import re
import logging
from dataclasses import dataclass

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


def extract_video_id(url: str) -> str:
    match = _VIDEO_ID_RE.search(url)
    if not match:
        raise ValueError(f"Could not extract video ID from: {url}")
    return match.group(1)


def fetch_transcript(video_id: str) -> TranscriptResult:
    transcript = None
    last_error = None

    # try english first
    try:
        transcript = _ytt.fetch(video_id, languages=["en"])
    except Exception as e:
        last_error = e
        log.warning(f"English transcript failed for {video_id}: {e}")

    # try any available language
    if transcript is None:
        try:
            transcript = _ytt.fetch(video_id)
        except Exception as e:
            last_error = e
            log.warning(f"Any-language transcript failed for {video_id}: {e}")

    # try listing and fetching first available
    if transcript is None:
        try:
            for t in _ytt.list(video_id):
                transcript = t.fetch()
                break
        except Exception as e:
            last_error = e
            log.warning(f"List transcripts failed for {video_id}: {e}")

    if transcript is None:
        error_msg = f"No transcript available for video {video_id}"
        if last_error:
            error_msg += f" ({type(last_error).__name__}: {last_error})"
        raise ValueError(error_msg)

    chunks = [
        TranscriptChunk(
            text=snippet.text,
            start_seconds=snippet.start,
            end_seconds=snippet.start + snippet.duration,
        )
        for snippet in transcript
    ]

    return TranscriptResult(
        video_id=video_id,
        language=transcript.language_code,
        chunks=chunks,
        full_text=_formatter.format_transcript(transcript),
    )


def build_transcript_from_raw(video_id: str, segments: list[dict]) -> TranscriptResult:
    """Build a TranscriptResult from client-provided transcript segments.
    Each segment: {"text": "...", "start": 0.0, "duration": 5.0}
    """
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
