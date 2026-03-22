import re
from dataclasses import dataclass

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter


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

    try:
        transcript = _ytt.fetch(video_id, languages=["en"])
    except Exception:
        try:
            for t in _ytt.list(video_id):
                transcript = t.fetch()
                break
        except Exception:
            pass

    if transcript is None:
        raise ValueError(f"No transcript available for video {video_id}")

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
