from pydantic import BaseModel, Field, field_validator
from typing import Any, Optional
import re


class TranscriptSegment(BaseModel):
    text: str
    start: float
    duration: float = 0.0


class LoadVideoRequest(BaseModel):
    url: str = Field(..., description="YouTube video URL")
    transcript: Optional[list[TranscriptSegment]] = Field(
        None, description="Client-provided transcript segments (fallback if server can't fetch)"
    )

    @field_validator("url")
    @classmethod
    def validate_youtube_url(cls, v: str) -> str:
        patterns = [
            r"(?:https?://)?(?:www\.)?youtube\.com/watch\?v=[\w-]+",
            r"(?:https?://)?youtu\.be/[\w-]+",
            r"(?:https?://)?(?:www\.)?youtube\.com/embed/[\w-]+",
            r"(?:https?://)?(?:www\.)?youtube\.com/shorts/[\w-]+",
        ]
        if not any(re.match(p, v) for p in patterns):
            raise ValueError("Invalid YouTube URL")
        return v


class LoadVideoResponse(BaseModel):
    video_id: str
    title: str
    duration_seconds: Optional[int] = None
    language: str
    chunk_count: int
    message: str
    quota: Optional[dict[str, Any]] = None


class ChatRequest(BaseModel):
    video_id: str
    question: str = Field(..., min_length=1, max_length=1000)
    chat_history: list[dict] = Field(default_factory=list)


class TimestampRef(BaseModel):
    text: str
    start_seconds: float


class ChatResponse(BaseModel):
    answer: str
    sources: list[TimestampRef] = Field(default_factory=list)
    video_id: str
    quota: Optional[dict[str, Any]] = None


class HealthResponse(BaseModel):
    status: str
    videos_cached: int
