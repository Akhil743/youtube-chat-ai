from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

from app.services.transcript import extract_video_id

VIDEO_ID_PATTERN = r"^[A-Za-z0-9_-]{11}$"


class LoadVideoRequest(BaseModel):
    url: str = Field(..., min_length=11, max_length=300, description="YouTube video URL or ID")

    @field_validator("url")
    @classmethod
    def validate_youtube_url(cls, v: str) -> str:
        extract_video_id(v)  # raises ValueError -> 422
        return v.strip()


class IngestUsage(BaseModel):
    model: str
    embedding_tokens: int
    tokens_estimated: bool = True
    cost_usd: Optional[float] = None
    latency_ms: int


class LoadVideoResponse(BaseModel):
    video_id: str
    title: str
    channel: Optional[str] = None
    duration_seconds: Optional[int] = None
    language: str
    chunk_count: int
    is_sample: bool = False
    already_loaded: bool = False
    message: str
    usage: IngestUsage


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., max_length=4000)


class ChatRequest(BaseModel):
    video_id: str = Field(..., pattern=VIDEO_ID_PATTERN)
    question: str = Field(..., min_length=1, max_length=1000)
    chat_history: list[ChatTurn] = Field(default_factory=list, max_length=20)

    @field_validator("question")
    @classmethod
    def strip_question(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Question is empty")
        return v


class Source(BaseModel):
    start_seconds: float
    end_seconds: float
    label: str
    snippet: str
    cited: bool
    rank: int


class CitationCheckOut(BaseModel):
    cited: int = Field(description="Timestamps the model cited")
    valid: int = Field(description="Cited timestamps inside a retrieved chunk")
    removed: list[str] = Field(default_factory=list, description="Citations dropped from the answer")


class ChatUsage(BaseModel):
    model: str
    input_tokens: int
    output_tokens: int
    embedding_model: str
    embedding_tokens: int
    cost_usd: Optional[float] = None
    latency_ms: int
    pricing_as_of: str
    usage_reported: bool = True


class ChatResponse(BaseModel):
    answer: str
    refused: bool = False
    sources: list[Source] = Field(default_factory=list)
    citation_check: CitationCheckOut
    usage: ChatUsage
    video_id: str


class HealthResponse(BaseModel):
    status: str
    videos_cached: int
    samples_ready: bool
