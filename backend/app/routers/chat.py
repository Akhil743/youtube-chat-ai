import asyncio
import logging
import os
import time

from fastapi import APIRouter, Request

from app.errors import ApiError
from app.limiter import CHAT_RATE_LIMIT, LOAD_RATE_LIMIT, daily_cap, limiter
from app.models.schemas import ChatRequest, ChatResponse, LoadVideoRequest, LoadVideoResponse
from app.services.rag import EmptyTranscript, VideoNotLoaded, rag_service
from app.services.samples import get_sample
from app.services.transcript import TranscriptUnavailable, extract_video_id, fetch_oembed, fetch_transcript

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["chat"])

MAX_TRANSCRIPT_CHARS = int(os.getenv("MAX_TRANSCRIPT_CHARS", "300000"))  # roughly 4-5 hours of speech


def _daily_cap_error() -> ApiError:
    return ApiError(429, "DAILY_CAP", "This demo has reached its daily usage budget. It resets at 00:00 UTC. Please try again tomorrow.")


def _load_response(store, already_loaded: bool) -> LoadVideoResponse:
    return LoadVideoResponse(
        video_id=store.video_id,
        title=store.title,
        channel=store.channel,
        duration_seconds=store.duration_seconds,
        language=store.language,
        chunk_count=store.chunk_count,
        is_sample=store.is_sample,
        already_loaded=already_loaded,
        message="Video ready. Ask your questions!",
        usage=store.ingest_usage.__dict__,
    )


async def ensure_sample_loaded(video_id: str):
    sample = get_sample(video_id)
    return await rag_service.ingest(
        sample.transcript, title=sample.title, channel=sample.channel,
        duration_seconds=sample.duration_seconds, is_sample=True,
    )


@router.post("/load-video", response_model=LoadVideoResponse)
@limiter.limit(LOAD_RATE_LIMIT)
async def load_video(request: Request, body: LoadVideoRequest):
    video_id = extract_video_id(body.url)

    if store := rag_service.get_video_store(video_id):
        return _load_response(store, already_loaded=True)

    try:
        if get_sample(video_id):
            # Bundled transcript: no YouTube call, not counted against the daily cap.
            return _load_response(await ensure_sample_loaded(video_id), already_loaded=False)

        if not daily_cap.try_consume():
            raise _daily_cap_error()

        meta_task = asyncio.create_task(asyncio.to_thread(fetch_oembed, video_id))
        try:
            transcript = await asyncio.to_thread(fetch_transcript, video_id)
        except TranscriptUnavailable:
            meta_task.cancel()
            raise ApiError(
                404, "TRANSCRIPT_UNAVAILABLE",
                "Couldn't get captions for this video. It may have none, be private or age-restricted, "
                "or YouTube may be blocking the server right now. Try one of the sample videos.",
            )
        if transcript.char_count > MAX_TRANSCRIPT_CHARS:
            meta_task.cancel()
            raise ApiError(413, "VIDEO_TOO_LONG", "This video's transcript is too long for the demo. Try a shorter video.")

        meta = await meta_task or {}
        store = await rag_service.ingest(transcript, title=meta.get("title", ""), channel=meta.get("channel") or None)
        return _load_response(store, already_loaded=False)
    except ApiError:
        raise
    except EmptyTranscript:
        raise ApiError(404, "TRANSCRIPT_UNAVAILABLE", "This video's captions are empty, so there's nothing to search.")
    except (asyncio.TimeoutError, TimeoutError):
        log.warning("Timed out indexing %s", video_id)
        raise ApiError(504, "TIMEOUT", "Indexing the video took too long. Please try again.")
    except Exception:
        log.exception("Failed to load video %s", video_id)
        raise ApiError(502, "INDEXING_FAILED", "Couldn't index this video right now. Please try again in a moment.")


@router.post("/chat", response_model=ChatResponse)
@limiter.limit(CHAT_RATE_LIMIT)
async def chat(request: Request, body: ChatRequest):
    if not rag_service.is_video_loaded(body.video_id):
        raise ApiError(404, "VIDEO_NOT_LOADED", "This video isn't loaded on the server (it may have restarted). Load it again.")
    if not daily_cap.try_consume():
        raise _daily_cap_error()

    t0 = time.perf_counter()
    try:
        result = await rag_service.answer(
            video_id=body.video_id,
            question=body.question,
            history=[t.model_dump() for t in body.chat_history],
        )
    except VideoNotLoaded:
        raise ApiError(404, "VIDEO_NOT_LOADED", "This video isn't loaded on the server (it may have restarted). Load it again.")
    except (asyncio.TimeoutError, TimeoutError):
        log.warning("LLM timed out for %s after %.1fs", body.video_id, time.perf_counter() - t0)
        raise ApiError(504, "TIMEOUT", "The model took too long to answer. Please try again.")
    except Exception as e:
        log.exception("Answer generation failed for %s", body.video_id)
        if "ResourceExhausted" in type(e).__name__ or "429" in str(e)[:200]:
            raise ApiError(503, "MODEL_BUSY", "The AI model is rate-limited right now. Please try again in a minute.")
        raise ApiError(502, "MODEL_ERROR", "The AI model didn't return an answer. Please try again.")

    check = result.citation_check
    return ChatResponse(
        answer=result.answer,
        refused=result.refused,
        sources=result.sources,
        citation_check={"cited": len(check.cited), "valid": len(check.valid), "removed": check.removed},
        usage=result.usage,
        video_id=body.video_id,
    )
