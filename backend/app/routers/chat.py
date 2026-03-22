from fastapi import APIRouter, HTTPException, Request

from app.models.schemas import (
    LoadVideoRequest, LoadVideoResponse,
    ChatRequest, ChatResponse, TimestampRef,
)
from app.services.transcript import extract_video_id, fetch_transcript
from app.services.rag import rag_service
from app.limiter import limiter

router = APIRouter(prefix="/api", tags=["chat"])


@router.post("/load-video", response_model=LoadVideoResponse)
@limiter.limit("10/minute")
async def load_video(request: Request, body: LoadVideoRequest):
    try:
        video_id = extract_video_id(body.url)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if store := rag_service.get_video_store(video_id):
        return LoadVideoResponse(
            video_id=video_id, title=store.title,
            language=store.language, chunk_count=store.chunk_count,
            message="Video already loaded and ready for questions.",
        )

    try:
        transcript = fetch_transcript(video_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch transcript: {e}")

    try:
        store = await rag_service.ingest_transcript(transcript)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process transcript: {e}")

    return LoadVideoResponse(
        video_id=video_id, title=store.title,
        language=store.language, chunk_count=store.chunk_count,
        message="Video loaded successfully. Ask your questions!",
    )


@router.post("/chat", response_model=ChatResponse)
@limiter.limit("30/minute")
async def chat(request: Request, body: ChatRequest):
    if not rag_service.is_video_loaded(body.video_id):
        raise HTTPException(status_code=404, detail="Video not loaded. Please load the video first.")

    try:
        answer, sources = await rag_service.query(
            video_id=body.video_id,
            question=body.question,
            chat_history=body.chat_history,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate answer: {e}")

    return ChatResponse(
        answer=answer,
        sources=[TimestampRef(text=s["text"], start_seconds=s["start_seconds"]) for s in sources],
        video_id=body.video_id,
    )
