import asyncio
import logging
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI  # noqa: E402
from fastapi.exceptions import RequestValidationError  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from slowapi.errors import RateLimitExceeded  # noqa: E402

from app.errors import (  # noqa: E402
    ApiError, api_error_handler, rate_limit_handler, unhandled_handler, validation_handler,
)
from app.limiter import limiter  # noqa: E402
from app.models.schemas import HealthResponse  # noqa: E402
from app.routers import chat  # noqa: E402
from app.services.rag import rag_service  # noqa: E402
from app.services.samples import load_samples  # noqa: E402

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("app")

# Exact origins (comma-separated) plus any localhost / 127.0.0.1 port for local testing.
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "https://akhilrajs.com").split(",") if o.strip()]
LOCAL_ORIGIN_REGEX = r"^http://(localhost|127\.0\.0\.1)(:\d+)?$"


async def preload_samples():
    """Index the bundled sample videos so the first question on them is fast."""
    for video_id in load_samples():
        try:
            await chat.ensure_sample_loaded(video_id)
        except Exception:
            log.exception("Preloading sample %s failed", video_id)


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = None
    if os.getenv("PRELOAD_SAMPLES", "1") != "0" and os.getenv("GOOGLE_API_KEY"):
        task = asyncio.create_task(preload_samples())
    elif not os.getenv("GOOGLE_API_KEY"):
        log.warning("GOOGLE_API_KEY is not set; model calls will fail")
    app.state.preload_task = task
    yield
    if task and not task.done():
        task.cancel()


app = FastAPI(
    title="YouTube Chat AI API",
    description="Grounded Q&A over a YouTube video's transcript, with verified timestamp citations.",
    version="2.0.0",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(ApiError, api_error_handler)
app.add_exception_handler(RateLimitExceeded, rate_limit_handler)
app.add_exception_handler(RequestValidationError, validation_handler)
app.add_exception_handler(Exception, unhandled_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_origin_regex=LOCAL_ORIGIN_REGEX,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
    max_age=600,
)

app.include_router(chat.router)


@app.api_route("/", methods=["GET", "HEAD"], response_model=HealthResponse)
@app.api_route("/health", methods=["GET", "HEAD"], response_model=HealthResponse)
async def health():
    samples = load_samples()
    ready = all(rag_service.is_video_loaded(v) for v in samples)
    return HealthResponse(status="healthy", videos_cached=rag_service.cached_count, samples_ready=ready)
