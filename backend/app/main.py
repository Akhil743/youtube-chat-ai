import os
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from app.limiter import limiter
from app.routers import chat, billing
from app.services.rag import rag_service
from app.models.schemas import HealthResponse

app = FastAPI(
    title="YouTube Chatbot API",
    description="Ask questions about any YouTube video using AI",
    version="1.0.0",
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

_raw_origins = os.getenv(
    "ALLOWED_ORIGINS",
    "https://akhilrajs.com,http://localhost:8080,http://127.0.0.1:8080",
).split(",")

origins = []
for o in _raw_origins:
    o = o.strip()
    if o:
        origins.append(o)

origins_regex = None
chrome_prefixes = [o for o in origins if o.startswith("chrome-extension://")]
if chrome_prefixes:
    origins = [o for o in origins if not o.startswith("chrome-extension://")]
    origins_regex = r"^chrome-extension://.*$"

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_origin_regex=origins_regex,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router)
app.include_router(billing.router)


@app.api_route("/", methods=["GET", "HEAD"], response_model=HealthResponse)
@app.api_route("/health", methods=["GET", "HEAD"], response_model=HealthResponse)
async def health():
    return HealthResponse(status="healthy", videos_cached=len(rag_service._cache))
