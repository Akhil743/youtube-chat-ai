"""User-facing errors. Responses carry a short message and a machine code;
exception details are logged server-side, never returned to the client."""

import logging

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from slowapi.errors import RateLimitExceeded

log = logging.getLogger("app.errors")


class ApiError(Exception):
    def __init__(self, status_code: int, code: str, message: str):
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message


def _body(code: str, message: str) -> dict:
    # "detail" keeps older clients (the Chrome extension) working.
    return {"detail": message, "code": code}


async def api_error_handler(_: Request, exc: ApiError) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content=_body(exc.code, exc.message))


async def rate_limit_handler(_: Request, exc: RateLimitExceeded) -> JSONResponse:
    return JSONResponse(
        status_code=429,
        content=_body("RATE_LIMITED", "You're sending requests too quickly. Please wait a minute and try again."),
        headers={"Retry-After": "60"},
    )


_FIELD_MESSAGES = {
    "url": "That doesn't look like a YouTube video link.",
    "question": "Questions must be between 1 and 1,000 characters.",
    "video_id": "Invalid video ID. Please load the video again.",
    "chat_history": "Conversation history is too long or malformed. Start a new chat.",
}


async def validation_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    for err in exc.errors():
        for part in err.get("loc", ()):
            if part in _FIELD_MESSAGES:
                return JSONResponse(status_code=422, content=_body("INVALID_INPUT", _FIELD_MESSAGES[part]))
    return JSONResponse(status_code=422, content=_body("INVALID_INPUT", "Invalid request."))


async def unhandled_handler(request: Request, exc: Exception) -> JSONResponse:
    log.exception("Unhandled error on %s %s", request.method, request.url.path, exc_info=exc)
    return JSONResponse(status_code=500, content=_body("INTERNAL", "Something went wrong on our side. Please try again."))
