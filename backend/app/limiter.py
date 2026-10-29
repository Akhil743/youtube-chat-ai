"""Abuse guards: per-client rate limits and a global daily cap on model calls.

Behind Render (and Cloudflare in front of it) every request arrives from a
proxy address, so keying on the socket address puts all users in one bucket.
We key on CF-Connecting-IP, then the first X-Forwarded-For hop, then the
socket address. Those headers can be spoofed by a client that bypasses the
proxy, so the global daily cap is the real backstop on spend.
"""

import os
import threading
import time
from typing import Callable

from slowapi import Limiter
from starlette.requests import Request

LOAD_RATE_LIMIT = os.getenv("LOAD_RATE_LIMIT", "10/minute")
CHAT_RATE_LIMIT = os.getenv("CHAT_RATE_LIMIT", "20/minute")


def client_ip(request: Request) -> str:
    cf = request.headers.get("cf-connecting-ip", "").strip()
    if cf:
        return cf
    xff = request.headers.get("x-forwarded-for", "")
    first = xff.split(",")[0].strip() if xff else ""
    if first:
        return first
    return request.client.host if request.client else "unknown"


limiter = Limiter(key_func=client_ip)


class DailyCap:
    """Process-wide counter of model calls per UTC day.

    In-memory, so it resets on restart; it bounds spend on a single always-on
    instance, it is not billing-grade accounting.
    """

    def __init__(self, limit: int, clock: Callable[[], float] = time.time):
        self.limit = limit
        self._clock = clock
        self._lock = threading.Lock()
        self._day = ""
        self._used = 0

    def _today(self) -> str:
        return time.strftime("%Y-%m-%d", time.gmtime(self._clock()))

    def _roll(self):
        today = self._today()
        if today != self._day:
            self._day, self._used = today, 0

    def try_consume(self, n: int = 1) -> bool:
        with self._lock:
            self._roll()
            if self._used + n > self.limit:
                return False
            self._used += n
            return True

    def remaining(self) -> int:
        with self._lock:
            self._roll()
            return max(0, self.limit - self._used)


daily_cap = DailyCap(int(os.getenv("DAILY_LLM_CAP", "500")))
