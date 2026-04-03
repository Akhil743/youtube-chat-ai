import os
import time
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

FREE_DAILY_LOADS = int(os.getenv("FREE_DAILY_LOADS", "5"))
FREE_DAILY_MESSAGES = int(os.getenv("FREE_DAILY_MESSAGES", "20"))


@dataclass
class UserUsage:
    loads: int = 0
    messages: int = 0
    day: str = ""


@dataclass
class UserRecord:
    user_id: str
    is_pro: bool = False
    stripe_customer_id: Optional[str] = None
    usage: UserUsage = field(default_factory=UserUsage)


class QuotaService:
    """In-memory per-user quota tracking. Resets daily. Swap to Redis/DB for persistence at scale."""

    def __init__(self):
        self._users: dict[str, UserRecord] = {}
        self._lock = threading.Lock()

    def _today(self) -> str:
        return time.strftime("%Y-%m-%d", time.gmtime())

    def _get_or_create(self, user_id: str) -> UserRecord:
        with self._lock:
            if user_id not in self._users:
                self._users[user_id] = UserRecord(user_id=user_id)
            record = self._users[user_id]
            today = self._today()
            if record.usage.day != today:
                record.usage = UserUsage(day=today)
            return record

    def set_pro(self, user_id: str, stripe_customer_id: str):
        record = self._get_or_create(user_id)
        with self._lock:
            record.is_pro = True
            record.stripe_customer_id = stripe_customer_id

    def revoke_pro(self, user_id: str):
        record = self._get_or_create(user_id)
        with self._lock:
            record.is_pro = False

    def check_load_quota(self, user_id: Optional[str]) -> dict:
        if not user_id:
            return {"allowed": True, "used": 0, "limit": FREE_DAILY_LOADS, "is_pro": False}

        record = self._get_or_create(user_id)
        if record.is_pro:
            return {"allowed": True, "used": record.usage.loads, "limit": -1, "is_pro": True}

        allowed = record.usage.loads < FREE_DAILY_LOADS
        return {
            "allowed": allowed,
            "used": record.usage.loads,
            "limit": FREE_DAILY_LOADS,
            "is_pro": False,
        }

    def record_load(self, user_id: Optional[str]):
        if not user_id:
            return
        record = self._get_or_create(user_id)
        with self._lock:
            record.usage.loads += 1

    def check_chat_quota(self, user_id: Optional[str]) -> dict:
        if not user_id:
            return {"allowed": True, "used": 0, "limit": FREE_DAILY_MESSAGES, "is_pro": False}

        record = self._get_or_create(user_id)
        if record.is_pro:
            return {"allowed": True, "used": record.usage.messages, "limit": -1, "is_pro": True}

        allowed = record.usage.messages < FREE_DAILY_MESSAGES
        return {
            "allowed": allowed,
            "used": record.usage.messages,
            "limit": FREE_DAILY_MESSAGES,
            "is_pro": False,
        }

    def record_chat(self, user_id: Optional[str]):
        if not user_id:
            return
        record = self._get_or_create(user_id)
        with self._lock:
            record.usage.messages += 1

    def get_quota_info(self, user_id: Optional[str]) -> dict:
        """Return current quota state for inclusion in API responses."""
        if not user_id:
            return None
        record = self._get_or_create(user_id)
        return {
            "used": record.usage.messages,
            "limit": FREE_DAILY_MESSAGES,
            "is_pro": record.is_pro,
        }


quota_service = QuotaService()
