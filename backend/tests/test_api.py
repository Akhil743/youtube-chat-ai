"""End-to-end API tests with fake embeddings and a fake LLM (no network)."""

import pytest
from fastapi.testclient import TestClient

from app.limiter import DailyCap, limiter
from app.main import app
from app.routers import chat as chat_router
from app.services.rag import rag_service
from tests.conftest import FakeLLM

SAMPLE = "0eKVizvYSUQ"  # "How Google Search Works", bundled transcript


@pytest.fixture
def client(monkeypatch, fake_embeddings):
    llm = FakeLLM(extra=" It was also said at [59:59].")
    monkeypatch.setattr(rag_service, "_embeddings", fake_embeddings)
    monkeypatch.setattr(rag_service, "_llm", llm)
    monkeypatch.setattr(chat_router, "daily_cap", DailyCap(100))
    rag_service._cache.clear()
    limiter.reset()
    with TestClient(app) as c:
        c.llm = llm
        yield c
    rag_service._cache.clear()


def load(client, url=f"https://youtu.be/{SAMPLE}"):
    return client.post("/api/load-video", json={"url": url})


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "healthy"
    assert client.head("/health").status_code == 200


def test_load_sample_returns_title_duration_and_embedding_cost(client):
    r = load(client)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["title"] == "How Google Search Works (in 5 minutes)"
    assert body["channel"] == "Google" and body["duration_seconds"] == 315
    assert body["is_sample"] and not body["already_loaded"]
    assert body["chunk_count"] > 5
    u = body["usage"]
    assert u["model"] == "gemini-embedding-001" and u["tokens_estimated"]
    assert u["embedding_tokens"] > 1000 and 0 < u["cost_usd"] < 0.01
    assert load(client).json()["already_loaded"] is True


def test_chat_returns_grounded_answer_sources_and_usage(client):
    load(client)
    r = client.post("/api/chat", json={"video_id": SAMPLE, "question": "How does Google rank pages?",
                                       "chat_history": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200, r.text
    body = r.json()
    # invented [59:59] dropped, real chunk citation kept
    assert "[59:59]" not in body["answer"]
    assert body["citation_check"] == {"cited": 2, "valid": 1, "removed": ["[59:59]"]}
    srcs = body["sources"]
    assert len(srcs) == 5 and sum(s["cited"] for s in srcs) >= 1
    for s in srcs:
        assert s["start_seconds"] < s["end_seconds"] and s["snippet"] and "–" in s["label"]
    u = body["usage"]
    assert (u["input_tokens"], u["output_tokens"]) == (1000, 100)
    assert u["cost_usd"] == pytest.approx(1000 * 0.10e-6 + 100 * 0.40e-6 + u["embedding_tokens"] * 0.15e-6)
    assert u["latency_ms"] >= 0 and u["model"] == "gemini-2.5-flash-lite" and u["pricing_as_of"]


def test_prompt_shows_timestamped_excerpts_in_time_order(client):
    load(client)
    client.post("/api/chat", json={"video_id": SAMPLE, "question": "What is the index?"})
    system, user = client.llm.calls[-1]
    assert "Only cite times that appear in the excerpt headers" in system.content
    import re
    starts = re.findall(r"^\[(\d{2}:\d{2})–\d{2}:\d{2}\] ", user.content, flags=re.M)
    assert len(starts) == 5 and starts == sorted(starts)


def test_chat_on_unloaded_video_says_so(client):
    r = client.post("/api/chat", json={"video_id": "aaaaaaaaaaa", "question": "hi"})
    assert r.status_code == 404 and r.json()["code"] == "VIDEO_NOT_LOADED"


def test_model_errors_do_not_leak_details(client, monkeypatch):
    load(client)
    monkeypatch.setattr(rag_service, "_llm", FakeLLM(raises=RuntimeError("secret internal path /etc/key=abc")))
    r = client.post("/api/chat", json={"video_id": SAMPLE, "question": "hi"})
    assert r.status_code == 502
    assert "secret" not in r.text and r.json()["code"] == "MODEL_ERROR"


def test_validation_errors_are_friendly(client):
    r = client.post("/api/load-video", json={"url": "https://vimeo.com/123456789012"})
    assert r.status_code == 422 and r.json() == {"detail": "That doesn't look like a YouTube video link.", "code": "INVALID_INPUT"}
    r = client.post("/api/chat", json={"video_id": SAMPLE, "question": "   "})
    assert r.status_code == 422 and r.json()["code"] == "INVALID_INPUT"


def test_daily_cap_returns_friendly_429(client, monkeypatch):
    load(client)
    monkeypatch.setattr(chat_router, "daily_cap", DailyCap(1))
    assert client.post("/api/chat", json={"video_id": SAMPLE, "question": "one"}).status_code == 200
    r = client.post("/api/chat", json={"video_id": SAMPLE, "question": "two"})
    assert r.status_code == 429 and r.json()["code"] == "DAILY_CAP"


def test_rate_limit_is_per_client_ip(client):
    load(client)
    for _ in range(20):
        assert client.post("/api/chat", json={"video_id": SAMPLE, "question": "q"},
                           headers={"CF-Connecting-IP": "203.0.113.1"}).status_code == 200
    r = client.post("/api/chat", json={"video_id": SAMPLE, "question": "q"}, headers={"CF-Connecting-IP": "203.0.113.1"})
    assert r.status_code == 429 and r.json()["code"] == "RATE_LIMITED"
    other = client.post("/api/chat", json={"video_id": SAMPLE, "question": "q"}, headers={"CF-Connecting-IP": "203.0.113.2"})
    assert other.status_code == 200


@pytest.mark.parametrize("origin,allowed", [
    ("https://akhilrajs.com", True),
    ("http://localhost:8080", True),
    ("http://127.0.0.1:5500", True),
    ("https://evil.example.com", False),
    ("chrome-extension://abcdefghijklmnop", False),
    ("http://localhost.evil.com", False),
])
def test_cors(client, origin, allowed):
    r = client.options("/api/chat", headers={"Origin": origin, "Access-Control-Request-Method": "POST",
                                            "Access-Control-Request-Headers": "content-type"})
    assert (r.headers.get("access-control-allow-origin") == origin) is allowed
    assert "access-control-allow-credentials" not in r.headers
