import pytest

from app import pricing


def test_flash_lite_cost_maths():
    # 1,240 input + 96 output at $0.10 / $0.40 per 1M
    assert pricing.cost_usd("gemini-2.5-flash-lite", 1240, 96) == pytest.approx(0.0001624)


def test_embedding_cost_and_models_prefix():
    assert pricing.cost_usd("models/gemini-embedding-001", 1_000_000) == pytest.approx(0.15)


def test_unknown_model_has_no_price():
    assert pricing.cost_usd("some-future-model", 1000, 1000) is None


def test_env_override(monkeypatch):
    monkeypatch.setenv("PRICE_GEMINI_2_5_FLASH_LITE_INPUT", "1.0")
    monkeypatch.setenv("PRICE_GEMINI_2_5_FLASH_LITE_OUTPUT", "2.0")
    assert pricing.cost_usd("gemini-2.5-flash-lite", 1_000_000, 1_000_000) == pytest.approx(3.0)


def test_env_can_price_a_new_model(monkeypatch):
    monkeypatch.setenv("PRICE_GEMINI_9_FLASH_INPUT", "0.5")
    assert pricing.cost_usd("gemini-9-flash", 2_000_000) == pytest.approx(1.0)


def test_bad_env_value_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("PRICE_GEMINI_2_5_FLASH_LITE_INPUT", "cheap")
    assert pricing.get_price("gemini-2.5-flash-lite").input_per_m == 0.10


def test_token_estimate():
    assert pricing.estimate_tokens("") == 0
    assert pricing.estimate_tokens("abcd") == 1
    assert pricing.estimate_tokens("abcde") == 2
    assert pricing.estimate_tokens(4000) == 1000


def test_prices_are_dated_and_sourced():
    assert pricing.AS_OF and pricing.SOURCE_URL.startswith("https://ai.google.dev/")
