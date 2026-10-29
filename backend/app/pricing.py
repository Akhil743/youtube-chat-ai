"""List prices for the Gemini models this service calls, and the cost maths.

Prices are USD per 1M tokens, Gemini API *paid* tier, standard (non-batch)
text input. Source: https://ai.google.dev/gemini-api/docs/pricing, checked on
AS_OF (the page said "Last updated 2026-10-28 UTC"). Both models are also
listed as "Free of charge" on the free tier, so the numbers the API returns are
what an answer *would* cost at list price, not necessarily what was billed.

Override any price without a code change, e.g.:
    PRICE_GEMINI_2_5_FLASH_LITE_INPUT=0.10
    PRICE_GEMINI_2_5_FLASH_LITE_OUTPUT=0.40
    PRICE_GEMINI_EMBEDDING_001_INPUT=0.15
    PRICING_AS_OF=2026-10-29
"""

import math
import os
import re
from dataclasses import dataclass
from typing import Optional

SOURCE_URL = "https://ai.google.dev/gemini-api/docs/pricing"
AS_OF = os.getenv("PRICING_AS_OF", "2026-10-29")


@dataclass(frozen=True)
class ModelPrice:
    input_per_m: float
    output_per_m: float = 0.0


DEFAULT_PRICES: dict[str, ModelPrice] = {
    # Chat model. $0.10 in (text/image/video) / $0.40 out per 1M tokens.
    "gemini-2.5-flash-lite": ModelPrice(input_per_m=0.10, output_per_m=0.40),
    # Embedding model. $0.15 per 1M input tokens; no output tokens.
    "gemini-embedding-001": ModelPrice(input_per_m=0.15, output_per_m=0.0),
}


def _env_prefix(model: str) -> str:
    return "PRICE_" + re.sub(r"[^A-Z0-9]+", "_", model.upper()).strip("_")


def _env_float(name: str) -> Optional[float]:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def get_price(model: str) -> Optional[ModelPrice]:
    """Price for `model` (accepts a leading "models/"), env overrides applied."""
    model = model.removeprefix("models/")
    base = DEFAULT_PRICES.get(model)
    prefix = _env_prefix(model)
    inp = _env_float(prefix + "_INPUT")
    out = _env_float(prefix + "_OUTPUT")
    if base is None and inp is None:
        return None
    return ModelPrice(
        input_per_m=inp if inp is not None else base.input_per_m,
        output_per_m=out if out is not None else (base.output_per_m if base else 0.0),
    )


def cost_usd(model: str, input_tokens: int, output_tokens: int = 0) -> Optional[float]:
    """Dollar cost at list price, or None if the model has no known price."""
    price = get_price(model)
    if price is None:
        return None
    cost = (input_tokens * price.input_per_m + output_tokens * price.output_per_m) / 1_000_000
    return round(cost, 8)


def estimate_tokens(text_or_chars) -> int:
    """Rough token count (~4 characters per token for English).

    The Gemini embeddings endpoint doesn't return token counts, so embedding
    costs are estimates and are labelled as such in API responses.
    """
    chars = text_or_chars if isinstance(text_or_chars, int) else len(text_or_chars)
    return max(1, math.ceil(chars / 4)) if chars else 0
