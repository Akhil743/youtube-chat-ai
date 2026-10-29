"""Tests run offline: no network, no API key. Model clients are replaced by fakes."""

import os
import re

os.environ["PRELOAD_SAMPLES"] = "0"
os.environ.pop("GOOGLE_API_KEY", None)

import pytest  # noqa: E402
from langchain_core.embeddings import DeterministicFakeEmbedding  # noqa: E402
from langchain_core.messages import AIMessage  # noqa: E402

HEADER_RE = re.compile(r"\[(\d{2}:\d{2})–\d{2}:\d{2}\]")


class FakeLLM:
    """Stands in for ChatGoogleGenerativeAI.invoke(). Records prompts and returns
    an answer citing the first excerpt header plus whatever `extra` contains."""

    def __init__(self, extra: str = "", usage=None, raises: Exception | None = None):
        self.extra = extra
        self.usage = usage if usage is not None else {"input_tokens": 1000, "output_tokens": 100, "total_tokens": 1100}
        self.raises = raises
        self.calls: list = []

    def invoke(self, messages):
        self.calls.append(messages)
        if self.raises:
            raise self.raises
        prompt = messages[-1].content
        first = HEADER_RE.search(prompt).group(1)
        return AIMessage(content=f"It is explained here [{first}].{self.extra}", usage_metadata=self.usage)


@pytest.fixture
def fake_embeddings():
    return DeterministicFakeEmbedding(size=32)


@pytest.fixture
def fake_llm():
    return FakeLLM()
