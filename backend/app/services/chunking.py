"""Turn caption segments into retrieval chunks that keep exact time ranges.

Caption segments are short (2-5 s). We merge consecutive segments until a chunk
reaches ~CHUNK_TARGET_CHARS characters (~30-40 s of speech), carry the last
segment over into the next chunk for continuity, and record the real start/end
of the segments each chunk contains. We deliberately don't run a generic text
splitter afterwards: splitting a chunk would leave both halves claiming the
same time range, which is exactly the imprecision citations need to avoid.
"""

import os
import re
from typing import Iterable

from langchain_core.documents import Document

from app.services.transcript import TranscriptChunk

CHUNK_TARGET_CHARS = int(os.getenv("CHUNK_TARGET_CHARS", "500"))
CHUNK_MAX_CHARS = int(os.getenv("CHUNK_MAX_CHARS", "900"))
CHUNK_OVERLAP_SEGMENTS = int(os.getenv("CHUNK_OVERLAP_SEGMENTS", "1"))

_WS = re.compile(r"\s+")


def clean_text(text: str) -> str:
    return _WS.sub(" ", text.replace("﻿", "")).strip()


def _split_long(seg: TranscriptChunk, max_chars: int) -> list[TranscriptChunk]:
    """Split a single oversized segment on word boundaries, keeping its time range."""
    words, parts, cur = seg.text.split(" "), [], ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > max_chars:
            parts.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}" if cur else w
    if cur:
        parts.append(cur)
    return [TranscriptChunk(text=p, start_seconds=seg.start_seconds, end_seconds=seg.end_seconds) for p in parts]


def build_documents(
    segments: Iterable[TranscriptChunk],
    target_chars: int = CHUNK_TARGET_CHARS,
    max_chars: int = CHUNK_MAX_CHARS,
    overlap_segments: int = CHUNK_OVERLAP_SEGMENTS,
) -> list[Document]:
    segs: list[TranscriptChunk] = []
    for s in segments:
        text = clean_text(s.text)
        if not text:
            continue
        end = max(s.end_seconds, s.start_seconds)
        seg = TranscriptChunk(text=text, start_seconds=float(s.start_seconds), end_seconds=float(end))
        segs.extend(_split_long(seg, max_chars) if len(text) > max_chars else [seg])

    docs: list[Document] = []
    buf: list[TranscriptChunk] = []
    new_in_buf = 0

    def emit():
        docs.append(Document(
            page_content=" ".join(s.text for s in buf),
            metadata={
                "chunk_index": len(docs),
                "start_seconds": round(buf[0].start_seconds, 2),
                "end_seconds": round(max(s.end_seconds for s in buf), 2),
            },
        ))

    for seg in segs:
        buf.append(seg)
        new_in_buf += 1
        if sum(len(s.text) + 1 for s in buf) >= target_chars:
            emit()
            keep = buf[-overlap_segments:] if 0 < overlap_segments < len(buf) else []
            buf, new_in_buf = list(keep), 0

    if buf and new_in_buf:
        emit()
    return docs
