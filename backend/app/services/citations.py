"""Timestamp formatting/parsing and post-validation of cited timestamps.

Every retrieved chunk is shown to the model as "[mm:ss–mm:ss] text". The model
is told to cite only those times. After generation we parse every bracketed
timestamp in the answer and keep it only if it falls inside the displayed range
of a chunk that was actually retrieved for this answer. Anything else is
removed from the answer and reported back, so the UI never shows a timestamp
the transcript can't back up.
"""

import math
import re
from dataclasses import dataclass, field

_TS = r"\d{1,2}:\d{2}(?::\d{2})?"
_SEP = r"\s*(?:[–—-]|,|;|\band\b|\bto\b)\s*"
# "[01:05]", "[1:05–1:32]", "[01:05, 02:10]", "(01:05)" ... with optional leading space.
CITATION_RE = re.compile(rf"(\s?)([\[(])\s*({_TS}(?:{_SEP}{_TS})*)\s*([\])])")
_TS_RE = re.compile(_TS)

REFUSAL_TEXT = "I couldn't find that in this video's transcript."
_REFUSAL_MARKERS = (
    "couldn't find that in this video",
    "could not find that in this video",
    "couldn’t find that in this video",
)


def format_ts(seconds: float) -> str:
    """83.7 -> "01:23"; 3725 -> "1:02:05". Seconds are floored."""
    s = max(0, int(seconds))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m:02d}:{sec:02d}"


def parse_ts(text: str) -> int:
    """"01:23" -> 83, "1:02:05" -> 3725. Raises ValueError on malformed input."""
    parts = text.strip().split(":")
    if len(parts) not in (2, 3) or not all(p.isdigit() for p in parts):
        raise ValueError(f"Not a timestamp: {text!r}")
    nums = [int(p) for p in parts]
    if nums[-1] >= 60 or (len(nums) == 3 and nums[1] >= 60):
        raise ValueError(f"Not a timestamp: {text!r}")
    if len(nums) == 2:
        return nums[0] * 60 + nums[1]
    return nums[0] * 3600 + nums[1] * 60 + nums[2]


def display_range(start: float, end: float) -> tuple[int, int]:
    """Integer second range as displayed to the model: floor(start)..ceil(end)."""
    lo = max(0, int(math.floor(start)))
    return lo, max(lo, int(math.ceil(end)))


def format_range(start: float, end: float) -> str:
    lo, hi = display_range(start, end)
    return f"{format_ts(lo)}–{format_ts(hi)}"


def format_chunk(text: str, start: float, end: float) -> str:
    return f"[{format_range(start, end)}] {text}"


def is_refusal(answer: str) -> bool:
    low = answer.lower()
    return any(m in low for m in _REFUSAL_MARKERS)


@dataclass
class CitationCheck:
    answer: str                                   # answer with invalid citations removed
    cited: list[int] = field(default_factory=list)    # every timestamp the model cited (seconds)
    valid: list[int] = field(default_factory=list)    # the subset inside a retrieved chunk
    removed: list[str] = field(default_factory=list)  # citation strings dropped from the answer

    @property
    def validity(self) -> float | None:
        return len(self.valid) / len(self.cited) if self.cited else None


def is_supported(seconds: int, spans: list[tuple[float, float]]) -> bool:
    """True if `seconds` lies inside the displayed range of any retrieved chunk."""
    for start, end in spans:
        lo, hi = display_range(start, end)
        if lo <= seconds <= hi:
            return True
    return False


def validate_citations(answer: str, spans: list[tuple[float, float]]) -> CitationCheck:
    """Keep bracketed timestamps that fall inside a retrieved chunk; drop the rest.

    Ranges ("[01:05–01:32]") are kept only if both ends are supported. Lists
    ("[01:05, 02:10]") keep only their supported members. Kept citations are
    normalised to "[mm:ss]" / "[mm:ss–mm:ss]" so the UI can link them.
    """
    check = CitationCheck(answer=answer)

    def _replace(m: re.Match) -> str:
        lead, body = m.group(1), m.group(3)
        is_range = bool(re.search(r"[–—-]|\bto\b", body)) and len(_TS_RE.findall(body)) == 2
        stamps = []
        for raw in _TS_RE.findall(body):
            try:
                stamps.append(parse_ts(raw))
            except ValueError:
                stamps.append(None)
        points = [s for s in stamps if s is not None]
        check.cited.extend(points)
        ok = [s for s in points if is_supported(s, spans)]
        check.valid.extend(ok)

        if is_range:
            if len(ok) == 2 and None not in stamps:
                return f"{lead}[{format_ts(ok[0])}–{format_ts(ok[1])}]"
            check.removed.append(m.group(0).strip())
            return ""
        if not ok:
            check.removed.append(m.group(0).strip())
            return ""
        if len(ok) < len(stamps):
            check.removed.append(m.group(0).strip())
        return lead + " ".join(f"[{format_ts(s)}]" for s in ok)

    cleaned = CITATION_RE.sub(_replace, answer)
    if check.removed:
        # Tidy spaces left before punctuation by removed citations.
        cleaned = re.sub(r"[ \t]+([.,;:!?])", r"\1", cleaned)
    check.answer = cleaned.strip()
    return check
