from app.services.chunking import build_documents
from app.services.samples import load_samples
from app.services.transcript import TranscriptChunk, build_transcript_from_raw


def seg(text, start, dur=3.0):
    return TranscriptChunk(text=text, start_seconds=start, end_seconds=start + dur)


def test_chunks_keep_exact_time_ranges():
    segs = [seg(f"sentence number {i} " + "x" * 40, i * 3.0) for i in range(30)]
    docs = build_documents(segs, target_chars=200, overlap_segments=1)
    assert len(docs) > 3
    for d in docs:
        md = d.metadata
        assert md["start_seconds"] < md["end_seconds"]
        # every segment inside the chunk lies within the chunk's range
        first_idx = int(d.page_content.split("sentence number ")[1].split(" ")[0])
        assert md["start_seconds"] == first_idx * 3.0
    # chunks are in time order and cover the whole transcript
    assert docs[0].metadata["start_seconds"] == 0.0
    assert docs[-1].metadata["end_seconds"] == 29 * 3.0 + 3.0
    starts = [d.metadata["start_seconds"] for d in docs]
    assert starts == sorted(starts)


def test_overlap_carries_last_segment_forward():
    segs = [seg("a" * 60, i * 2.0) for i in range(10)]
    docs = build_documents(segs, target_chars=150, overlap_segments=1)
    for prev, nxt in zip(docs, docs[1:]):
        assert nxt.metadata["start_seconds"] < prev.metadata["end_seconds"]


def test_no_overlap_and_trailing_chunk():
    segs = [seg("word " * 20, i * 5.0) for i in range(5)]  # 100 chars each
    docs = build_documents(segs, target_chars=250, overlap_segments=0)
    assert [d.metadata["start_seconds"] for d in docs] == [0.0, 15.0]
    assert docs[-1].metadata["end_seconds"] == 23.0  # last segment: start 20 + 3 s


def test_trailing_overlap_only_is_not_emitted_twice():
    segs = [seg("b" * 100, i * 2.0) for i in range(4)]
    docs = build_documents(segs, target_chars=200, overlap_segments=1)
    assert len(docs) == 3 and docs[-1].metadata["end_seconds"] == 9.0
    assert len({(d.metadata["start_seconds"], d.metadata["end_seconds"]) for d in docs}) == len(docs)


def test_cleans_whitespace_bom_and_skips_empty():
    segs = [seg("﻿Hello\nworld", 0), seg("   ", 3), seg("again", 6)]
    docs = build_documents(segs, target_chars=1000)
    assert docs[0].page_content == "Hello world again"


def test_oversized_segment_is_split_but_keeps_its_range():
    long = seg("word " * 400, 10.0, dur=20.0)  # 2000 chars
    docs = build_documents([long], target_chars=500, max_chars=600, overlap_segments=0)
    assert len(docs) >= 3
    assert all(d.metadata["start_seconds"] == 10.0 and d.metadata["end_seconds"] == 30.0 for d in docs)
    assert all(len(d.page_content) <= 600 for d in docs)


def test_build_transcript_from_raw_tolerates_missing_fields():
    t = build_transcript_from_raw("x" * 11, [{"text": "hi", "start": 1}, {"text": "", "start": 2}, {"text": "yo", "start": "3.5", "duration": 2}])
    assert [(c.text, c.start_seconds, c.end_seconds) for c in t.chunks] == [("hi", 1.0, 1.0), ("yo", 3.5, 5.5)]


def test_bundled_samples_load_and_chunk():
    samples = load_samples()
    assert len(samples) == 3
    for s in samples.values():
        assert s.license.startswith("Creative Commons Attribution")
        docs = build_documents(s.transcript.chunks)
        assert docs and docs[-1].metadata["end_seconds"] <= s.duration_seconds + 1
