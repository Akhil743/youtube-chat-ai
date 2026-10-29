"""Small offline-reproducible eval of the RAG pipeline on the bundled sample videos.

Metrics
- citation_validity: share of timestamps the model cited (before post-validation)
  that fall inside a retrieved chunk's range. After post-validation the UI only
  ever shows valid ones; this measures how often the model needed correcting.
- refusal_accuracy: share of unanswerable questions the model declined.
- false_refusal_rate: share of answerable questions it wrongly declined.
- faithfulness: share of non-refusal answers an LLM judge rated fully supported
  by the retrieved excerpts (judge sees the same excerpts the model saw).
- answer_rate: share of answerable questions the judge says were actually answered.
- latency p50/p95 (server-side retrieval + generation) and mean $/answer at list price.

Usage (from backend/):
    python evals/run_eval.py               # live: needs GOOGLE_API_KEY, writes evals/results.json
    python evals/run_eval.py --offline     # plumbing check with fake models, writes nothing
    EVAL_JUDGE_MODEL=gemini-2.5-flash python evals/run_eval.py   # stronger judge

Live budget: 15 answers + <=15 judge calls + 3 index builds + 15 query embeddings,
hard-capped by MAX_LIVE_GENERATIONS.
"""

import argparse
import asyncio
import json
import os
import re
import statistics
import sys
import time
from datetime import date
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))
os.environ.setdefault("PRELOAD_SAMPLES", "0")

from dotenv import load_dotenv  # noqa: E402

load_dotenv(BACKEND / ".env")

from app import pricing  # noqa: E402
from app.services.chunking import CHUNK_TARGET_CHARS  # noqa: E402
from app.services.citations import format_chunk  # noqa: E402
from app.services.rag import RAGService  # noqa: E402
from app.services.samples import load_samples  # noqa: E402

QUESTIONS = Path(__file__).resolve().parent / "questions.json"
RESULTS = Path(__file__).resolve().parent / "results.json"
MAX_LIVE_GENERATIONS = 40  # answers + judge calls

JUDGE_PROMPT = """You are grading an answer produced by a question-answering system that must answer ONLY from the transcript excerpts below.

Transcript excerpts:
{excerpts}

Question: {question}

Answer to grade:
{answer}

Reply with a JSON object and nothing else:
{{"faithful": true|false, "answers_question": true|false, "unsupported": ["claims not supported by the excerpts"]}}
- faithful: true only if every factual claim in the answer is supported by the excerpts (ignore the bracketed timestamps).
- answers_question: true if the answer actually addresses the question (a refusal is false)."""


def percentile(values, p):
    if not values:
        return None
    s = sorted(values)
    k = (len(s) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def parse_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}


class FakeJudge:
    def invoke(self, prompt):
        from langchain_core.messages import AIMessage
        return AIMessage(content='{"faithful": true, "answers_question": true, "unsupported": []}',
                         usage_metadata={"input_tokens": 500, "output_tokens": 20, "total_tokens": 520})


def offline_service() -> RAGService:
    from langchain_core.embeddings import DeterministicFakeEmbedding
    from langchain_core.messages import AIMessage

    class FakeLLM:
        def invoke(self, messages):
            first = re.search(r"\[(\d{2}:\d{2})–", messages[-1].content).group(1)
            return AIMessage(content=f"Offline placeholder answer [{first}] [59:59].",
                             usage_metadata={"input_tokens": 900, "output_tokens": 60, "total_tokens": 960})

    return RAGService(embeddings=DeterministicFakeEmbedding(size=32), llm=FakeLLM())


def make_judge(model: str):
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(model=model, google_api_key=os.getenv("GOOGLE_API_KEY"), temperature=0, max_retries=1)


async def run(offline: bool, out: Path | None):
    if not offline and not os.getenv("GOOGLE_API_KEY"):
        sys.exit("GOOGLE_API_KEY is not set (put it in backend/.env). Use --offline for a plumbing check.")

    questions = json.loads(QUESTIONS.read_text())["questions"]
    samples = load_samples()
    service = offline_service() if offline else RAGService()
    judge_model = os.getenv("EVAL_JUDGE_MODEL", service.chat_model)
    judge = FakeJudge() if offline else make_judge(judge_model)
    generations = 0

    index_costs = {}
    for vid in sorted({q["video_id"] for q in questions}):
        s = samples[vid]
        store = await service.ingest(s.transcript, title=s.title, channel=s.channel,
                                     duration_seconds=s.duration_seconds, is_sample=True)
        index_costs[vid] = store.ingest_usage.cost_usd

    rows, judge_cost = [], 0.0
    for q in questions:
        if generations >= MAX_LIVE_GENERATIONS:
            sys.exit("Live call budget exhausted; aborting.")
        res = await service.answer(q["video_id"], q["question"])
        generations += 1
        verdict = {}
        if not res.refused:
            excerpts = "\n\n".join(format_chunk(r["text"], r["start_seconds"], r["end_seconds"])
                                   for r in sorted(res.retrieved, key=lambda r: r["start_seconds"]))
            prompt = JUDGE_PROMPT.format(excerpts=excerpts, question=q["question"], answer=res.raw_answer)
            msg = await asyncio.to_thread(judge.invoke, prompt)
            generations += 1
            verdict = parse_json(msg.content if isinstance(msg.content, str) else str(msg.content))
            um = getattr(msg, "usage_metadata", None) or {}
            judge_cost += pricing.cost_usd(judge_model, um.get("input_tokens", 0), um.get("output_tokens", 0)) or 0.0
        check = res.citation_check
        rows.append({
            "id": q["id"], "video_id": q["video_id"], "question": q["question"], "answerable": q["answerable"],
            "refused": res.refused, "answer": res.answer, "raw_answer": res.raw_answer,
            "citations_cited": len(check.cited), "citations_valid": len(check.valid), "citations_removed": check.removed,
            "judge_faithful": verdict.get("faithful"), "judge_answers_question": verdict.get("answers_question"),
            "judge_unsupported": verdict.get("unsupported", []),
            "latency_ms": res.usage["latency_ms"], "cost_usd": res.usage["cost_usd"],
            "input_tokens": res.usage["input_tokens"], "output_tokens": res.usage["output_tokens"],
        })
        print(f"{q['id']:5} refused={res.refused!s:5} cites={len(check.valid)}/{len(check.cited)} "
              f"faithful={verdict.get('faithful')} {res.usage['latency_ms']} ms")

    answerable = [r for r in rows if r["answerable"]]
    unanswerable = [r for r in rows if not r["answerable"]]
    judged = [r for r in rows if r["judge_faithful"] is not None]
    cited = sum(r["citations_cited"] for r in rows)
    latencies = [r["latency_ms"] for r in rows]
    costs = [r["cost_usd"] for r in rows if r["cost_usd"] is not None]

    def share(n, d):
        return round(n / d, 3) if d else None

    metrics = {
        "n_questions": len(rows),
        "n_unanswerable": len(unanswerable),
        "citation_validity": share(sum(r["citations_valid"] for r in rows), cited),
        "timestamps_cited": cited,
        "answers_with_citation": share(sum(1 for r in answerable if not r["refused"] and r["citations_valid"] > 0),
                                       sum(1 for r in answerable if not r["refused"])),
        "refusal_accuracy": share(sum(r["refused"] for r in unanswerable), len(unanswerable)),
        "false_refusal_rate": share(sum(r["refused"] for r in answerable), len(answerable)),
        "faithfulness": share(sum(bool(r["judge_faithful"]) for r in judged), len(judged)),
        "answer_rate": share(sum(bool(r["judge_answers_question"]) for r in answerable), len(answerable)),
        "latency_ms_p50": round(percentile(latencies, 50)) if latencies else None,
        "latency_ms_p95": round(percentile(latencies, 95)) if latencies else None,
        "mean_cost_usd_per_answer": round(statistics.mean(costs), 7) if costs else None,
        "mean_input_tokens": round(statistics.mean(r["input_tokens"] for r in rows)),
        "mean_output_tokens": round(statistics.mean(r["output_tokens"] for r in rows)),
        "index_cost_usd_per_video": index_costs,
        "judge_cost_usd_total": round(judge_cost, 6),
    }
    report = {
        "run_date": date.today().isoformat(),
        "offline": offline,
        "config": {
            "chat_model": service.chat_model, "embedding_model": service.embedding_model,
            "judge_model": judge_model, "retrieval_k": service.k, "chunk_target_chars": CHUNK_TARGET_CHARS,
            "pricing_as_of": pricing.AS_OF, "pricing_source": pricing.SOURCE_URL,
        },
        "metrics": metrics,
        "rows": rows,
    }
    print(json.dumps(metrics, indent=2))
    if out:
        out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--offline", action="store_true", help="fake models; checks plumbing, writes nothing by default")
    ap.add_argument("--out", type=Path, help="where to write results (default: evals/results.json for live runs)")
    args = ap.parse_args()
    out = args.out or (None if args.offline else RESULTS)
    t0 = time.time()
    asyncio.run(run(args.offline, out))
    print(f"done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
