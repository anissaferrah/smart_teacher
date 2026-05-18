"""Replay a list of questions through the same Q&A pipeline a real student
question uses (agentic.qa.runner.run_qa_graph). The existing eval_log
hook in runner.py captures (question, chunks, answer) into
logs/eval_log.jsonl automatically — this script just feeds questions in.

Questions file (scripts/eval_questions.json) is a JSON list of objects:
    [
      {
        "question":         "What is the median?",
        "slide":            5,
        "type":             "definition",
        "good_answer_hint": "Middle value of a sorted dataset.",
        "course_id":        "550e9734-415d-48eb-b1f1-c4a9d23b01d1"
      },
      ...
    ]

Only `question` and `course_id` are passed to the pipeline. The other
fields are eval metadata for the human grader — they live in the
question file so you can see them next to logs/eval_log.jsonl rows
when scoring.

Usage:
    python scripts/run_eval_dataset.py
    python scripts/run_eval_dataset.py --questions scripts/eval_questions.json \\
        --session_id eval_run_2 --delay 5
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path

# Disable SIGHT confusion detector for eval. It pulls in xlm-roberta-base
# (~1GB resident) on top of bge-m3 + reranker, which has been observed to
# OOM-kill Python on Windows. The QA pipeline doesn't need it for retrieval
# evaluation — intent.py:55-57 handles the missing import gracefully, and
# detector.py:63-65 short-circuits when the bundle file is absent.
# MUST be set before Config is imported (Config reads the env var at class
# definition time).
os.environ.setdefault("CONFUSION_MODEL_PATH", "__sight_disabled_for_eval__")

# Make the project root importable when running the script from anywhere
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
)
log = logging.getLogger("eval.run_dataset")


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default="scripts/eval_questions.json")
    ap.add_argument("--session_id", default="eval_run_1")
    ap.add_argument("--delay", type=float, default=3.0,
                    help="seconds between questions (avoid Groq rate limits)")
    ap.add_argument("--skip", type=int, default=0,
                    help="skip the first N questions (resume after a partial run)")
    ap.add_argument("--model", default=None,
                    help="override GROQ_MODEL for this eval session only "
                         "(e.g. llama-3.1-8b-instant for higher TPM). "
                         "Default = whatever is in .env.")
    args = ap.parse_args()

    qpath = Path(args.questions)
    if not qpath.exists():
        log.error("questions file not found: %s", qpath)
        return 1

    questions = json.loads(qpath.read_text(encoding="utf-8"))
    if not isinstance(questions, list):
        log.error("questions file must be a JSON list")
        return 1

    if args.skip > 0:
        log.info("skipping first %d question(s) — resuming from #%d", args.skip, args.skip + 1)
        questions = questions[args.skip:]

    # Override Groq model for this eval session only (process-local env
    # var, doesn't touch .env or affect the live system). MUST be set
    # before Config is imported.
    if args.model:
        os.environ["GROQ_MODEL"] = args.model
        log.info("GROQ_MODEL overridden to %r for this eval session", args.model)

    # Build the same brain + rag + qa_graph the server boots with
    # (mirrors main.py:131-169). We deliberately avoid importing main.py:
    # it has top-level side effects (Transcriber, VoiceEngine, service
    # registry) that aren't needed for offline replay.
    from core.config import Config
    Config.validate()
    from ai.llm import Brain
    from rag.multimodal_rag import MultiModalRAG
    from agentic import build_qa_graph
    from agentic.qa.runner import run_qa_graph

    log.info("building Brain + MultiModalRAG (cross-encoder warmup ≈ 20-40s) …")
    brain = Brain()

    # Monkey-patch: in the live system, Brain permanently disables Groq
    # after a single 429 (ai/llm.py:701-702) — sensible for production
    # ("don't pound a broken provider") but lethal for eval, where a
    # transient TPM hit on one question cascades to "all subsequent
    # answers are the French fallback". We override it to a no-op for
    # this process only; ai/llm.py is untouched. The 429-triggering
    # question still fails (we delete its polluted row), but the next
    # question's call goes through normally.
    brain._disable_openai = lambda reason: None  # type: ignore[method-assign]
    log.info("brain._disable_openai patched to no-op (eval session only)")

    rag = MultiModalRAG(
        db_dir=Config.RAG_DB_DIR,
        force_local_embeddings=not Config.RAG_ENABLED,
    )
    rag.warmup()
    qa_graph = build_qa_graph(brain, rag=rag)

    errors: list[tuple[int, str, str]] = []
    ran = 0
    total = len(questions)

    def _print_summary() -> None:
        print()
        print(f"Total questions run: {ran}/{total}")
        if errors:
            print(f"\nErrors ({len(errors)}):")
            for idx, text, err in errors:
                print(f"  [{idx}] {text[:60]} → {err[:120]}")
        print("\nRun: python scripts/show_eval_log.py --limit 40 to inspect results")

    for i, q in enumerate(questions, start=1):
        text = (q.get("question") or "").strip()
        course_id = q.get("course_id") or ""
        # Per-question language so the same dataset can mix EN/FR.
        # Defaults to "fr" (project default) when the field is absent.
        language = (q.get("language") or "fr")[:2]
        if not text or not course_id:
            log.warning("[%d/%d] skipped — missing question or course_id", i, total)
            continue

        log.info("[%d/%d] lang=%s q=%r course=%s…", i, total, language, text[:80], course_id[:8])
        try:
            await run_qa_graph(
                text=text,
                session_id=args.session_id,
                course_id=course_id,
                student_id=None,
                language=language,
                chapter_idx=0,
                chapter_title="",
                section_idx=0,
                section_title="",
                last_slide_content="",
                history=[],
                student_level="lycée",
                qa_graph=qa_graph,
                brain=brain,
            )
            ran += 1
        except Exception as exc:  # noqa: BLE001
            log.exception("[%d/%d] failed: %s", i, total, exc)
            errors.append((i, text, str(exc)))

        if args.delay > 0 and i < total:
            await asyncio.sleep(args.delay)

    _print_summary()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(asyncio.run(main()))
    except KeyboardInterrupt:
        log.warning("interrupted by user — see logs/eval_log.jsonl for partial results")
        raise SystemExit(130)
