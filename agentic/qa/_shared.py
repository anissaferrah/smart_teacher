"""Shared constants and helpers for the agentic Q&A nodes.

Three families of constants used to live as bare literals duplicated
across :mod:`responder`, :mod:`rewriter`, and :mod:`intent`:
  - slide-content caps (``2000`` in responder, ``500`` in rewriter/intent)
  - chunk-content caps (``500`` in responder)
  - history-window sizes (``3`` pairs, ``240`` chars per message)

Pulling them here makes the prompt-budget story explicit in one file so
a future "switch model / different latency budget" change is one diff
instead of four.

The helper :func:`format_history` is also factored here — three near-
identical copies were drifting ; the responder version had role labels,
the others didn't. We pick the labelled version as the canonical one
since labels help the LLM distinguish "what the student said" from
"what we already answered".
"""
from __future__ import annotations

from typing import Iterable

# ── Prompt budget (calibrated for GPT-4o-mini ~128k context, target
# 4-6k effective tokens for latency + cost). See responder.py:69-83 for
# the original derivation. Loosen if switching to a wider-context model
# OR if you observe the LLM ignoring relevant chunks ; tighten if
# latency/cost is the bottleneck.
SLIDE_CONTENT_CAP = 2000        # full slide block, used by responder QA prompt
SLIDE_EXCERPT_CAP = 500         # short excerpt, used by intent / rewriter
CHUNK_TEXT_CAP = 500            # per-chunk in the responder grounding block
MAX_CITED_CHUNKS = 5            # top-K chunks visible to the LLM

# ── Conversational memory window
HISTORY_PAIRS = 3                # last N (user, assistant) exchanges fed back
HISTORY_MSG_MAX = 240            # per-message char cap (~60 tokens)


def format_history(
    history: Iterable[dict] | None,
    lang: str = "fr",
    *,
    pairs: int = HISTORY_PAIRS,
    msg_cap: int = HISTORY_MSG_MAX,
    labelled: bool = True,
) -> str:
    """Render the last ``pairs`` exchanges in role-tagged plain text.

    Parameters
    ----------
    pairs:
        Number of (user, assistant) round-trips to feed back. Each
        round-trip is two messages, so the actual tail length is
        ``pairs * 2``.
    msg_cap:
        Per-message char cap. Lower budgets (e.g. Ollama 4k context)
        pass a smaller cap; the responder uses the module default.
    labelled:
        When True (default), roles are rendered as "Étudiant" / "Prof"
        (FR) or "Student" / "Teacher" (EN). When False, the raw role
        strings ``user`` / ``assistant`` are emitted instead — useful
        for the merged Intent+Rewriter prompt where role-style cues
        aren't needed.

    Empty / malformed messages are skipped silently; multiline content
    is flattened so the prompt stays one line per turn.
    """
    if not history:
        return "(début de la conversation)" if lang == "fr" else "(start of conversation)"
    history_list = list(history)
    if labelled:
        student_label = "Étudiant" if lang == "fr" else "Student"
        teacher_label = "Prof" if lang == "fr" else "Teacher"
        role_map = {
            "user": student_label, "student": student_label,
            "assistant": teacher_label, "teacher": teacher_label, "ai": teacher_label,
        }
    else:
        role_map = {"user": "user", "student": "user", "assistant": "assistant",
                    "teacher": "assistant", "ai": "assistant"}
    tail = history_list[-(pairs * 2):]
    lines: list[str] = []
    for msg in tail:
        if not isinstance(msg, dict):
            continue
        role = role_map.get(str(msg.get("role", "")).lower())
        content = (msg.get("content") or "").strip().replace("\n", " ")
        if not role or not content:
            continue
        lines.append(f"{role}: {content[:msg_cap]}")
    if not lines:
        return "(début de la conversation)" if lang == "fr" else "(start of conversation)"
    return "\n".join(lines)
