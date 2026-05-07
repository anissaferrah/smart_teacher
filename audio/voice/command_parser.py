"""Simple voice navigation command parser.

Recognises navigation intents from short transcripts and returns a
normalised action that the websocket handler can execute without calling
the LLM.

Supported actions: next, previous, repeat, quiz, explain, pause, resume,
flag_confusion, goto.

# Match strategy

Three strict matchers, each unambiguous:

  1. EXACT match — the normalised transcript equals one of the trigger
     phrases verbatim (after lowercasing + whitespace squeeze).
  2. SINGLE-WORD match — the transcript is a single token that is in the
     known short-command set ("suivant", "next", "pause", …).
  3. GOTO — regex extracts a slide number from a GOTO phrase.

The previous version also matched on substring containment ("next" found
anywhere in the sentence) and returned confidence floats (0.95 / 0.8 /
0.7). Both were retired:

  - Substring containment over-fires on natural utterances ("I'd like to
    repeat the explanation…" → repeat command).
  - The confidence floats were hand-picked to make a downstream threshold
    work, not measured.

The function now returns either ``{"action": ...}`` (a real match) or
``None`` (no command). No tuning knob.
"""
from __future__ import annotations

import re
from typing import Optional


TRIGGERS = {
    "next":            ["suivant", "suivante", "suivant s'il vous plaît", "next", "next slide", "suivant slide", "suiv"],
    "previous":        ["précédent", "précédente", "avant", "previous", "back", "revenir"],
    "repeat":          ["répète", "répéter", "recommence", "répète s'il vous plaît", "repeat", "again", "redis"],
    "quiz":            ["quiz", "test", "interroge", "teste-moi", "teste moi", "test me", "quiz me"],
    "explain":         ["explique", "explique-moi", "explique moi", "explain", "can you explain", "explain this"],
    "pause":           ["pause", "arrête", "stop", "arrete", "mets en pause"],
    "resume":          ["reprendre", "reprends", "continue", "resume", "reprise", "poursuis"],
    "flag_confusion":  ["c'est pas intelligent", "pas intelligent", "tu n'es pas intelligent",
                        "je suis perdu", "je ne comprends pas", "c'est nul", "pas utile"],
}

# Single-word triggers — only fire when the WHOLE transcript is one
# of these words (so "I want to next time…" does NOT trigger "next").
_SINGLE_WORD_TRIGGERS = {
    "suivant": "next", "suivante": "next", "next": "next", "suiv": "next",
    "précédent": "previous", "previous": "previous", "back": "previous",
    "répète": "repeat", "repeat": "repeat", "again": "repeat",
    "pause": "pause", "arrête": "pause", "stop": "pause",
    "reprends": "resume", "reprendre": "resume", "continue": "resume",
}

GOTO_PATTERNS = [
    "passe au slide", "passe à la slide", "passe au diapositive",
    "aller au slide", "aller à la diapositive", "va à la diapositive",
    "va au slide", "go to slide", "slide", "diapositive", "page",
]


def _normalize(text: str) -> str:
    return " ".join((text or "").lower().strip().split())


def parse_voice_command(text: str, language: str = "fr") -> Optional[dict]:
    """Parse a short transcript and return a command dict or None.

    Returns ``{"action": str, "slide": int (only for goto)}`` on a match,
    None otherwise. The ``language`` argument is accepted for API
    compatibility but not currently used (triggers are bilingual).
    """
    if not text:
        return None
    norm = _normalize(text)

    # 1. Exact match (full transcript == known phrase)
    for action, phrases in TRIGGERS.items():
        for p in phrases:
            if norm == p:
                return {"action": action}

    # 2. GOTO slide (regex with captured number)
    for token in GOTO_PATTERNS:
        m = re.search(re.escape(token) + r"\s*(?:numéro\s*)?(?:n°\s*)?(\d{1,4})\b", norm)
        if m:
            try:
                return {"action": "goto", "slide": int(m.group(1))}
            except Exception:
                pass

    # 3. Single-word match (whole transcript is exactly one trigger word)
    words = norm.split()
    if len(words) == 1 and words[0] in _SINGLE_WORD_TRIGGERS:
        return {"action": _SINGLE_WORD_TRIGGERS[words[0]]}

    return None
