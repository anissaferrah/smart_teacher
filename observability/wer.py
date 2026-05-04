"""
╔══════════════════════════════════════════════════════════════════════╗
║       SMART TEACHER — Word Error Rate (WER) computation            ║
║                                                                      ║
║  WER = (S + D + I) / N                                              ║
║    S = substitutions, D = deletions, I = insertions                ║
║    N = number of words in reference                                ║
║                                                                      ║
║  Implémentation : édit distance Levenshtein sur tokens (mots).     ║
║  Pas de dépendance externe (jiwer optionnel pour validation).      ║
╚══════════════════════════════════════════════════════════════════════╝
"""
from __future__ import annotations

import re
from dataclasses import dataclass


# Régex normalisation : retire ponctuation et hyphens (split mots composés
# car STT outputs souvent sans hyphen — "allez-vous" → "allez vous").
# Apostrophes intérieures gardées (l'arbre → l'arbre).
_PUNCT_RE = re.compile(r"[^\w\s']", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    """Normalize text into a list of comparable word tokens.

    - Lowercase
    - Strip ponctuation (sauf apostrophes intérieures)
    - Split sur whitespace ET sur hyphens (pratique standard ASR : Whisper,
      Deepgram, etc. n'incluent quasi jamais d'hyphens dans leur output)
    """
    if not text:
        return []
    s = text.lower().strip()
    s = _PUNCT_RE.sub(" ", s)
    s = s.replace("-", " ")           # split mots composés
    return [w for w in s.split() if w]


@dataclass
class WERResult:
    wer: float                  # ratio 0.0 (perfect) to 1.0+ (worse)
    cer: float                  # character-level
    insertions: int
    deletions: int
    substitutions: int
    ref_words: int
    hyp_words: int

    def to_dict(self) -> dict:
        return {
            "wer":           round(self.wer, 4),
            "cer":           round(self.cer, 4),
            "insertions":    self.insertions,
            "deletions":     self.deletions,
            "substitutions": self.substitutions,
            "ref_words":     self.ref_words,
            "hyp_words":     self.hyp_words,
            "meets_kpi":     self.wer <= 0.05,    # cahier des charges
        }


def _edit_ops(ref_tokens: list[str], hyp_tokens: list[str]) -> tuple[int, int, int]:
    """Levenshtein DP — returns (substitutions, deletions, insertions)."""
    n, m = len(ref_tokens), len(hyp_tokens)
    if n == 0:
        return 0, 0, m              # all hyp tokens are insertions
    if m == 0:
        return 0, n, 0              # all ref tokens are deletions

    # DP table: dp[i][j] = (cost, S, D, I) at edit distance for ref[:i], hyp[:j]
    # We track only counts to keep it light
    dp = [[(0, 0, 0, 0)] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = (i, 0, i, 0)     # delete all ref[:i]
    for j in range(m + 1):
        dp[0][j] = (j, 0, 0, j)     # insert all hyp[:j]

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_tokens[i - 1] == hyp_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]                      # match (free)
            else:
                # 3 candidates: substitute, delete, insert
                sub_c, sub_s, sub_d, sub_i = dp[i - 1][j - 1]
                del_c, del_s, del_d, del_i = dp[i - 1][j]
                ins_c, ins_s, ins_d, ins_i = dp[i][j - 1]
                cands = [
                    (sub_c + 1, sub_s + 1, sub_d, sub_i),         # substitute
                    (del_c + 1, del_s, del_d + 1, del_i),         # delete
                    (ins_c + 1, ins_s, ins_d, ins_i + 1),         # insert
                ]
                dp[i][j] = min(cands, key=lambda x: x[0])

    _, S, D, I = dp[n][m]
    return S, D, I


def compute_wer(reference: str, hypothesis: str) -> WERResult:
    """Compare hypothesis (STT output) to reference (ground truth).

    Returns word-level + character-level error rates plus operation counts.
    A WER of 0.0 = perfect. WER > 1.0 possible if hypothesis has lots of insertions.
    """
    ref_tokens = _tokenize(reference)
    hyp_tokens = _tokenize(hypothesis)

    S, D, I = _edit_ops(ref_tokens, hyp_tokens)
    n_ref = len(ref_tokens)
    n_hyp = len(hyp_tokens)

    wer = (S + D + I) / n_ref if n_ref > 0 else (1.0 if n_hyp > 0 else 0.0)

    # Character-level (simpler) — useful for languages where word boundaries fuzz
    ref_chars = list(reference.lower().replace(" ", ""))
    hyp_chars = list(hypothesis.lower().replace(" ", ""))
    if ref_chars:
        Sc, Dc, Ic = _edit_ops(ref_chars, hyp_chars)
        cer = (Sc + Dc + Ic) / len(ref_chars)
    else:
        cer = 1.0 if hyp_chars else 0.0

    return WERResult(
        wer=wer, cer=cer,
        insertions=I, deletions=D, substitutions=S,
        ref_words=n_ref, hyp_words=n_hyp,
    )
