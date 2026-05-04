"""Math → spoken text conversion at TTS output time.

# Why this lives here

Math is information. Indexing "x²" as the literal token "x²" preserves
retrieval precision; rewriting it to "x squared" before indexing
destroys it (a student searching for "x²" finds nothing). The previous
``_wordify_math`` ran at index time on a hardcoded English-only symbol
table — wrong layer and wrong scope. This module runs at the **output
layer** (after the LLM, before TTS) so the index stays clean and the
voice still sounds natural.

# Architecture (3-tier)

  Tier 1 — LaTeX block parser (sympy-based when available)
           Detects ``$..$``, ``$$..$$``, ``\\(..\\)``, ``\\[..\\]``,
           parses with ``sympy.parsing.latex.parse_latex``, then walks
           the expression tree to produce localized speech. Falls
           through to Tier 2 on parse failure.

  Tier 2 — LaTeX command rewriting (regex)
           Targets common commands the sympy parser misses or that
           don't need a full parse: ``\frac{a}{b}``, ``\sqrt{x}``,
           ``^{n}``, ``_{i}``, ``\sum``, ``\int`` etc. Bilingual.

  Tier 3 — Unicode symbol substitution
           Last pass on bare unicode symbols (∫ ∑ √ α β …) that
           survived the previous tiers. Bilingual.

Each tier is opt-out independent: if sympy isn't installed, Tiers 2 + 3
still run and cover ~80% of the practical cases (slides rarely contain
fully-formed LaTeX expressions; they mostly mix unicode with prose).

# Bilingual lexicons

The unicode and command lexicons carry both ``fr`` and ``en`` entries.
The caller passes ``lang`` and we route accordingly. No single-language
hardcoding inside the substitution function.

# References

  - Mathematical Markup Language (MathML), W3C Recommendation 2014.
    Speech rules per MathML are the canonical reference for "how to say"
    LaTeX expressions.
  - Speech Synthesis Markup Language (SSML), W3C 2010, ``<say-as>``
    semantics for math.
  - Sympy parsing.latex (since 1.5), the de-facto Python LaTeX parser
    when ANTLR runtime is installed.
  - "Reading Mathematics: An Empirical Investigation" (Soiffer 2005)
    on the linguistics of spoken math.
"""
from __future__ import annotations

import logging
import re
from typing import Optional

log = logging.getLogger("audio.math_speech")


# ── Unicode lexicon (Tier 3) ───────────────────────────────────────────
# Each symbol maps to a (fr, en) tuple. Entries that take a parameter
# (subscript / superscript) are handled by Tier 2 via regex and don't
# appear here.

_UNICODE_LEXICON: dict[str, tuple[str, str]] = {
    # Operators
    "∫": ("intégrale", "integral"),
    "∑": ("somme",     "sum"),
    "∏": ("produit",   "product"),
    "∇": ("gradient",  "gradient"),
    "∂": ("dérivée partielle", "partial derivative"),
    "√": ("racine carrée", "square root"),
    "∞": ("infini",    "infinity"),
    # Comparators
    "≤": ("inférieur ou égal à", "less than or equal to"),
    "≥": ("supérieur ou égal à", "greater than or equal to"),
    "≠": ("différent de",        "not equal to"),
    "≈": ("environ",             "approximately"),
    "≡": ("équivalent à",        "equivalent to"),
    # Set theory
    "∈": ("appartient à",        "in"),
    "∉": ("n'appartient pas à",  "not in"),
    "⊂": ("inclus dans",         "subset of"),
    "⊆": ("inclus ou égal",      "subset or equal"),
    "∪": ("union",               "union"),
    "∩": ("intersection",        "intersection"),
    "∅": ("ensemble vide",       "empty set"),
    "ℕ": ("N",                   "N"),
    "ℤ": ("Z",                   "Z"),
    "ℝ": ("R",                   "R"),
    "ℂ": ("C",                   "C"),
    "ℚ": ("Q",                   "Q"),
    # Logic
    "∀": ("pour tout",           "for all"),
    "∃": ("il existe",           "there exists"),
    "∧": ("et",                  "and"),
    "∨": ("ou",                  "or"),
    "¬": ("non",                 "not"),
    "⇒": ("implique",            "implies"),
    "⇔": ("équivalent à",        "if and only if"),
    "→": ("vers",                "to"),
    "↔": ("équivalent à",        "iff"),
    # Greek (lowercase): keep the Greek letter name verbatim — it's the
    # standard register in spoken science. Localized only where the
    # French and English names diverge meaningfully.
    "α": ("alpha",   "alpha"),
    "β": ("bêta",    "beta"),
    "γ": ("gamma",   "gamma"),
    "δ": ("delta",   "delta"),
    "ε": ("epsilon", "epsilon"),
    "ζ": ("zêta",    "zeta"),
    "η": ("êta",     "eta"),
    "θ": ("thêta",   "theta"),
    "ι": ("iota",    "iota"),
    "κ": ("kappa",   "kappa"),
    "λ": ("lambda",  "lambda"),
    "μ": ("mu",      "mu"),
    "ν": ("nu",      "nu"),
    "ξ": ("xi",      "xi"),
    "ο": ("omicron", "omicron"),
    "π": ("pi",      "pi"),
    "ρ": ("rho",     "rho"),
    "σ": ("sigma",   "sigma"),
    "τ": ("tau",     "tau"),
    "υ": ("upsilon", "upsilon"),
    "φ": ("phi",     "phi"),
    "χ": ("chi",     "chi"),
    "ψ": ("psi",     "psi"),
    "ω": ("oméga",   "omega"),
    # Greek (uppercase) — used capitalised in formulas
    "Α": ("Alpha",   "Alpha"),
    "Β": ("Bêta",    "Beta"),
    "Γ": ("Gamma",   "Gamma"),
    "Δ": ("Delta",   "Delta"),
    "Θ": ("Thêta",   "Theta"),
    "Λ": ("Lambda",  "Lambda"),
    "Π": ("Pi",      "Pi"),
    "Σ": ("Sigma",   "Sigma"),
    "Φ": ("Phi",     "Phi"),
    "Ψ": ("Psi",     "Psi"),
    "Ω": ("Oméga",   "Omega"),
    # Superscripts (small numbers) — handled here as fallback for chars
    # that survive Tier 2 regex (e.g. ⁰, ⁵, …).
    "⁰": ("zéro",    "zero"),
    "¹": ("un",      "one"),
    "²": ("au carré", "squared"),
    "³": ("au cube",  "cubed"),
    "⁴": ("puissance quatre", "to the fourth"),
    "⁵": ("puissance cinq",   "to the fifth"),
    "⁶": ("puissance six",    "to the sixth"),
    "⁷": ("puissance sept",   "to the seventh"),
    "⁸": ("puissance huit",   "to the eighth"),
    "⁹": ("puissance neuf",   "to the ninth"),
    # Subscript digits — rare in slides, but harmless to support
    "₀": ("zéro",  "zero"),
    "₁": ("un",    "one"),
    "₂": ("deux",  "two"),
    "₃": ("trois", "three"),
    "₄": ("quatre", "four"),
    # Other operators
    "·": ("fois",   "times"),
    "×": ("fois",   "times"),
    "÷": ("divisé par", "divided by"),
    "±": ("plus ou moins", "plus or minus"),
}


# ── LaTeX command lexicon (Tier 2) ─────────────────────────────────────

_LATEX_COMMAND_LEXICON: dict[str, tuple[str, str]] = {
    # Set names
    "\\mathbb{R}":  ("R", "R"),
    "\\mathbb{N}":  ("N", "N"),
    "\\mathbb{Z}":  ("Z", "Z"),
    "\\mathbb{Q}":  ("Q", "Q"),
    "\\mathbb{C}":  ("C", "C"),
    # Operators
    "\\sum":        ("somme de", "sum of"),
    "\\prod":       ("produit de", "product of"),
    "\\int":        ("intégrale de", "integral of"),
    "\\partial":    ("dérivée partielle", "partial derivative"),
    "\\nabla":      ("gradient", "gradient"),
    "\\infty":      ("infini", "infinity"),
    # Greek letters as commands
    "\\alpha":      ("alpha",   "alpha"),
    "\\beta":       ("bêta",    "beta"),
    "\\gamma":      ("gamma",   "gamma"),
    "\\delta":      ("delta",   "delta"),
    "\\epsilon":    ("epsilon", "epsilon"),
    "\\theta":      ("thêta",   "theta"),
    "\\lambda":     ("lambda",  "lambda"),
    "\\mu":         ("mu",      "mu"),
    "\\pi":         ("pi",      "pi"),
    "\\sigma":      ("sigma",   "sigma"),
    "\\phi":        ("phi",     "phi"),
    "\\omega":      ("oméga",   "omega"),
    "\\Delta":      ("Delta",   "Delta"),
    "\\Sigma":      ("Sigma",   "Sigma"),
    "\\Pi":         ("Pi",      "Pi"),
    "\\Omega":      ("Oméga",   "Omega"),
    # Comparators
    "\\leq":        ("inférieur ou égal à", "less than or equal to"),
    "\\geq":        ("supérieur ou égal à", "greater than or equal to"),
    "\\neq":        ("différent de",        "not equal to"),
    "\\approx":     ("environ",             "approximately"),
    "\\equiv":      ("équivalent à",        "equivalent to"),
    # Logic
    "\\forall":     ("pour tout",  "for all"),
    "\\exists":     ("il existe",  "there exists"),
    "\\implies":    ("implique",   "implies"),
    "\\iff":        ("équivalent à", "if and only if"),
    "\\to":         ("vers", "to"),
    "\\rightarrow": ("vers", "to"),
    "\\leftarrow":  ("vers", "to"),
    # Set membership
    "\\in":         ("appartient à",       "in"),
    "\\notin":      ("n'appartient pas à", "not in"),
    "\\subset":     ("inclus dans",        "subset of"),
    "\\cup":        ("union",              "union"),
    "\\cap":        ("intersection",       "intersection"),
    # Operators
    "\\cdot":       ("fois",         "times"),
    "\\times":      ("fois",         "times"),
    "\\div":        ("divisé par",   "divided by"),
    "\\pm":         ("plus ou moins", "plus or minus"),
}


# ── Tier 1: LaTeX block parser via sympy ───────────────────────────────


def _try_sympy_parse(latex: str, lang: str) -> Optional[str]:
    """Attempt to parse a LaTeX expression with sympy and verbalize it.

    Returns ``None`` on parse failure or if sympy isn't available — the
    caller should fall through to Tier 2/3 in that case. Sympy isn't a
    hard dependency; we lazy-import.
    """
    try:
        from sympy.parsing.latex import parse_latex   # type: ignore
    except Exception:
        return None
    try:
        expr = parse_latex(latex)
        if expr is None:
            return None
        # Use sympy's built-in pretty printer in "ASCII" mode and then
        # strip residual punctuation. For complex expressions this is a
        # decent first pass; deep semantic verbalization (MathML →
        # speech) is out of scope for this module.
        spoken = str(expr)
        # Replace common operators by their spoken form via the same
        # lexicon used in Tier 2/3, so the result is bilingual.
        return _apply_lexicon_passes(spoken, lang)
    except Exception as exc:                                              # noqa: BLE001
        log.debug("sympy LaTeX parse failed: %s", exc)
        return None


# ── Tier 2: regex-driven LaTeX command rewriting ───────────────────────


_FRAC_PATTERN = re.compile(r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}")
_SQRT_PATTERN = re.compile(r"\\sqrt\s*\{([^{}]+)\}")
# Subscripts / superscripts: anchor the operator to an *isolated single-letter
# variable* (a math-style identifier like x_i, w_i, σ_A), not to a character
# in the middle of a longer token. The previous unbounded patterns matched
# any "_X" or "^X" anywhere, so when the LLM emitted compound identifiers
# from planner ids ("concept_main") or placeholder concept names
# ("Indice_1", "Method_Indice_1"), this pass tore them apart letter by
# letter — e.g. "Method_Indice_1" → "Method indice I ndice indice 1".
# The lookbehind (?<![A-Za-z0-9]) requires the variable letter to start a
# token; the trailing \b stops the match at a word boundary so we don't
# bite into legitimate words like "x_index".
_SUPER_PATTERN = re.compile(r"(?<![A-Za-z0-9])([A-Za-z0-9])\^\s*\{([^{}]+)\}")
_SUPER_SHORT = re.compile(r"(?<![A-Za-z0-9])([A-Za-z0-9])\^\s*([A-Za-z0-9])\b")
_SUB_PATTERN = re.compile(r"(?<![A-Za-z0-9])([A-Za-z])_\s*\{([^{}]+)\}")
_SUB_SHORT = re.compile(r"(?<![A-Za-z0-9])([A-Za-z])_\s*([A-Za-z0-9])\b")

_LATEX_BLOCK_PATTERNS = (
    re.compile(r"\$\$([^$]+)\$\$"),
    re.compile(r"\$([^$\n]+)\$"),
    re.compile(r"\\\[(.+?)\\\]", re.DOTALL),
    re.compile(r"\\\((.+?)\\\)"),
)


def _verbalize_frac(match: re.Match, lang: str) -> str:
    num = match.group(1)
    den = match.group(2)
    if lang == "fr":
        return f" {num} divisé par {den} "
    return f" {num} over {den} "


def _verbalize_sqrt(match: re.Match, lang: str) -> str:
    inner = match.group(1)
    if lang == "fr":
        return f" racine carrée de {inner} "
    return f" square root of {inner} "


def _verbalize_super(match: re.Match, lang: str) -> str:
    var = match.group(1)
    exp = match.group(2)
    # Common cases get a clean wording
    if exp == "2":
        return f"{var} au carré " if lang == "fr" else f"{var} squared "
    if exp == "3":
        return f"{var} au cube " if lang == "fr" else f"{var} cubed "
    if lang == "fr":
        return f"{var} puissance {exp} "
    return f"{var} to the power of {exp} "


def _verbalize_sub(match: re.Match, lang: str) -> str:
    var = match.group(1)
    sub = match.group(2)
    if lang == "fr":
        return f"{var} indice {sub} "
    return f"{var} subscript {sub} "


def _apply_lexicon_passes(text: str, lang: str) -> str:
    """Apply Tier 2 + Tier 3 substitutions in order."""
    lang_idx = 0 if lang == "fr" else 1

    # Tier 2a: structural commands (fractions / roots / scripts).
    text = _FRAC_PATTERN.sub(lambda m: _verbalize_frac(m, lang), text)
    text = _SQRT_PATTERN.sub(lambda m: _verbalize_sqrt(m, lang), text)
    text = _SUPER_PATTERN.sub(lambda m: _verbalize_super(m, lang), text)
    text = _SUPER_SHORT.sub(lambda m: _verbalize_super(m, lang), text)
    text = _SUB_PATTERN.sub(lambda m: _verbalize_sub(m, lang), text)
    text = _SUB_SHORT.sub(lambda m: _verbalize_sub(m, lang), text)

    # Tier 2b: lexical commands (\sum, \alpha, \leq…). Sort by length
    # descending so longer commands win over shorter prefixes
    # (\mathbb{R} before \mathbb).
    for cmd in sorted(_LATEX_COMMAND_LEXICON, key=len, reverse=True):
        if cmd in text:
            text = text.replace(cmd, f" {_LATEX_COMMAND_LEXICON[cmd][lang_idx]} ")

    # Strip any residual single backslashes (commands we don't know)
    text = re.sub(r"\\([a-zA-Z]+)", r"\1", text)

    # Tier 3: unicode symbols.
    for sym, (fr, en) in _UNICODE_LEXICON.items():
        if sym in text:
            text = text.replace(sym, f" {fr if lang == 'fr' else en} ")

    # Whitespace cleanup
    text = re.sub(r"\s{2,}", " ", text).strip()
    return text


# ── Public API ─────────────────────────────────────────────────────────


def to_speech(text: str, lang: str = "en") -> str:
    """Minimal, safe cleanup of stray LaTeX/math notation that the LLM
    failed to verbalise itself.

    The OLD pipeline ran heavy regex substitutions to convert math
    notation to spoken form (``x_i`` → ``x indice i``, ``\\frac{a}{b}`` →
    ``a sur b``, …). Two problems with that approach :

      1. Regex damages identifiers. ``Method_Indice_1`` was being torn
         apart into ``Method indice I ndice indice 1`` because the
         subscript regex ``_X`` matched any underscore-letter pattern,
         not only math contexts.
      2. The LLM has full semantic context — it knows whether ``x_i``
         is a math variable or a snake_case identifier. Asking the LLM
         to verbalise math directly (via system-prompt rules) produces
         far better results than any regex post-pass.

    The new approach :
      - The system prompts (``get_system_prompt`` / ``get_presentation_prompt``)
        instruct the LLM to convert all math to spoken form BEFORE
        producing its answer, with worked examples covering the common
        operators (subscript, superscript, sum, integral, fractions,
        Greek letters, comparisons).
      - This function is now a *safety net*: it strips the few
        residual LaTeX delimiters that occasionally slip through
        (``$``, ``\\(``, ``\\)``, ``\\[``, ``\\]``), without trying to
        verbalise anything. If the LLM ignored its instructions and
        produced ``x^2``, this function leaves it as ``x^2`` — TTS will
        say "x caret two" or similar; that's a regression we can see
        and fix at the prompt level, instead of a silent regex shred.

    Returns the input unchanged if it contains no LaTeX delimiters.
    """
    if not text:
        return text
    # Strip LaTeX block delimiters but preserve their content. Caret
    # ``^`` and underscore ``_`` are LEFT INTACT — TTS will pronounce
    # them poorly but at least no identifier gets damaged. The right
    # fix lives in the prompt, not here.
    text = text.replace("$$", " ").replace("$", " ")
    text = text.replace("\\(", " ").replace("\\)", " ")
    text = text.replace("\\[", " ").replace("\\]", " ")

    # Drop markup-like ``<tag>`` artefacts before verbalising standalone
    # ``<`` / ``>``. Patterns like ``<concept>`` or ``</answer>`` are
    # template leakage from the LLM, not math, so they get stripped.
    text = re.sub(r"</?[A-Za-z][A-Za-z0-9_-]*\s*/?>", " ", text)

    # Verbalise math comparison symbols. The LLM is *supposed* to do this
    # itself via the system prompt, but raw ``<`` / ``>`` still slip
    # through (TTS reads them as "less than" anyway, but inconsistently
    # across engines). We force a deterministic rendering. Order matters :
    # 2-char operators first so ``<=`` doesn't get split.
    if lang == "fr":
        comparators = [
            ("<=", " inférieur ou égal à "),
            (">=", " supérieur ou égal à "),
            ("≤",  " inférieur ou égal à "),
            ("≥",  " supérieur ou égal à "),
            ("≠",  " différent de "),
            ("≈",  " environ "),
            ("<",  " inférieur à "),
            (">",  " supérieur à "),
        ]
    else:
        comparators = [
            ("<=", " less than or equal to "),
            (">=", " greater than or equal to "),
            ("≤",  " less than or equal to "),
            ("≥",  " greater than or equal to "),
            ("≠",  " not equal to "),
            ("≈",  " approximately "),
            ("<",  " less than "),
            (">",  " greater than "),
        ]
    for sym, word in comparators:
        text = text.replace(sym, word)

    # Collapse whitespace introduced by the strips
    text = re.sub(r"\s{2,}", " ", text).strip()
    return text
