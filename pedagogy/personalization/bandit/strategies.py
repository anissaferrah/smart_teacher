"""Pedagogical strategy taxonomy — the action space of the bandit.

# Why an explicit taxonomy

A bandit picks an *arm*. To do that, the arms must be enumerated and
distinguishable — both internally (the bandit needs a stable ID per
arm) and externally (the prompt-injection layer needs to know what
to add to the LLM prompt for each arm).

Six pedagogical strategies covering the standard ITS literature
(Collins & Stevens 1982, Polya 1945, Sweller 1985, Bloom 1968) :

  - **analogy**       : map the unfamiliar concept onto a familiar domain
  - **example**       : show a worked-out concrete case
  - **decomposition** : break the concept into smaller sub-concepts
  - **socratic**      : ask guiding questions instead of stating
  - **recap**         : summarize what's been seen so far
  - **simpler_words** : rephrase using everyday vocabulary

# Speech-rate as a separate action group

Speech rate (slow / normal / fast) is **orthogonal** to the strategy
choice : you can have an "analogy at slow rate" or "example at fast
rate". So the bandit could either :

  (a) work over a flattened action space (6 strategies × 3 rates = 18 arms),
  (b) decompose into two sub-bandits.

For Phase 1 we use option (a) for simplicity — fewer abstractions,
and Thompson sampling handles the larger arm space cheaply.

# Stability

The ``StrategyAction`` enum values are persisted in Postgres via the
RL logger. Renaming a value is a schema migration. Add new actions at
the end; never reorder.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterator


class Strategy(str, Enum):
    """Pedagogical strategy — what kind of explanation to produce."""
    ANALOGY        = "analogy"
    EXAMPLE        = "example"
    DECOMPOSITION  = "decomposition"
    SOCRATIC       = "socratic"
    RECAP          = "recap"
    SIMPLER_WORDS  = "simpler_words"


class SpeechRate(str, Enum):
    """Speech rate adaptation — orthogonal to the strategy."""
    SLOW   = "slow"
    NORMAL = "normal"
    FAST   = "fast"


@dataclass(frozen=True)
class StrategyAction:
    """One arm of the bandit : a (strategy, speech_rate) pair."""

    strategy: Strategy
    speech_rate: SpeechRate

    @property
    def arm_id(self) -> str:
        """Stable identifier used as a dict key and Postgres column value."""
        return f"{self.strategy.value}:{self.speech_rate.value}"

    @classmethod
    def from_arm_id(cls, arm_id: str) -> "StrategyAction":
        strat, rate = arm_id.split(":")
        return cls(strategy=Strategy(strat), speech_rate=SpeechRate(rate))

    def to_dict(self) -> dict:
        return {"strategy": self.strategy.value, "speech_rate": self.speech_rate.value}


def all_actions() -> Iterator[StrategyAction]:
    """Enumerate the full action space (6 × 3 = 18 arms)."""
    for strat in Strategy:
        for rate in SpeechRate:
            yield StrategyAction(strategy=strat, speech_rate=rate)


def action_from_arm_id(arm_id: str) -> StrategyAction | None:
    """Reconstruct a StrategyAction from its arm_id string.
    
    arm_id format: "strategy_value:rate_value"
    Example: "socratic:fast" -> StrategyAction(Strategy.SOCRATIC, SpeechRate.FAST)
    Returns None if arm_id is invalid.
    """
    try:
        parts = arm_id.split(":")
        if len(parts) != 2:
            return None
        strategy = Strategy(parts[0])
        rate = SpeechRate(parts[1])
        return StrategyAction(strategy, rate)
    except (ValueError, KeyError, AttributeError):
        return None


# ── Prompt fragments per strategy (FR + EN) ────────────────────────────
#
# These fragments are injected into the LLM prompt when the bandit
# selects the corresponding strategy. They're not "templates" in the
# old _REFORM_INTROS sense — they're *style hints* given on top of the
# main task prompt. The LLM still generates freely; the fragment biases
# the register, not the content.

STRATEGY_PROMPT_FR: dict[Strategy, str] = {
    Strategy.ANALOGY: (
        "Construis ta réponse autour d'une ANALOGIE concrète : "
        "compare le concept à un objet ou une situation du quotidien. "
        "Précise rapidement en quoi l'analogie tient (et où elle s'arrête)."
    ),
    Strategy.EXAMPLE: (
        "Centre ta réponse sur un EXEMPLE résolu : montre le concept "
        "appliqué à un cas concret, étape par étape, avec des chiffres "
        "ou des cas réels si possible."
    ),
    Strategy.DECOMPOSITION: (
        "Décompose le concept en SOUS-CONCEPTS plus petits. Présente-les "
        "dans l'ordre logique : prérequis d'abord, puis l'idée principale, "
        "puis les conséquences."
    ),
    Strategy.SOCRATIC: (
        "Adopte un style SOCRATIQUE : pose 1 ou 2 questions guidantes "
        "AVANT de donner la réponse, pour que l'étudiant déduise lui-même "
        "une partie. Termine par une réponse claire."
    ),
    Strategy.RECAP: (
        "Fais un RÉCAPITULATIF synthétique des points clés vus précédemment "
        "AVANT de répondre. Ancre la nouvelle information dans ce résumé."
    ),
    Strategy.SIMPLER_WORDS: (
        "Reformule avec un VOCABULAIRE SIMPLE et accessible. Évite le "
        "jargon technique non indispensable. Garde les termes spécifiques, "
        "mais définis-les en langage courant."
    ),
}


STRATEGY_PROMPT_EN: dict[Strategy, str] = {
    Strategy.ANALOGY: (
        "Build your answer around a concrete ANALOGY: compare the concept "
        "to an everyday object or situation. Briefly state where the analogy "
        "holds (and where it breaks)."
    ),
    Strategy.EXAMPLE: (
        "Center your answer on a worked EXAMPLE: show the concept applied "
        "to a concrete case, step by step, with numbers or real cases if "
        "possible."
    ),
    Strategy.DECOMPOSITION: (
        "DECOMPOSE the concept into smaller sub-concepts. Present them in "
        "logical order: prerequisites first, then the main idea, then "
        "consequences."
    ),
    Strategy.SOCRATIC: (
        "Use a SOCRATIC style: ask 1 or 2 guiding questions BEFORE giving "
        "the answer, so the student deduces part of it themselves. End with "
        "a clear answer."
    ),
    Strategy.RECAP: (
        "First give a synthetic RECAP of the key points seen previously, "
        "THEN answer. Anchor the new information in that summary."
    ),
    Strategy.SIMPLER_WORDS: (
        "Rephrase using SIMPLE accessible vocabulary. Avoid non-essential "
        "technical jargon. Keep domain-specific terms but define them in "
        "plain language."
    ),
}


def strategy_prompt(strategy: Strategy, lang: str = "fr") -> str:
    """Return the prompt fragment for ``strategy`` in ``lang``."""
    table = STRATEGY_PROMPT_FR if (lang or "fr").lower().startswith("fr") else STRATEGY_PROMPT_EN
    return table.get(strategy, "")
