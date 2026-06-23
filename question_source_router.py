"""
question_source_router.py — Decides, per question, whether to pull from the
bank or generate via the LLM, according to the play-count schedule.

Schedule (v2 — first session already gets LLM questions)
---------------------------------------------------------
    p_llm = min(0.70, play_count * 0.05)

    Play 1  →  5 %  LLM
    Play 2  → 10 %  LLM
    Play 3  → 15 %  LLM
    ...
    Play 14 → 70 %  LLM  (cap)
    >= 14   → 70 %  LLM

Two mixing modes:
    exact_count (default) : pre-compute exactly floor(p_llm * n_questions)
                            LLM slots, shuffle their positions → reproducible
                            ratios, clean for defence.
    probabilistic         : each question independently draws from Bernoulli(p_llm).

Public API
----------
    from question_source_router import QuestionSourceRouter, compute_p_llm

    router = QuestionSourceRouter(play_count=1, n_questions=20, seed=42)
    for i in range(20):
        source = router.source_for_step(i)   # "bank" or "llm"
"""
from __future__ import annotations

import random
from typing import Literal

SourceType = Literal["bank", "llm"]


def compute_p_llm(play_count: int) -> float:
    """Return the LLM fraction for a given play count (1-indexed).

    Play 1 → 5 %, play 2 → 10 %, …, play 14+ → 70 % (cap).
    """
    return min(0.70, max(0.0, play_count * 0.05))


class QuestionSourceRouter:
    """
    Stateful router: call source_for_step(i) for each question index i.

    Parameters
    ----------
    play_count   : How many times the student has played (1 = first ever).
    n_questions  : Total questions in the session (default 20).
    seed         : RNG seed for reproducibility.
    probabilistic: If True, use per-question Bernoulli draws instead of
                   exact-count pre-assignment.
    """

    def __init__(
        self,
        play_count: int,
        n_questions: int = 20,
        seed: int | None = None,
        probabilistic: bool = False,
    ) -> None:
        self.play_count = play_count
        self.n_questions = n_questions
        self.probabilistic = probabilistic
        self.p_llm = compute_p_llm(play_count)

        self._rng = random.Random(seed)
        self._assignment: list[SourceType] = self._build_assignment()

    # ------------------------------------------------------------------
    def _build_assignment(self) -> list[SourceType]:
        if self.probabilistic:
            return []   # built lazily per step

        n_llm = int(self.p_llm * self.n_questions)
        llm_positions: set[int] = (
            set(self._rng.sample(range(self.n_questions), n_llm))
            if n_llm > 0 else set()
        )

        assignment: list[SourceType] = []
        for i in range(self.n_questions):
            assignment.append("llm" if i in llm_positions else "bank")
        return assignment

    # ------------------------------------------------------------------
    def source_for_step(self, step: int) -> SourceType:
        """Return 'bank' or 'llm' for question index ``step``."""
        if self.probabilistic:
            return "llm" if self._rng.random() < self.p_llm else "bank"
        if step < len(self._assignment):
            return self._assignment[step]
        return "bank"

    # ------------------------------------------------------------------
    def summary(self) -> dict:
        """Return a summary dict for logging."""
        if self.probabilistic:
            return {
                "mode": "probabilistic",
                "play_count": self.play_count,
                "p_llm": self.p_llm,
            }
        n_llm = self._assignment.count("llm")
        return {
            "mode": "exact_count",
            "play_count": self.play_count,
            "p_llm": self.p_llm,
            "n_llm_planned": n_llm,
            "n_bank_planned": self.n_questions - n_llm,
            "llm_positions": [i for i, s in enumerate(self._assignment) if s == "llm"],
        }
