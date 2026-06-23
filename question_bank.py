"""
The raw questions live in five files (``_qbank_part1_expanded.py`` ...
``_qbank_part5_expanded.py``), each exposing a ``QUESTIONS`` list — the expanded
bank of 45 topics x 15 graded-difficulty MCQs = 675.  This module:

  * concatenates them into a single ordered list,
  * attaches *assumed* per-question metadata used by the student simulator
    (``base_time`` and a per-ability ``difficulty_profile``),
  * validates the bank against the curriculum (45 topics x 4 questions = 180),
  * exposes convenience lookups (by id, by topic).

There are **no sub-topics used in the RL pipeline** - the agent reasons over the
45 canonical topics only.  ``subtopic`` text is preserved purely for human
inspection / reporting and never enters the observation or reward.

This file is intentionally self-contained (no import from the student
simulator) so there is no circular dependency:

    curriculum.py        <- (no deps)
    question_bank.py     <- curriculum
    student_simulator.py <- curriculum, question_bank
    mcq_env.py           <- curriculum, question_bank, student_simulator
"""
from __future__ import annotations

import importlib
from copy import deepcopy
from typing import Any

import curriculum

# Ability scale shared across the whole project.
# NOTE: widened from 10..30 to 10..50 to give strong students headroom (they
# no longer saturate/clip at the old ceiling).  All ability-dependent formulas
# below are normalised by (MAX_ABILITY - MIN_ABILITY), so they rescale
# automatically.  Consequence to remember: a given effective ability now sits
# lower on the normalised scale (e.g. 30 is the midpoint, not the top), so the
# simulator's perceived-difficulty calibration shifts — the models MUST be
# retrained for this range to take effect.
MIN_ABILITY = 10
MAX_ABILITY = 50
DIFFICULTY_LOW = 0.0
DIFFICULTY_HIGH = 16.0
OPTIONS = ["A", "B", "C", "D"]

_PART_MODULES = [
    "_qbank_part1_expanded",
    "_qbank_part2_expanded",
    "_qbank_part3_expanded",
    "_qbank_part4_expanded",
    "_qbank_part5_expanded",
]


# ---------------------------------------------------------------------------
# Raw bank assembly
# ---------------------------------------------------------------------------
def _load_raw_questions() -> list[dict[str, Any]]:
    questions: list[dict[str, Any]] = []
    for name in _PART_MODULES:
        module = importlib.import_module(name)
        questions.extend(deepcopy(module.QUESTIONS))
    return questions


# ---------------------------------------------------------------------------
# Difficulty / time helpers (assumed metadata)
# ---------------------------------------------------------------------------
def clip(value: float, low: float, high: float) -> float:
    return float(max(low, min(high, value)))


def make_default_distractor_strength(correct_answer: str) -> dict[str, float]:
    return {option: 1.0 for option in OPTIONS if option != correct_answer}


def ability_to_difficulty_scale(ability: int | float) -> float:
    """Map ability 10..30 onto the 0..DIFFICULTY_HIGH perceived-difficulty scale."""
    ability = clip(float(ability), MIN_ABILITY, MAX_ABILITY)
    ability_norm = (ability - MIN_ABILITY) / (MAX_ABILITY - MIN_ABILITY)
    return DIFFICULTY_LOW + ability_norm * (DIFFICULTY_HIGH - DIFFICULTY_LOW)


def build_difficulty_profile(inherent_difficulty: float) -> dict[int, dict[str, float]]:
    """Per-ability {center, spread} of perceived difficulty for a question.

    Technical questions climb more sharply for hard items while preserving
    separation among easy ones (same shaping used in the earlier simulator).
    """
    high_excess = max(0.0, inherent_difficulty - 7.0)
    low_ability_center = clip(
        1.24 * inherent_difficulty + 1.15 + 0.55 * high_excess + 0.10 * high_excess ** 2,
        1.2,
        14.8,
    )
    high_ability_center = clip(
        0.52 * inherent_difficulty + 0.55 + 0.24 * high_excess,
        0.8,
        7.4,
    )

    profile: dict[int, dict[str, float]] = {}
    for ability in range(MIN_ABILITY, MAX_ABILITY + 1):
        t = (ability - MIN_ABILITY) / (MAX_ABILITY - MIN_ABILITY)
        center = low_ability_center * (1.0 - t) + high_ability_center * t
        spread = 0.32 + 0.045 * inherent_difficulty + 0.035 * high_excess + 0.018 * abs(center - 6.0)
        spread = clip(spread, 0.32, 1.12)
        profile[ability] = {"center": round(center, 2), "spread": round(spread, 2)}
    return profile


def estimate_base_time(question: dict[str, Any], inherent_difficulty: float) -> float:
    """Assumed base response time (seconds): harder + longer => more time."""
    text = " ".join(
        str(question.get(key, ""))
        for key in ("question", "option_A", "option_B", "option_C", "option_D")
    )
    length_factor = min(len(text) / 70.0, 10.0)
    high_excess = max(0.0, inherent_difficulty - 7.0)
    high_difficulty_time = 4.0 * (high_excess ** 1.45)
    advanced_reasoning_time = 2.0 * (max(0.0, inherent_difficulty - 8.5) ** 2)
    base_time = (
        12.0
        + inherent_difficulty * 4.1
        + length_factor
        + high_difficulty_time
        + advanced_reasoning_time
    )
    return round(base_time, 2)


def attach_metadata(raw_questions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    questions = deepcopy(raw_questions)
    for question in questions:
        inherent = float(question["inherent_difficulty"])
        question["base_time"] = estimate_base_time(question, inherent)
        question["difficulty_profile"] = build_difficulty_profile(inherent)
        if "distractor_strength" not in question or not question["distractor_strength"]:
            question["distractor_strength"] = make_default_distractor_strength(question["answer"])
        question.setdefault("subtopic", "")
    return questions


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate_bank(questions: list[dict[str, Any]], *, strict: bool = True) -> dict[str, Any]:
    problems: list[str] = []

    ids = [q["question_id"] for q in questions]
    dup = {i for i in ids if ids.count(i) > 1}
    if dup:
        problems.append(f"duplicate question_ids: {sorted(dup)}")

    by_topic: dict[str, list[dict[str, Any]]] = {}
    for q in questions:
        by_topic.setdefault(q["topic"], []).append(q)

    for topic in curriculum.TOPICS:
        n = len(by_topic.get(topic, []))
        if n != 15:
            problems.append(f"topic {topic!r} has {n} questions (expected 15)")

    unknown = set(by_topic) - set(curriculum.TOPICS)
    if unknown:
        problems.append(f"questions reference unknown topics: {sorted(unknown)}")

    for topic, qs in by_topic.items():
        diffs = sorted(float(q["inherent_difficulty"]) for q in qs)
        if len(set(diffs)) < len(diffs):
            problems.append(f"topic {topic!r} has repeated difficulty values: {diffs}")

    if strict and problems:
        raise ValueError("Question-bank validation failed:\n  - " + "\n  - ".join(problems))

    return {
        "n_questions": len(questions),
        "n_topics": len(by_topic),
        "problems": problems,
    }


# ---------------------------------------------------------------------------
# Public objects
# ---------------------------------------------------------------------------
QUESTIONS_WITH_METADATA: list[dict[str, Any]] = attach_metadata(_load_raw_questions())
validate_bank(QUESTIONS_WITH_METADATA, strict=True)

QUESTION_BY_ID: dict[str, dict[str, Any]] = {
    q["question_id"]: q for q in QUESTIONS_WITH_METADATA
}

# Inherent-difficulty span of the bank, and a linear map from ability to the
# difficulty a learner at that ability is expected to handle.  Used by the
# difficulty-aware (Elo-style) ability update so the effective ability converges
# to the difficulty level the student can actually answer (LIMITATIONS C1).
BANK_MIN_DIFF = min(float(q["inherent_difficulty"]) for q in QUESTIONS_WITH_METADATA)
BANK_MAX_DIFF = max(float(q["inherent_difficulty"]) for q in QUESTIONS_WITH_METADATA)


def ability_to_bank_difficulty(ability: float) -> float:
    """Map ability in [MIN_ABILITY, MAX_ABILITY] to the bank difficulty scale."""
    norm = (clip(float(ability), MIN_ABILITY, MAX_ABILITY) - MIN_ABILITY) / (
        MAX_ABILITY - MIN_ABILITY
    )
    return BANK_MIN_DIFF + norm * (BANK_MAX_DIFF - BANK_MIN_DIFF)

# topic -> list of its questions, sorted easy -> hard by inherent_difficulty.
QUESTIONS_BY_TOPIC: dict[str, list[dict[str, Any]]] = {}
for _q in QUESTIONS_WITH_METADATA:
    QUESTIONS_BY_TOPIC.setdefault(_q["topic"], []).append(_q)
for _topic in QUESTIONS_BY_TOPIC:
    QUESTIONS_BY_TOPIC[_topic].sort(key=lambda q: float(q["inherent_difficulty"]))


def questions_for_topic(topic: str) -> list[dict[str, Any]]:
    return QUESTIONS_BY_TOPIC.get(topic, [])


if __name__ == "__main__":
    report = validate_bank(QUESTIONS_WITH_METADATA, strict=True)
    print("Question bank OK")
    for key, value in report.items():
        print(f"  {key}: {value}")
    print("\nExample topic difficulty ladders:")
    for topic in curriculum.TOPICS[:3]:
        ladder = [round(q["inherent_difficulty"], 1) for q in questions_for_topic(topic)]
        times = [q["base_time"] for q in questions_for_topic(topic)]
        print(f"  {topic}: difficulties={ladder} base_times={times}")
