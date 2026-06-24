"""
difficulty_policy.py — Richer target-difficulty computation for the LLM branch.

The RL model still selects the *topic*; this module computes how hard the next
question should be.  It replaces the single-line ability_to_target_difficulty()
call with a formula that also incorporates:

  * recent per-question accuracy (last 3 / last 4)
  * recent average time ratio (last 3 / last 4)
  * consecutive correct / wrong streaks
  * a small random dither so the target is never identical twice

The result stays in the bank's difficulty range and is used by BOTH the bank
branch (selecting the closest question) and the LLM branch (instructing the
LLM what level to target).

Public API
----------
    from difficulty_policy import compute_target_difficulty

    target = compute_target_difficulty(state)

where ``state`` is the dict returned by ``env._get_state_dict()`` or built
manually during a real session.  Required keys:

    effective_ability        float  10..30
    last_3_accuracy          float  0..1   (0 if < 3 questions answered)
    last_4_accuracy          float  0..1
    last_3_avg_time_ratio    float  ≥0     (0 if < 3 questions answered)
    last_4_avg_time_ratio    float  ≥0
    consecutive_correct      int    ≥0
    consecutive_wrong        int    ≥0
"""
from __future__ import annotations

import random

from mcq_env import ability_to_target_difficulty, BANK_MIN_DIFF, BANK_MAX_DIFF
from student_simulator import MIN_ABILITY, MAX_ABILITY, clip


# ---------------------------------------------------------------------------
# Tuning knobs
# ---------------------------------------------------------------------------
ACCURACY_WEIGHT = 0.25       # how much recent accuracy shifts the target
TIME_WEIGHT = 0.10           # how much time pressure shifts the target
STREAK_WEIGHT = 0.08         # per-question shift per streak count
MAX_STREAK_SHIFT = 0.5       # cap streak contribution (in difficulty units)
DITHER_SCALE = 0.15          # Gaussian dither std-dev (difficulty units)


def compute_target_difficulty(state: dict) -> float:
    """
    Compute a per-question target difficulty blending ability and recent
    performance signals.

    Returns a float clamped to [BANK_MIN_DIFF, BANK_MAX_DIFF].
    """
    ability: float = float(state.get("effective_ability", 20.0))
    last3_acc: float = float(state.get("last_3_accuracy", 0.5))
    last4_acc: float = float(state.get("last_4_accuracy", 0.5))
    last3_time: float = float(state.get("last_3_avg_time_ratio", 1.0))
    last4_time: float = float(state.get("last_4_avg_time_ratio", 1.0))
    consec_correct: int = int(state.get("consecutive_correct", 0))
    consec_wrong: int = int(state.get("consecutive_wrong", 0))

    # --- Base from ability ---------------------------------------------------
    base = ability_to_target_difficulty(ability)

    # --- Accuracy adjustment ------------------------------------------------
    # avg recent accuracy: > 0.75 → push harder; < 0.45 → ease off
    avg_acc = (last3_acc + last4_acc) / 2.0
    acc_shift = (avg_acc - 0.60) * ACCURACY_WEIGHT * (BANK_MAX_DIFF - BANK_MIN_DIFF)

    # --- Time-ratio adjustment ----------------------------------------------
    # time_ratio < 1 means fast → likely easy → push harder
    avg_time = (last3_time + last4_time) / 2.0
    time_shift = (1.0 - min(avg_time, 2.0)) * TIME_WEIGHT * (BANK_MAX_DIFF - BANK_MIN_DIFF)

    # --- Streak adjustment --------------------------------------------------
    streak_shift = 0.0
    if consec_correct >= 2:
        streak_shift = min(consec_correct * STREAK_WEIGHT, MAX_STREAK_SHIFT)
    elif consec_wrong >= 2:
        streak_shift = -min(consec_wrong * STREAK_WEIGHT, MAX_STREAK_SHIFT)

    # --- Small random dither (so LLM doesn't get exactly the same target) ---
    dither = random.gauss(0, DITHER_SCALE)

    target = base + acc_shift + time_shift + streak_shift + dither
    return float(clip(target, BANK_MIN_DIFF, BANK_MAX_DIFF))


def difficulty_label(difficulty: float) -> str:
    """Human-readable difficulty band for prompts."""
    span = BANK_MAX_DIFF - BANK_MIN_DIFF
    norm = (difficulty - BANK_MIN_DIFF) / (span + 1e-9)
    if norm < 0.25:
        return "easy (introductory)"
    elif norm < 0.50:
        return "moderate"
    elif norm < 0.75:
        return "challenging"
    else:
        return "hard (advanced)"
