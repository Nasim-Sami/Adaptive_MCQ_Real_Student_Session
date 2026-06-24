"""
session_ability.py — Richer effective-ability update for the real session.

Why this exists
---------------
The base formula (student_simulator.effective_ability_delta) uses a STEP
function with coarse time-ratio buckets, e.g. any correct answer with
0.30 < ratio ≤ 0.60 gets exactly +0.45 — so ratio 0.35 and ratio 0.53
produce identical deltas.  It also has NO reward for a consecutive-correct
streak (only consecutive_wrong is penalised).

This module replaces that step function with a *continuous* formula that:
  • Gives a strictly different delta for every distinct time ratio
  • Rewards consecutive-correct streaks (+0.03 per answer in streak, ≤ +0.15)
  • Penalises consecutive-wrong streaks (-0.10 per answer, ≤ -0.50) — same
    weight as the base formula, but now combined with continuous time signal
  • Is calibrated so that a "normal" answer (correct, ratio ≈ 0.45) still
    gives roughly +0.43, matching the feel of the original +0.45 bucket

Usage (in run_real_student_session.py)
---------------------------------------
    from session_ability import update_session_ability

    # 1. Save ability BEFORE the env update
    prev_ability = env.effective_ability

    # 2. Let env update counters (consecutive_correct/wrong) + its own ability
    info = env.apply_external_answer(topic_idx, q_global, is_correct, time_taken)
    time_ratio = info["time_ratio"]

    # 3. Override env.effective_ability with the richer value
    rich_after, delta = update_session_ability(
        env, is_correct, time_ratio, prev_ability
    )
    # env.effective_ability is now set to rich_after; info["effective_ability_after"]
    # still holds the simple-formula value (for comparison / logging).

Training files (student_simulator.py, mcq_env.py) are NOT modified.
"""
from __future__ import annotations

from question_bank import MIN_ABILITY, MAX_ABILITY, clip

# ── Constants ─────────────────────────────────────────────────────────────────
# Scale the (range-agnostic) delta shape to the current ability range so a strong
# session actually raises ability / difficulty on the wider 10..50 scale
# (LIMITATIONS_and_NEXT.md C1).  range / 20 == 1.0 on the old 10..30 scale.
STEP_SCALE = (MAX_ABILITY - MIN_ABILITY) / 20.0
MAX_STEP = 1.0 * STEP_SCALE          # hard cap per question (matches student_simulator)
# Soft-bound margin: damp steps toward a bound within this many units of it.
SOFT_MARGIN = 0.15 * (MAX_ABILITY - MIN_ABILITY)


def _soft_bounded_step(ability: float, delta: float) -> float:
    if delta > 0:
        room = (MAX_ABILITY - ability) / max(SOFT_MARGIN, 1e-6)
    elif delta < 0:
        room = (ability - MIN_ABILITY) / max(SOFT_MARGIN, 1e-6)
    else:
        return 0.0
    return delta * clip(room, 0.0, 1.0)

# Correct-answer: two-slope linear formula
#   Phase 1 (tr ≤ 1.5): gentle decline from +0.55 → +0.05
#     base = 0.55 - 0.333 × tr
#   Phase 2 (tr > 1.5): steep decline (answering slowly then guessing correctly)
#     base = 0.05 - 0.80 × (tr - 1.5), clamped at -MAX_STEP
_CORR_P1_INTERCEPT = 0.55    # delta at tr = 0
_CORR_P1_SLOPE     = 0.333   # decline per unit tr in phase 1
_CORR_P1_CUTOFF    = 1.5     # breakpoint between phases
_CORR_P2_SLOPE     = 0.80    # steep decline per unit tr in phase 2

# Consecutive-correct streak bonus
_CORR_STREAK_PER = 0.03   # per consecutive-correct answer (including current)
_CORR_STREAK_CAP = 0.15   # max bonus (+0.15 at streak ≥ 5)

# Wrong-answer: single-slope linear formula
#   base = -(0.35 + 0.30 × min(tr, 3.0))
#   Fast wrong (-0.35 at tr=0) → slow wrong (-1.25 at tr=3.0)
_WRONG_INTERCEPT  = 0.35   # base penalty at tr = 0 (positive; negated in delta)
_WRONG_SLOPE      = 0.30   # extra penalty per unit of time_ratio
_WRONG_TR_MAX     = 3.0

# Consecutive-wrong extra penalty (same weight as base formula)
_WRONG_STREAK_PER = 0.10  # per consecutive-wrong extra penalty
_WRONG_STREAK_MAX = 5     # cap streak count for penalty


# ── Core delta function ────────────────────────────────────────────────────────
def rich_ability_delta(
    is_correct: bool,
    time_ratio: float,
    consecutive_correct: int = 0,
    consecutive_wrong: int = 0,
) -> float:
    """
    Return the ability change for a single answered question.

    Parameters
    ----------
    is_correct        : Whether the student answered correctly.
    time_ratio        : time_taken / base_time (>1 means slower than baseline).
    consecutive_correct : Number of consecutive correct answers INCLUDING this
                          one (as updated by env._update_counters before this
                          call).  0 if not correct.
    consecutive_wrong   : Likewise for wrong answers.

    Returns a float in [-MAX_STEP, +MAX_STEP].

    Formula summary
    ---------------
    Correct (two-slope):
        if tr ≤ 1.5: base = 0.55 - 0.333 × tr
                       → +0.55 (tr=0), +0.43 (tr=0.35), +0.37 (tr=0.53), +0.05 (tr=1.5)
        if tr > 1.5: base = 0.05 - 0.80 × (tr - 1.5)
                       → -0.19 (tr=1.8), -0.35 (tr=2.0), -0.51 (tr=2.2)
        streak = min(consecutive_correct × 0.03, 0.15)
        delta  = clip(base + streak, -MAX_STEP, +MAX_STEP)

    Wrong (single-slope):
        base   = -(0.35 + 0.30 × min(time_ratio, 3.0))
                  → -0.35 (instant), -0.45 (tr=0.34), -0.56 (tr=0.70),
                    -0.65 (tr=1.0), -0.80 (tr=1.5), -1.01 (tr=2.2)
        streak = 0.10 × min(consecutive_wrong, 5)
        delta  = clip(base - streak, -MAX_STEP, +MAX_STEP)
    """
    tr = max(0.0, float(time_ratio))

    if is_correct:
        if tr <= _CORR_P1_CUTOFF:
            base = _CORR_P1_INTERCEPT - _CORR_P1_SLOPE * tr
        else:
            base_at_cutoff = _CORR_P1_INTERCEPT - _CORR_P1_SLOPE * _CORR_P1_CUTOFF
            base = base_at_cutoff - _CORR_P2_SLOPE * (tr - _CORR_P1_CUTOFF)
        streak = min(int(consecutive_correct) * _CORR_STREAK_PER, _CORR_STREAK_CAP)
        d = base + streak
    else:
        base   = -(_WRONG_INTERCEPT + _WRONG_SLOPE * min(tr, _WRONG_TR_MAX))
        streak = _WRONG_STREAK_PER * min(int(consecutive_wrong), _WRONG_STREAK_MAX)
        d = base - streak

    return float(clip(d * STEP_SCALE, -MAX_STEP, MAX_STEP))


# ── High-level update (overrides env.effective_ability) ───────────────────────
def update_session_ability(
    env,
    is_correct: bool,
    time_ratio: float,
    prev_ability: float,
    q_difficulty: float | None = None,
) -> tuple[float, float]:
    """
    Compute the richer ability delta and write it back to ``env.effective_ability``.

    Must be called **after** ``env.apply_external_answer()`` so that
    ``env.consecutive_correct`` and ``env.consecutive_wrong`` are already
    updated by ``_update_counters``.

    Parameters
    ----------
    env          : MCQEnv instance.
    is_correct   : Answer correctness (same value passed to apply_external_answer).
    time_ratio   : time_taken / base_time (from apply_external_answer return dict).
    prev_ability : ``env.effective_ability`` *before* apply_external_answer was called.

    Returns
    -------
    (new_ability, delta) — new_ability is also written to env.effective_ability.
    """
    if q_difficulty is None:
        # backward-compatible fallback: assume an at-level question
        from question_bank import ability_to_bank_difficulty
        q_difficulty = ability_to_bank_difficulty(prev_ability)
    # Use the same difficulty-aware (Elo-style) update as training so the real
    # session and the env stay consistent (LIMITATIONS_and_NEXT.md C1).
    import student_simulator as _sim
    delta = _sim.ability_delta(
        prev_ability, is_correct, time_ratio, float(q_difficulty),
        consecutive_wrong=int(env.consecutive_wrong),
    )
    delta = _soft_bounded_step(prev_ability, delta)
    new_ability = float(clip(prev_ability + delta, MIN_ABILITY, MAX_ABILITY))
    env.effective_ability = new_ability
    return new_ability, delta
