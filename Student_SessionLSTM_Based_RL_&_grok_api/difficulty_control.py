"""
difficulty_control.py — delta-driven difficulty controller.

Instead of mapping effective ability (10..50) onto the difficulty scale to pick
the next question (which only serves the hardest items when ability ≈ 50), the
NEXT QUESTION DIFFICULTY is a controller state ``target_difficulty`` that is
nudged every answer by a continuous ``difficulty_delta``:

    target_difficulty <- clip(target_difficulty + difficulty_delta(...),
                              BANK_MIN_DIFF, BANK_MAX_DIFF)

so hard questions appear as soon as the learner *earns* them, not only at the
top of the ability scale.

The delta is a generalised linear equation over interpretable performance
features (no fixed range/bucket rules):

    delta =  A·sgn                               base step (correct +, wrong −)
           + B·correct·speed                     correct & fast → climb more
           − H·wrong·max(0, tr−1)                wrong & slow (too hard) → drop more
           + C·cc2                               two-in-a-row correct → push up
           + D·cc2·speed_avg(curr,prev)          fast correct streak → push up more
           − E·ww2                               two-in-a-row wrong → drop
           + F·cc3                               sustained mastery (≥3 correct)
           − G·ww3                               sustained struggle (≥3 wrong)

where  sgn = +1 correct / −1 wrong,
       speed = clip(1 − time_ratio, −1, 1)  (+ = faster than baseline),
       speed_avg uses the mean time_ratio of the current and previous answer,
       cc2/cc3 = 1 if the last 2/3 answers were all correct, ww2/ww3 likewise.

This dependency-light module imports only ``question_bank`` (no cycle with
``mcq_env`` / ``difficulty_policy``).
"""
from __future__ import annotations

from question_bank import BANK_MIN_DIFF, BANK_MAX_DIFF, ability_to_bank_difficulty, clip

# ── Coefficients (bank-difficulty units). Tunable. ────────────────────────────
A_BASE        = 0.45   # correct(+) / wrong(−) base step
B_SPEED_CORR  = 0.35   # correct & fast → climb more (slow correct → less)
H_SLOW_WRONG  = 0.30   # wrong & slow (too hard) → drop more
C_CC2         = 0.30   # two-in-a-row correct bonus
D_CC2_SPEED   = 0.30   # fast correct-streak bonus (× avg speed of last two)
E_WW2         = 0.45   # two-in-a-row wrong penalty
F_CC3         = 0.25   # sustained mastery (≥3 correct)
G_WW3         = 0.30   # sustained struggle (≥3 wrong)
DELTA_CAP     = 2.0    # max |delta| per question


def difficulty_delta(
    is_correct: bool,
    time_ratio: float,
    prev_time_ratio: float,
    consecutive_correct: int,
    consecutive_wrong: int,
) -> float:
    """Continuous change in target difficulty for one answered question."""
    o = 1.0 if is_correct else 0.0
    sgn = 2.0 * o - 1.0
    tr = clip(float(time_ratio), 0.0, 3.0)
    tr_prev = clip(float(prev_time_ratio), 0.0, 3.0)
    speed = clip(1.0 - tr, -1.0, 1.0)
    speed_avg = clip(1.0 - (tr + tr_prev) / 2.0, -1.0, 1.0)
    cc2 = 1.0 if consecutive_correct >= 2 else 0.0
    ww2 = 1.0 if consecutive_wrong >= 2 else 0.0
    cc3 = 1.0 if consecutive_correct >= 3 else 0.0
    ww3 = 1.0 if consecutive_wrong >= 3 else 0.0

    delta = (
        A_BASE * sgn
        + B_SPEED_CORR * o * speed
        - H_SLOW_WRONG * (1.0 - o) * max(0.0, tr - 1.0)
        + C_CC2 * cc2
        + D_CC2_SPEED * cc2 * speed_avg
        - E_WW2 * ww2
        + F_CC3 * cc3
        - G_WW3 * ww3
    )
    return float(clip(delta, -DELTA_CAP, DELTA_CAP))


def initial_target_difficulty(effective_ability: float) -> float:
    """One-time seed for a new session, mapped from the starting ability.
    After this the target is driven purely by ``difficulty_delta``."""
    return float(clip(ability_to_bank_difficulty(effective_ability),
                      BANK_MIN_DIFF, BANK_MAX_DIFF))


def step_target_difficulty(
    target_difficulty: float,
    is_correct: bool,
    time_ratio: float,
    prev_time_ratio: float,
    consecutive_correct: int,
    consecutive_wrong: int,
) -> float:
    """Apply one delta update and clip to the bank difficulty range."""
    d = difficulty_delta(is_correct, time_ratio, prev_time_ratio,
                         consecutive_correct, consecutive_wrong)
    return float(clip(target_difficulty + d, BANK_MIN_DIFF, BANK_MAX_DIFF))
