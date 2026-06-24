"""
Dynamic student simulator for the June-15 adaptive-mechatronics RL system.

What this models (and why it justifies RL over a rule):

  * effective_ability (10..30, continuous, HIDDEN from the agent): the student's
    overall skill.  It drifts up/down with performance and is the quantity the
    *rule* uses to pick which difficulty of question to ask inside a topic.

  * topic_mastery[topic] in [0, 1] (HIDDEN): per-topic knowledge.  Together with
    effective_ability it drives answer probability and response time.

  * PREREQUISITE GATING: studying a topic whose prerequisites are not yet
    mastered yields little learning gain.  Sequencing therefore matters.

  * TRANSFER / INTERFERENCE: studying topic A spills mastery onto related topics
    (positive) or slightly erodes distant ones (negative), per
    ``curriculum.TRANSFER_MATRIX``.  Gains compound across a curriculum.

  * HIDDEN LEARNING STYLE in {massed, interleaved, blocked}: changes how mastery
    consolidates given the *sequence* of topics studied.  The agent never sees
    the style; it must infer it from early behaviour and adapt - exactly the
    kind of latent-state credit assignment RL does and a fixed rule cannot.

The agent observation (built in ``mcq_env.py``) is restricted to *behavioural*
signals (correctness, time ratios, streaks, observed per-topic accuracy, asked
counts, prerequisite structure).  None of effective_ability / topic_mastery is
ever exposed.
"""
from __future__ import annotations

import math
from collections import deque
from copy import deepcopy
from typing import Any

import numpy as np

import curriculum
from question_bank import (
    BANK_MAX_DIFF,
    BANK_MIN_DIFF,
    DIFFICULTY_HIGH,
    DIFFICULTY_LOW,
    MAX_ABILITY,
    MIN_ABILITY,
    OPTIONS,
    ability_to_bank_difficulty,
    ability_to_difficulty_scale,
    clip,
    make_default_distractor_strength,
)

LEARNING_STYLES = ["massed", "interleaved", "blocked"]

# Per-question ability step, scaled to the ability range.  The delta SHAPES below
# were tuned for a 20-wide range (10..30); when the range widened to 10..50 they
# became too small (a near-perfect session barely moved ability, freezing the
# difficulty target — see LIMITATIONS_and_NEXT.md L2).  We therefore scale every
# delta by ABILITY_STEP_SCALE = range / 20, preserving the original feel while
# restoring responsiveness on the wider scale.
ABILITY_STEP_SCALE = (MAX_ABILITY - MIN_ABILITY) / 20.0
# Per-question safety cap, sized to the range so the difficulty-aware update can
# converge within a session (a big "surprise" may move several ability units).
MAX_ABILITY_STEP = 0.12 * (MAX_ABILITY - MIN_ABILITY)
# Soft-bound margin: within this many ability units of a bound, the step toward
# that bound is damped (asymptotic approach) so learners don't slam into / stick
# at the floor or ceiling (L3).
ABILITY_SOFT_MARGIN = 0.15 * (MAX_ABILITY - MIN_ABILITY)


def _soft_bounded_step(ability: float, delta: float) -> float:
    """Damp ``delta`` as ``ability`` nears the bound it is moving toward."""
    if delta > 0:
        room = (MAX_ABILITY - ability) / max(ABILITY_SOFT_MARGIN, 1e-6)
    elif delta < 0:
        room = (ability - MIN_ABILITY) / max(ABILITY_SOFT_MARGIN, 1e-6)
    else:
        return 0.0
    return delta * clip(room, 0.0, 1.0)


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-clip(x, -30.0, 30.0)))


def ability_to_10_scale(ability: int | float) -> float:
    ability = clip(float(ability), MIN_ABILITY, MAX_ABILITY)
    return ((ability - MIN_ABILITY) / (MAX_ABILITY - MIN_ABILITY)) * 10.0


def mastery_to_bias(mastery: float) -> float:
    """Map topic mastery in [0,1] to an additive difficulty-scale bonus (~[-3,3])."""
    return (clip(mastery, 0.0, 1.0) - 0.5) * 6.0


# ---------------------------------------------------------------------------
# student creation
# ---------------------------------------------------------------------------
def _initial_topic_mastery(ability: int, rng: np.random.Generator) -> dict[str, float]:
    """Higher ability => higher baseline mastery, with per-topic noise.

    Advanced topics (deeper in the DAG) start a little lower than roots.
    """
    ability_norm = (clip(ability, MIN_ABILITY, MAX_ABILITY) - MIN_ABILITY) / (MAX_ABILITY - MIN_ABILITY)
    mastery: dict[str, float] = {}
    for topic in curriculum.TOPICS:
        depth = len(curriculum.all_ancestors(topic))
        depth_penalty = min(0.18, 0.012 * depth)
        base = 0.10 + 0.55 * ability_norm - depth_penalty
        value = base + float(rng.normal(0.0, 0.07))
        mastery[topic] = round(clip(value, 0.03, 0.93), 4)
    return mastery


def create_student(
    student_id: str,
    ability: int,
    *,
    rng: np.random.Generator | None = None,
    learning_style: str | None = None,
    carelessness: float | None = None,
    time_multiplier: float | None = None,
    time_alpha: float | None = None,
    time_beta: float | None = None,
    confidence: float | None = None,
    fatigue: float | None = None,
    guessing_tendency: float | None = None,
    speed_tendency: float | None = None,
    learning_rate: float | None = None,
    fatigue_rate: float | None = None,
    recovery_rate: float | None = None,
    forgetting_rate: float | None = None,
    time_noise_sigma: float | None = None,
    topic_mastery: dict[str, float] | None = None,
) -> dict[str, Any]:
    if ability < MIN_ABILITY or ability > MAX_ABILITY:
        raise ValueError(f"ability must be in [{MIN_ABILITY}, {MAX_ABILITY}]")
    if rng is None:
        rng = np.random.default_rng()

    ability_10 = ability_to_10_scale(ability)

    if learning_style is None:
        learning_style = str(rng.choice(LEARNING_STYLES))
    if carelessness is None:
        carelessness = clip(0.16 - ability_10 * 0.011, 0.025, 0.16)
    if time_multiplier is None:
        time_multiplier = clip(1.35 - ability_10 * 0.055, 0.65, 1.35)
    if time_alpha is None:
        time_alpha = clip(2.2 + ability_10 * 0.06, 2.0, 3.0)
    if time_beta is None:
        time_beta = clip(3.2 + ability_10 * 0.10, 3.0, 4.5)
    if confidence is None:
        confidence = clip(0.25 + ability_10 * 0.045, 0.15, 0.85)
    if fatigue is None:
        fatigue = 0.05
    if guessing_tendency is None:
        guessing_tendency = clip(0.30 - ability_10 * 0.020, 0.04, 0.30)
    if speed_tendency is None:
        speed_tendency = 1.0
    if learning_rate is None:
        # raised so a FOCUSED, style-matched learner can drive a topic to mastery
        # in a few visits (and thus master many topics in an episode), while a
        # blind spreader masters few - widening the smart-vs-dumb gap that RL can
        # exploit.  Forgetting still erodes anything left un-revisited.
        learning_rate = clip(0.18 + ability_10 * 0.008, 0.15, 0.32)
    if fatigue_rate is None:
        fatigue_rate = clip(0.16 - ability_10 * 0.006, 0.07, 0.16)
    if recovery_rate is None:
        recovery_rate = clip(0.07 + ability_10 * 0.006, 0.07, 0.14)
    if forgetting_rate is None:
        # per-step memory decay for UN-practiced topics (hidden).  Higher-ability
        # students forget more slowly.  This is the spaced-repetition pressure: a
        # topic left untouched erodes toward a low floor, so the optimal policy
        # must time revisits - something a fixed rule cannot tune per student.
        forgetting_rate = clip(0.012 - ability_10 * 0.0005, 0.005, 0.012)
    if time_noise_sigma is None:
        time_noise_sigma = 0.18
    if topic_mastery is None:
        topic_mastery = _initial_topic_mastery(ability, rng)

    return {
        "student_id": student_id,
        "ability": int(ability),
        "ability_10": round(ability_10, 2),
        "learning_style": str(learning_style),
        "carelessness": round(float(carelessness), 3),
        "time_multiplier": round(float(time_multiplier), 3),
        "time_alpha": round(float(time_alpha), 3),
        "time_beta": round(float(time_beta), 3),
        "confidence": round(float(confidence), 3),
        "fatigue": round(float(fatigue), 3),
        "guessing_tendency": round(float(guessing_tendency), 3),
        "speed_tendency": round(float(speed_tendency), 3),
        "learning_rate": round(float(learning_rate), 3),
        "fatigue_rate": round(float(fatigue_rate), 3),
        "recovery_rate": round(float(recovery_rate), 3),
        "forgetting_rate": round(float(forgetting_rate), 4),
        "time_noise_sigma": round(float(time_noise_sigma), 3),
        "topic_mastery": {k: round(float(v), 4) for k, v in topic_mastery.items()},
        # rolling record of studied topics (most recent last); hidden internal state
        "study_history": [],
    }


def create_student_population(
    variants_per_ability: int = 4,
    seed: int | None = None,
    student_id_prefix: str = "DS",
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    population: list[dict[str, Any]] = []
    styles = LEARNING_STYLES
    for ability in range(MIN_ABILITY, MAX_ABILITY + 1):
        for variant in range(1, max(1, variants_per_ability) + 1):
            # spread styles evenly so each style is well represented
            style = styles[(ability + variant) % len(styles)]
            population.append(
                create_student(
                    student_id=f"{student_id_prefix}{ability:02d}_v{variant:02d}",
                    ability=ability,
                    rng=rng,
                    learning_style=style,
                )
            )
    return population


def ensure_student(student: dict[str, Any]) -> dict[str, Any]:
    """Return a deep copy guaranteed to carry all dynamic fields."""
    s = deepcopy(student)
    s.setdefault("study_history", [])
    s.setdefault("learning_style", "interleaved")
    if "forgetting_rate" not in s:
        ability_10 = ability_to_10_scale(float(s.get("ability", MIN_ABILITY)))
        s["forgetting_rate"] = round(float(clip(0.012 - ability_10 * 0.0005, 0.005, 0.012)), 4)
    if "topic_mastery" not in s or not s["topic_mastery"]:
        s["topic_mastery"] = _initial_topic_mastery(int(s.get("ability", MIN_ABILITY)), np.random.default_rng(0))
    return s


def with_ability(student: dict[str, Any], ability: int | float) -> dict[str, Any]:
    s = ensure_student(student)
    # Keep ability CONTINUOUS (no integer rounding).  Effective ability evolves
    # in fractional steps during a session and we want the simulator to react to
    # those fractions instead of snapping 20.3 / 20.7 onto the same integer.
    a = float(clip(float(ability), MIN_ABILITY, MAX_ABILITY))
    s["ability"] = a
    s["ability_10"] = round(ability_to_10_scale(a), 2)
    return s


# ---------------------------------------------------------------------------
# answer / time simulation
# ---------------------------------------------------------------------------
def sample_perceived_difficulty(student: dict[str, Any], question: dict[str, Any], rng: np.random.Generator) -> float:
    # Continuous ability: linearly interpolate the {center, spread} profile
    # between the two bracketing integer ability levels so fractional ability
    # produces a smoothly varying perceived difficulty.
    ability = clip(float(student["ability"]), MIN_ABILITY, MAX_ABILITY)
    profile = question["difficulty_profile"]
    lo = int(clip(np.floor(ability), MIN_ABILITY, MAX_ABILITY))
    hi = int(clip(np.ceil(ability), MIN_ABILITY, MAX_ABILITY))
    if hi == lo:
        center, spread = profile[lo]["center"], profile[lo]["spread"]
    else:
        frac = ability - lo
        center = profile[lo]["center"] * (1.0 - frac) + profile[hi]["center"] * frac
        spread = profile[lo]["spread"] * (1.0 - frac) + profile[hi]["spread"] * frac
    values = np.round(np.arange(DIFFICULTY_LOW, DIFFICULTY_HIGH + 0.1, 0.1), 2)
    weights = np.exp(-0.5 * ((values - center) / spread) ** 2)
    probs = weights / weights.sum()
    return float(rng.choice(values, p=probs))


def choose_acting_ability(real_ability: float, rng: np.random.Generator) -> tuple[float, str]:
    """Per-question hidden 'acting' ability so same-ability students vary.

    Continuous + range-agnostic: thresholds and boost/slump magnitudes are
    expressed as fractions of the ability range ``R = MAX_ABILITY - MIN_ABILITY``
    so this keeps working after the 10..30 -> 10..50 widening.  The dominant
    'normal' branch returns the exact continuous ability; excursions draw a
    continuous boost/slump.
    """
    real_ability = float(clip(real_ability, MIN_ABILITY, MAX_ABILITY))
    R = MAX_ABILITY - MIN_ABILITY
    t_low = MIN_ABILITY + 0.30 * R          # ~ old 16 on a 10..30 scale
    t_mid = MIN_ABILITY + 0.65 * R          # ~ old 23 on a 10..30 scale
    roll = float(rng.random())
    if real_ability <= t_low:
        if roll < 0.75:
            return real_ability, "normal"
        low = min(MAX_ABILITY, real_ability + 0.40 * R)
        return float(rng.uniform(low, MAX_ABILITY)), "boost"
    if real_ability <= t_mid:
        if roll < 0.75:
            return real_ability, "normal"
        if roll < 0.88:
            low = min(MAX_ABILITY, real_ability + 0.25 * R)
            return float(rng.uniform(low, MAX_ABILITY)), "boost"
        high = max(MIN_ABILITY, real_ability - 0.25 * R)
        return float(rng.uniform(MIN_ABILITY, high)), "slump"
    if roll < 0.75:
        return real_ability, "normal"
    high = max(MIN_ABILITY, real_ability - 0.45 * R)
    return float(rng.uniform(MIN_ABILITY, high)), "slump"


def get_option_distribution(student: dict[str, Any], question: dict[str, Any], perceived_difficulty: float) -> dict[str, float]:
    topic = str(question["topic"])
    ability_scale = ability_to_difficulty_scale(student["ability"])
    mastery = float(student.get("topic_mastery", {}).get(topic, 0.4))
    confidence = float(student.get("confidence", 0.5))
    fatigue = float(student.get("fatigue", 0.0))
    guessing = float(student.get("guessing_tendency", 0.1))

    effective_scale = (
        ability_scale
        + mastery_to_bias(mastery)
        + 0.75 * (confidence - 0.50)
        - 1.05 * fatigue
        - 0.45 * guessing
    )
    mastery_prob = sigmoid(0.58 * (effective_scale - perceived_difficulty))

    correct = question["answer"]
    min_correct = 1.0 / len(OPTIONS)
    carelessness = float(student.get("carelessness", 0.05))
    eff_careless = clip(carelessness + 0.05 * guessing + 0.06 * fatigue - 0.03 * confidence, 0.02, 0.30)
    max_correct = clip(0.92 - eff_careless, min_correct + 0.05, 0.92)
    p_correct = clip(min_correct + (max_correct - min_correct) * mastery_prob, min_correct, max_correct)

    distribution = {correct: p_correct}
    wrong_prob = 1.0 - p_correct
    distractors = question.get("distractor_strength") or make_default_distractor_strength(correct)
    distractors = {o: s for o, s in distractors.items() if o != correct}
    total = sum(distractors.values()) or 1.0
    for option, strength in distractors.items():
        distribution[option] = wrong_prob * (strength / total)
    for option in OPTIONS:
        distribution.setdefault(option, 0.0)
    norm = sum(distribution.values())
    return {o: p / norm for o, p in distribution.items()}


def sample_response_time(student: dict[str, Any], question: dict[str, Any], perceived_difficulty: float, rng: np.random.Generator) -> float:
    base_time = float(question["base_time"])
    topic = str(question["topic"])
    ability_scale = ability_to_difficulty_scale(student["ability"])
    mastery = float(student.get("topic_mastery", {}).get(topic, 0.4))
    confidence = float(student.get("confidence", 0.5))
    fatigue = float(student.get("fatigue", 0.0))
    guessing = float(student.get("guessing_tendency", 0.1))

    effective_scale = ability_scale + mastery_to_bias(mastery) + 0.50 * (confidence - 0.50) - 0.85 * fatigue
    time_multiplier = float(student.get("time_multiplier", 1.0)) * float(student.get("speed_tendency", 1.0))
    time_multiplier *= 1.0 + 0.25 * fatigue - 0.10 * confidence + 0.12 * guessing
    time_multiplier = clip(time_multiplier, 0.45, 1.90)

    difficulty_gap = perceived_difficulty - effective_scale
    high_excess = max(0.0, float(question["inherent_difficulty"]) - 7.0)
    struggle_factor = 1.0 + max(0.0, difficulty_gap) * 0.125 + high_excess * 0.045
    ease_factor = 1.0 - min(max(0.0, effective_scale - perceived_difficulty) * 0.014, 0.20)

    min_time = max(3.0, base_time * time_multiplier * 0.38 * ease_factor)
    max_time = max(min_time + 2.0, base_time * time_multiplier * (2.10 + high_excess * 0.12) * struggle_factor)

    x = rng.beta(float(student.get("time_alpha", 2.5)), float(student.get("time_beta", 4.0)))
    time_taken = min_time + x * (max_time - min_time)
    noise_sigma = clip(float(student.get("time_noise_sigma", 0.18)) + high_excess * 0.012, 0.10, 0.34)
    time_taken *= float(rng.lognormal(mean=0.0, sigma=noise_sigma))
    return round(float(clip(time_taken, 3.0, max_time * 1.45)), 2)


def simulate_answer(student: dict[str, Any], question: dict[str, Any], rng: np.random.Generator | None = None) -> dict[str, Any]:
    if rng is None:
        rng = np.random.default_rng()
    student = ensure_student(student)
    real_ability = float(student["ability"])
    acting_ability, acting_mode = choose_acting_ability(real_ability, rng)
    acting_student = with_ability(student, acting_ability)
    acting_student["topic_mastery"] = student["topic_mastery"]

    perceived = sample_perceived_difficulty(acting_student, question, rng)
    dist = get_option_distribution(acting_student, question, perceived)
    probs = [dist[o] for o in OPTIONS]
    chosen = str(rng.choice(OPTIONS, p=probs))
    time_taken = sample_response_time(acting_student, question, perceived, rng)
    return {
        "student_id": student["student_id"],
        "acting_mode": acting_mode,
        "acting_ability": acting_ability,
        "question_id": question["question_id"],
        "topic": question["topic"],
        "inherent_difficulty": question["inherent_difficulty"],
        "base_time": question["base_time"],
        "sampled_perceived_difficulty": perceived,
        "chosen_option": chosen,
        "correct_answer": question["answer"],
        "is_correct": chosen == question["answer"],
        "time_taken": time_taken,
        "option_distribution": dist,
    }


# ---------------------------------------------------------------------------
# effective-ability update — difficulty-aware (Elo-style), shared by the
# training env AND the real session.  Ability converges to the difficulty level
# the learner can actually answer: correctly handling a question ABOVE your
# level raises ability a lot; failing one BELOW your level lowers it a lot;
# at-level answers nudge gently.  Modulated mildly by response time.
# (LIMITATIONS_and_NEXT.md C1)
# ---------------------------------------------------------------------------
ABILITY_PER_DIFF = (MAX_ABILITY - MIN_ABILITY) / max(BANK_MAX_DIFF - BANK_MIN_DIFF, 1e-6)
ELO_K_DIFF = 0.6     # base move per question, in difficulty units
ELO_SPREAD = 1.6     # logistic spread over difficulty units


def ability_delta(ability: float, is_correct: bool, time_ratio: float,
                  q_difficulty: float, *, consecutive_wrong: int = 0) -> float:
    """Difficulty-aware ability change for one answered question."""
    d_eq = ability_to_bank_difficulty(ability)
    p_correct = 1.0 / (1.0 + float(np.exp(-(d_eq - float(q_difficulty)) / ELO_SPREAD)))
    outcome = 1.0 if is_correct else 0.0
    delta_d = ELO_K_DIFF * (outcome - p_correct)          # surprise, difficulty units
    tr = max(0.0, float(time_ratio))
    tf = clip(1.15 - 0.12 * min(tr, 3.0), 0.6, 1.15)      # mild time modulation
    delta_d *= tf
    if not is_correct:
        delta_d -= 0.04 * min(int(consecutive_wrong), 5)  # discourage flailing
    delta = delta_d * ABILITY_PER_DIFF
    return clip(delta, -MAX_ABILITY_STEP, MAX_ABILITY_STEP)


def effective_ability_delta(is_correct: bool, time_ratio: float, *, consecutive_wrong: int = 0) -> float:
    """Deprecated time-only delta (kept for backward compatibility).  Assumes an
    at-level question; prefer ``ability_delta`` which is difficulty-aware."""
    mid_ability = (MIN_ABILITY + MAX_ABILITY) / 2.0
    return ability_delta(mid_ability, is_correct, time_ratio,
                         ability_to_bank_difficulty(mid_ability),
                         consecutive_wrong=consecutive_wrong)


def update_effective_ability(effective_ability: float, is_correct: bool, time_ratio: float,
                             q_difficulty: float, *, consecutive_wrong: int = 0) -> float:
    delta = ability_delta(effective_ability, is_correct, time_ratio, q_difficulty,
                          consecutive_wrong=consecutive_wrong)
    delta = _soft_bounded_step(effective_ability, delta)
    return clip(effective_ability + delta, MIN_ABILITY, MAX_ABILITY)


# ---------------------------------------------------------------------------
# prerequisite readiness & learning-style consolidation
# ---------------------------------------------------------------------------
def prerequisite_readiness(student: dict[str, Any], topic: str) -> float:
    """Mean mastery of the topic's direct prerequisites (1.0 if none)."""
    prereqs = curriculum.prerequisites_of(topic)
    if not prereqs:
        return 1.0
    mastery = student.get("topic_mastery", {})
    return float(np.mean([float(mastery.get(p, 0.0)) for p in prereqs]))


def _style_multiplier(student: dict[str, Any], topic: str) -> float:
    """How well the *sequence* matches the hidden learning style (~0.45..1.6).

    Widened from the old ~0.75..1.35 so that MATCHING the hidden style is worth a
    lot: an agent that infers a student is 'massed' and drills, vs one that
    interleaves, sees a large difference in learning.  This makes inferring the
    latent style the single biggest lever - and a static rule cannot infer it.
    """
    style = student.get("learning_style", "interleaved")
    history = list(student.get("study_history", []))
    last = history[-1] if history else None
    last3 = history[-3:]

    if style == "massed":
        # likes staying on one topic until mastered
        if last == topic:
            return 1.80
        return 0.20
    if style == "interleaved":
        # likes alternating; immediate repeats barely teach anything
        if last == topic:
            return 0.20
        if topic not in last3:
            return 1.80
        return 0.70
    # blocked: likes working within a cluster (chapter) before switching
    cluster = curriculum.TOPIC_TO_CLUSTER.get(topic)
    last_cluster = curriculum.TOPIC_TO_CLUSTER.get(last) if last else None
    if last_cluster is None:
        return 1.0
    if cluster == last_cluster:
        return 1.75
    return 0.25


FORGETTING_FLOOR = 0.05


def set_mastery_baseline(student: dict[str, Any]) -> dict[str, Any]:
    """Snapshot the current per-topic mastery as the 'stable' baseline.

    Forgetting erodes recently-acquired mastery back toward this baseline (the
    student's prior knowledge), NOT toward zero - you forget what you just
    crammed, not what you already knew.  Call once at the start of an episode.
    """
    student["mastery_baseline"] = {k: float(v) for k, v in student.get("topic_mastery", {}).items()}
    return student


def apply_forgetting(student: dict[str, Any], practiced_topic: str | None = None,
                     *, steps: int = 1) -> dict[str, Any]:
    """Decay UN-practiced topics one (or ``steps``) step toward their baseline.

    Called once per environment step.  Only mastery *above* the baseline (recent
    gains) decays, at the student's hidden ``forgetting_rate``; the topic just
    practiced is exempt.  This creates the spaced-repetition problem - gains
    evaporate unless revisited - while leaving prior knowledge intact, so a good
    policy can still achieve net-positive retained learning.  Mutates in place.
    """
    rate = float(student.get("forgetting_rate", 0.0))
    if rate <= 0.0 or steps <= 0:
        return student
    retain = (1.0 - rate) ** int(steps)
    mastery = student.get("topic_mastery", {})
    baseline = student.get("mastery_baseline", {})
    for t, m in mastery.items():
        if t == practiced_topic:
            continue
        m = float(m)
        base = float(baseline.get(t, FORGETTING_FLOOR))
        # consolidate toward the baseline from BOTH sides: un-rehearsed *gains*
        # fade (spacing pressure), and *dips* from recent wrong answers recover.
        mastery[t] = round(base + (m - base) * retain, 4)
    student["topic_mastery"] = mastery
    return student


def apply_learning_update(
    student: dict[str, Any],
    question: dict[str, Any],
    is_correct: bool,
    time_ratio: float,
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    """Update hidden mastery / affect after one answered question.

    Returns a report dict including the learning gains (used by the env reward).
    Mutates a copy and returns the updated student under key ``student``.
    """
    if rng is None:
        rng = np.random.default_rng()
    s = ensure_student(student)
    topic = str(question["topic"])
    mastery = dict(s["topic_mastery"])

    learning_rate = float(s.get("learning_rate", 0.12))
    fatigue_rate = float(s.get("fatigue_rate", 0.10))
    recovery_rate = float(s.get("recovery_rate", 0.10))
    confidence = float(s.get("confidence", 0.5))
    fatigue = float(s.get("fatigue", 0.0))
    guessing = float(s.get("guessing_tendency", 0.1))

    # base learning increment from performance.  Learning (correct) is the main
    # driver; forgetting (wrong) is real but milder - you do not unlearn as fast
    # as you learn - so a well-sequenced curriculum yields NET POSITIVE mastery
    # and the agent has a gain to maximise.
    if is_correct:
        if time_ratio <= 0.80:
            base_delta = learning_rate * 1.10
            confidence += recovery_rate * 0.95
            fatigue -= recovery_rate * 0.75
            guessing -= 0.020
        elif time_ratio <= 1.50:
            base_delta = learning_rate * 0.80
            confidence += recovery_rate * 0.55
            fatigue -= recovery_rate * 0.35
            guessing -= 0.012
        elif time_ratio <= 2.20:
            base_delta = learning_rate * 0.45
            confidence += recovery_rate * 0.10
            fatigue += 0.025
        else:
            base_delta = learning_rate * 0.20
            confidence -= 0.035
            fatigue += 0.060
    else:
        if time_ratio < 0.70:
            base_delta = -learning_rate * 0.12
            confidence -= 0.060
            fatigue += 0.045
            guessing += 0.040
        elif time_ratio <= 1.50:
            base_delta = -learning_rate * 0.22
            confidence -= 0.090
            fatigue += 0.075
            guessing += 0.020
        else:
            base_delta = -learning_rate * 0.32
            confidence -= 0.120
            fatigue += fatigue_rate
    fatigue += 0.008

    # prerequisite gating: you cannot learn a topic well without its prereqs.
    # Quadratic throttle makes ORDER consequential - teaching a topic before its
    # prerequisites are solid yields almost nothing (readiness 0.5 -> 0.25x,
    # 0.3 -> ~0.09x), so the optimal policy must unlock topics in a good order.
    readiness = prerequisite_readiness(s, topic)
    prereq_throttle = max(0.05, readiness ** 2)  # in [0.05, 1.0]

    # learning style consolidation (depends on the *sequence* of topics)
    style_mult = _style_multiplier(s, topic)

    # diminishing returns near mastery ceiling
    current = float(mastery.get(topic, 0.3))

    if base_delta >= 0:
        # GAINS are shaped by prerequisites, learning style and remaining headroom:
        # this is what a smart curriculum policy exploits.
        headroom = 1.0 - current
        topic_delta = base_delta * prereq_throttle * style_mult * headroom
    else:
        # FORGETTING is only mildly worsened by attempting an unprepared topic.
        topic_delta = base_delta * (1.0 + 0.4 * (1.0 - readiness))
    new_topic_mastery = clip(current + topic_delta, 0.0, 1.0)
    topic_gain = new_topic_mastery - current
    mastery[topic] = round(new_topic_mastery, 4)

    # transfer / interference to related topics (scaled by how much was learned)
    transfer_gain = 0.0
    transfer_detail: dict[str, float] = {}
    if topic_gain > 0:
        for target, weight in curriculum.transfer_targets(topic).items():
            before = float(mastery.get(target, 0.0))
            if weight > 0:
                spill = weight * topic_gain
            else:
                # interference is a small fixed erosion regardless of gain sign
                spill = weight * 0.5 * abs(base_delta)
            after = clip(before + spill, 0.0, 1.0)
            mastery[target] = round(after, 4)
            delta = after - before
            transfer_gain += delta
            if abs(delta) > 1e-6:
                transfer_detail[target] = round(delta, 4)

    # commit affect + mastery + history
    s["confidence"] = round(clip(confidence, 0.05, 0.95), 4)
    s["fatigue"] = round(clip(fatigue, 0.0, 1.0), 4)
    s["guessing_tendency"] = round(clip(guessing, 0.02, 0.45), 4)
    s["topic_mastery"] = mastery
    history = list(s.get("study_history", []))
    history.append(topic)
    s["study_history"] = history[-12:]
    s["last_topic"] = topic
    s["last_topic_gain"] = round(topic_gain, 4)

    return {
        "student": s,
        "topic": topic,
        "topic_gain": round(topic_gain, 4),
        "transfer_gain": round(transfer_gain, 4),
        "transfer_detail": transfer_detail,
        "total_learning_gain": round(topic_gain + transfer_gain, 4),
        "prerequisite_readiness": round(readiness, 4),
        "prereq_throttle": round(prereq_throttle, 4),
        "style_multiplier": round(style_mult, 4),
        "base_delta": round(base_delta, 4),
    }


def mean_mastery(student: dict[str, Any]) -> float:
    values = list(student.get("topic_mastery", {}).values())
    return float(np.mean(values)) if values else 0.0


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    pop = create_student_population(variants_per_ability=2, seed=1)
    print("population size:", len(pop))
    from collections import Counter
    print("styles:", Counter(s["learning_style"] for s in pop))
    s = create_student("demo", ability=18, rng=rng, learning_style="massed")
    from question_bank import questions_for_topic
    q = questions_for_topic("Sensors and transducers")[1]
    print("mean mastery before:", round(mean_mastery(s), 4))
    res = simulate_answer(s, q, rng)
    rep = apply_learning_update(s, q, res["is_correct"], 0.7, rng)
    print("correct:", res["is_correct"], "topic_gain:", rep["topic_gain"], "transfer:", rep["transfer_detail"])
    print("mean mastery after:", round(mean_mastery(rep["student"]), 4))
    print("eff-ability after:", update_effective_ability(18.0, res["is_correct"], 0.7))
