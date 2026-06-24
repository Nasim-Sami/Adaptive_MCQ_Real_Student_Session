"""
question_selector.py — Gap-based question selection for the real student session.

Replaces the play-count LLM schedule with a principled, need-driven approach:

    1. Identify unasked questions for the chosen topic.
       "Unasked" = not asked this episode AND not asked in any previous session
       (tracked via asked_ids_history passed in from student_history).
    2. Check if any unasked question falls within GAP_THRESHOLD (1.0) of the
       target difficulty (from compute_target_difficulty).
    3. YES → use the closest unasked bank question within the threshold.
    4. NO  → LLM generates a fresh question at the exact target difficulty.
             (Grounded in PDF chunks via retriever.)
    5. LLM fails / disabled → use closest available bank question as fallback.
    6. All questions asked before → allow repeats (logged as was_repeat=True).

The LLM is now used precisely when needed rather than on a fixed schedule,
making every generated question pedagogically motivated.

Public API
----------
    from question_selector import select_question_for_topic

    q_global, question, source, was_repeat, gap_ok = select_question_for_topic(
        topic=topic,
        topic_idx=topic_idx,
        target_diff=target_diff,
        asked_ids=asked_ids_session | asked_ids_history,
        env=env,
        use_llm=True,
        retrieve_fn=lambda t: retriever.retrieve_for_topic(t, k=6),
        generate_fn=question_generator.generate_mcq,
    )
"""
from __future__ import annotations

from typing import Any, Callable

import question_bank as qb

GAP_THRESHOLD = 1.0   # max |difficulty - target| to accept a bank question


def select_question_for_topic(
    topic: str,
    topic_idx: int,
    target_diff: float,
    asked_ids: set[str],                         # session + history combined
    env,                                          # MCQEnv — for qid_to_idx
    use_llm: bool,
    retrieve_fn: Callable[[str], list[dict]],    # retriever.retrieve_for_topic(topic)
    generate_fn: Callable[..., dict | None],     # question_generator.generate_mcq
) -> tuple[int, dict[str, Any], str, bool, bool]:
    """
    Select the best question for a topic given the target difficulty.

    Returns
    -------
    q_global   : Global question index in env.questions (bank question or proxy).
    question   : Full question dict (bank or LLM-generated).
    source     : "bank" or "llm".
    was_repeat : True if every bank question for this topic was previously asked.
    gap_ok     : True if the chosen bank question is within GAP_THRESHOLD.
                 Always True for LLM questions (they target exact difficulty).
    """
    topic_qs = qb.questions_for_topic(topic)
    if not topic_qs:
        raise ValueError(f"No questions found for topic {topic!r}")

    # Separate unasked vs already-seen questions
    unasked = [q for q in topic_qs if q["question_id"] not in asked_ids]
    was_repeat = len(unasked) == 0
    pool = unasked if unasked else topic_qs  # allow repeats only when exhausted

    # ── Step 1: look for a bank question within the gap threshold ────────────
    close = [q for q in pool if abs(float(q["inherent_difficulty"]) - target_diff) < GAP_THRESHOLD]

    if close:
        best = min(close, key=lambda q: abs(float(q["inherent_difficulty"]) - target_diff))
        q_global = env.qid_to_idx[best["question_id"]]
        question = dict(best)
        question.setdefault("source", "bank")
        return q_global, question, "bank", was_repeat, True

    # ── Step 2: no close bank question — try LLM ────────────────────────────
    if use_llm:
        try:
            chunks = retrieve_fn(topic)
            llm_q = generate_fn(topic, target_diff, chunks)
        except Exception:
            llm_q = None

        if llm_q is not None:
            # Use closest bank question as env bookkeeping proxy
            proxy = min(pool, key=lambda q: abs(float(q["inherent_difficulty"]) - target_diff))
            q_global = env.qid_to_idx[proxy["question_id"]]
            llm_q.setdefault("source", "llm")
            return q_global, llm_q, "llm", was_repeat, True   # LLM targets exact diff

    # ── Step 3: fallback — closest bank question even if gap > threshold ─────
    best = min(pool, key=lambda q: abs(float(q["inherent_difficulty"]) - target_diff))
    q_global = env.qid_to_idx[best["question_id"]]
    gap = abs(float(best["inherent_difficulty"]) - target_diff)
    question = dict(best)
    question.setdefault("source", "bank")
    return q_global, question, "bank", was_repeat, gap < GAP_THRESHOLD
