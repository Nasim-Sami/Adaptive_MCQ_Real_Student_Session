"""
explain_answer.py — Dynamic, grounded explanations after each MCQ answer.

Public API
----------
    from explain_answer import generate_explanation

    text = generate_explanation(
        question=question_dict,
        student_answer="B",
        is_correct=False,
        chunks=retrieve_for_topic(question["topic"], k=5),
    )

The function returns a freshly generated explanation grounded in the retrieved
PDF chunks.  Falls back to the bank's static explanation string if the LLM is
unavailable or produces an unusable response.
"""
from __future__ import annotations

import random
from typing import Any

import llm_client as llm


# ---------------------------------------------------------------------------
# Prompt engineering helpers
# ---------------------------------------------------------------------------
_PHRASING_VARIANTS = [
    "Provide a clear, concise explanation.",
    "Give a focused, student-friendly explanation.",
    "Write a direct and informative explanation.",
    "Offer a brief but insightful explanation.",
]

_SYSTEM = (
    "You are a mechatronics tutor. "
    "You ONLY use information from the provided textbook excerpts. "
    "Do NOT introduce facts not found in those excerpts. "
    "Respond in plain English suitable for an engineering student."
)


def _build_prompt(
    question: dict[str, Any],
    student_answer: str,
    is_correct: bool,
    chunks: list[dict[str, Any]],
) -> str:
    context_parts = []
    for i, c in enumerate(chunks[:5], start=1):
        # cap chunk length - some retrieved chunks run 2000+ chars; uncapped,
        # multiple large chunks risk a 413 Payload Too Large from the LLM API
        context_parts.append(
            f"[Excerpt {i} — {c['book'].capitalize()}, p.{c['page']}]\n{c['text'][:600]}"
        )
    context = "\n\n".join(context_parts) if context_parts else "(No textbook excerpts available.)"

    correct_opt = question["answer"]
    correct_text = question.get(f"option_{correct_opt}", "")
    student_text = question.get(f"option_{student_answer}", "")

    outcome = "correct" if is_correct else "incorrect"
    phrasing = random.choice(_PHRASING_VARIANTS)

    return (
        f"TEXTBOOK EXCERPTS:\n{context}\n\n"
        f"---\n"
        f"QUESTION (topic: {question['topic']}):\n{question['question']}\n\n"
        f"Options:\n"
        f"  A. {question.get('option_A', '')}\n"
        f"  B. {question.get('option_B', '')}\n"
        f"  C. {question.get('option_C', '')}\n"
        f"  D. {question.get('option_D', '')}\n\n"
        f"The student answered {student_answer} (\"{student_text}\") — {outcome}.\n"
        f"The correct answer is {correct_opt} (\"{correct_text}\").\n\n"
        f"Task: {phrasing} "
        "Explain WHY the correct answer is right and, if the student was wrong, "
        "briefly clarify the misconception. "
        "Ground your explanation ONLY in the textbook excerpts above. "
        "Keep it under 120 words."
    )


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------
def generate_explanation(
    question: dict[str, Any],
    student_answer: str,
    is_correct: bool,
    chunks: list[dict[str, Any]] | None = None,
) -> str:
    """
    Generate a grounded, student-specific explanation.

    Parameters
    ----------
    question      : Question dict from the bank (or LLM-generated).
    student_answer: The letter the student chose ('A'–'D').
    is_correct    : Whether the student was correct.
    chunks        : Retrieved PDF chunks for context.  If None or empty,
                    falls back immediately to the bank's static explanation.

    Returns
    -------
    Explanation string (LLM-generated or static fallback).
    """
    static_fallback: str = question.get("explanation", "No explanation available.")

    if not chunks:
        return static_fallback

    try:
        prompt = _build_prompt(question, student_answer, is_correct, chunks)
        text = llm.generate_text(
            prompt=prompt,
            system=_SYSTEM,
            temperature=0.65,
            max_tokens=200,
        )
        # Basic sanity: must be non-empty and different from a blank LLM failure
        if len(text.strip().split()) < 10:
            return static_fallback
        return text.strip()

    except (llm.LLMUnavailable, Exception):
        return static_fallback
