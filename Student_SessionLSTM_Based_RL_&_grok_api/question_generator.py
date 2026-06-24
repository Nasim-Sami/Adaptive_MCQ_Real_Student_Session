"""
question_generator.py — Generate grounded MCQs via the LLM (Groq, see llm_client.py).

The generated question is EPHEMERAL: used in the session and logged, but
NEVER written back into _qbank_part*.py.

Public API
----------
    from question_generator import generate_mcq

    q = generate_mcq(
        topic="PID control",
        target_difficulty=6.5,
        chunks=retrieve_for_topic("PID control", k=6),
    )
    # q is a dict with the same schema as bank questions, plus source="llm"
    # Returns None if generation fails validation → caller falls back to bank.
"""
from __future__ import annotations

import re
from typing import Any

import llm_client as llm
from difficulty_policy import difficulty_label, BANK_MIN_DIFF, BANK_MAX_DIFF


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
_SYSTEM = (
    "You are a mechatronics exam author. "
    "You ONLY use information from the provided textbook excerpts. "
    "Do NOT invent facts. "
    "Always respond with a single valid JSON object and nothing else."
)

_MCQ_SCHEMA = """\
{
  "question": "<the question text>",
  "option_A": "<option A text>",
  "option_B": "<option B text>",
  "option_C": "<option C text>",
  "option_D": "<option D text>",
  "answer": "<A, B, C, or D>",
  "explanation": "<brief explanation referencing the excerpts>"
}"""


def _build_prompt(
    topic: str,
    target_difficulty: float,
    chunks: list[dict[str, Any]],
) -> str:
    context_parts = []
    for i, c in enumerate(chunks[:6], start=1):
        # cap chunk length - some retrieved chunks run 2000+ chars; uncapped,
        # multiple large chunks risk a 413 Payload Too Large from the LLM API
        context_parts.append(
            f"[Excerpt {i} — {c['book'].capitalize()}, p.{c['page']}]\n{c['text'][:600]}"
        )
    context = "\n\n".join(context_parts) if context_parts else "(No excerpts.)"
    label = difficulty_label(target_difficulty)

    return (
        f"TEXTBOOK EXCERPTS:\n{context}\n\n"
        f"---\n"
        f"TASK: Write ONE multiple-choice question on the mechatronics topic "
        f"\"{topic}\" at {label} difficulty ({target_difficulty:.1f}/10).\n\n"
        "Requirements:\n"
        "1. Exactly 4 options (A, B, C, D) — all plausible, only one correct.\n"
        "2. The question and correct answer MUST be grounded in the excerpts above.\n"
        "3. Do NOT copy a sentence verbatim; paraphrase and test understanding.\n"
        "4. The distractors should reflect common misconceptions.\n"
        "5. Include a brief explanation citing the relevant excerpt.\n\n"
        f"Return ONLY the JSON object matching this schema:\n{_MCQ_SCHEMA}"
    )


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------
def _validate(raw: Any, topic: str, target_difficulty: float) -> dict[str, Any] | None:
    """
    Validate the raw dict returned by the LLM.
    Returns a cleaned question dict or None on failure.
    """
    if not isinstance(raw, dict):
        return None

    required = {"question", "option_A", "option_B", "option_C", "option_D", "answer", "explanation"}
    if not required.issubset(raw.keys()):
        return None

    answer = str(raw.get("answer", "")).strip().upper()
    if answer not in ("A", "B", "C", "D"):
        return None

    # All options must be non-empty and distinct
    options = [str(raw.get(f"option_{x}", "")).strip() for x in ("A", "B", "C", "D")]
    if any(len(o) < 5 for o in options):
        return None
    if len(set(options)) < 4:
        return None

    question_text = str(raw.get("question", "")).strip()
    if len(question_text.split()) < 6:
        return None

    # Assign a question_id (ephemeral, won't clash with bank T##Q# pattern)
    safe_topic = re.sub(r"[^A-Za-z0-9]", "_", topic)[:20]
    qid = f"LLM_{safe_topic}_{int(target_difficulty * 10):03d}"

    question = {
        "question_id": qid,
        "topic": topic,
        "inherent_difficulty": round(float(target_difficulty), 2),
        "question": question_text,
        "option_A": options[0],
        "option_B": options[1],
        "option_C": options[2],
        "option_D": options[3],
        "answer": answer,
        "distractor_strength": {},            # not available for LLM questions
        "subtopic": "llm-generated",
        "explanation": str(raw.get("explanation", "")).strip(),
        "source": "llm",                      # tag — never in bank files
    }

    # The student simulator (sample_perceived_difficulty / estimate_base_time
    # callers) requires the same per-ability difficulty_profile / base_time
    # metadata that question_bank.attach_metadata() adds to bank questions.
    # LLM-generated questions skip that pipeline, so attach it here too
    # (local import — question_bank intentionally has no dep on this module).
    import question_bank as qb
    question["base_time"] = qb.estimate_base_time(question, float(target_difficulty))
    question["difficulty_profile"] = qb.build_difficulty_profile(float(target_difficulty))
    return question


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------
def generate_mcq(
    topic: str,
    target_difficulty: float,
    chunks: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """
    Generate and validate one MCQ grounded in the provided PDF chunks.

    Returns
    -------
    A question dict (same schema as bank, plus source="llm") on success,
    or None on failure (caller should fall back to bank).
    """
    if not chunks:
        return None

    target_difficulty = max(BANK_MIN_DIFF, min(BANK_MAX_DIFF, float(target_difficulty)))

    try:
        prompt = _build_prompt(topic, target_difficulty, chunks)
        raw = llm.generate_json(
            prompt=prompt,
            system=_SYSTEM,
            temperature=0.75,
            max_tokens=600,
        )
        return _validate(raw, topic, target_difficulty)

    except (llm.LLMUnavailable, ValueError, Exception):
        return None
