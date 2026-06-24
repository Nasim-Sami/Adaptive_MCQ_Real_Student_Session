"""
learning_path.py — Personalized learning-path narrative at episode end.

Keeps the prerequisite-correct topological backbone from the base system's
``suggest_learning_path()``, then has the LLM (Groq, see llm_client.py) turn it
into a personalized, grounded narrative that references:
  * the student's specific mistakes and weak topics (from the topic report)
  * relevant PDF chapter/section pointers from the retriever

Output is written to ``learning_path.md`` inside the session folder.

Public API
----------
    from learning_path import generate_learning_path

    path_text = generate_learning_path(
        ordered_topics=env.suggest_learning_path(),   # list[str]
        topic_report=env.get_topic_report(),          # dict
        student_id="s42",
        session_dir=Path("real_student_sessions/20240619_120000_dqn"),
    )
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import llm_client as llm
import retriever


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
_SYSTEM = (
    "You are a mechatronics study advisor. "
    "You ONLY reference information from the provided textbook excerpts. "
    "Write concise, actionable advice in plain English. "
    "Do not introduce topics or facts absent from the excerpts."
)

CHUNKS_PER_TOPIC = 1          # how many PDF chunks to pull per priority topic
MAX_PRIORITY_TOPICS = 3       # limit to avoid oversized prompts
MAX_CHUNK_CHARS = 600         # truncate each chunk - some retrieved chunks run
                               # 2000+ chars, and with 5 topics x 2 chunks the
                               # untruncated prompt hit ~24KB / ~6000 tokens,
                               # which Groq rejected with 413 Payload Too Large


def _collect_context(priority_topics: list[str]) -> str:
    """Retrieve PDF chunks for the top priority topics and format as context."""
    parts: list[str] = []
    for topic in priority_topics[:MAX_PRIORITY_TOPICS]:
        chunks = retriever.retrieve_for_topic(topic, k=CHUNKS_PER_TOPIC)
        for c in chunks:
            text = c["text"][:MAX_CHUNK_CHARS]
            parts.append(
                f"[{c['book'].capitalize()}, p.{c['page']} — {topic}]\n{text}"
            )
    return "\n\n".join(parts) if parts else "(No textbook excerpts available.)"


def _build_prompt(
    ordered_topics: list[str],
    topic_report: dict[str, Any],
    student_id: str,
    context: str,
) -> str:
    # Summarize weak topics from the report
    weak_lines: list[str] = []
    for topic, data in topic_report.items():
        acc = data.get("accuracy", None)
        asked = data.get("attempts", 0)  # mcq_env.get_topic_report() key is "attempts"
        if asked > 0 and acc is not None and acc < 0.6:
            weak_lines.append(f"  - {topic}: {acc*100:.0f}% accuracy ({asked} questions)")

    weak_summary = "\n".join(weak_lines) if weak_lines else "  (No clearly weak topics identified.)"

    path_str = "\n".join(f"  {i+1}. {t}" for i, t in enumerate(ordered_topics[:10]))

    return (
        f"TEXTBOOK EXCERPTS:\n{context}\n\n"
        f"---\n"
        f"STUDENT: {student_id}\n\n"
        f"WEAK TOPICS (accuracy < 60%):\n{weak_summary}\n\n"
        f"SUGGESTED STUDY ORDER (prerequisite-respecting):\n{path_str}\n\n"
        "TASK: Write a personalized learning-path recommendation (200–300 words) for "
        "this student. For each of the top 3–5 priority topics:\n"
        "  1. Briefly explain why it matters and what the student should focus on.\n"
        "  2. Reference the specific textbook and page number from the excerpts above.\n"
        "  3. Suggest a concrete study action (re-read a section, practice a type of problem).\n\n"
        "Use the excerpts ONLY. Do not invent textbook references. "
        "Write in second person ('You should ...'). "
        "Start with a one-sentence summary of the student's overall performance."
    )


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------
def generate_learning_path(
    ordered_topics: list[str],
    topic_report: dict[str, Any],
    student_id: str,
    session_dir: Path | None = None,
) -> str:
    """
    Generate a grounded, personalized learning-path narrative.

    Parameters
    ----------
    ordered_topics : Prerequisite-sorted list from env.suggest_learning_path().
    topic_report   : Dict from env.get_topic_report().
    student_id     : Student identifier string.
    session_dir    : If provided, writes learning_path.md there.

    Returns
    -------
    The narrative string (LLM-generated or plain-list fallback).
    """
    # Plain fallback
    fallback = "Suggested study order:\n" + "\n".join(
        f"  {i+1}. {t}" for i, t in enumerate(ordered_topics[:10])
    )

    try:
        context = _collect_context(ordered_topics)
        prompt = _build_prompt(ordered_topics, topic_report, student_id, context)
        narrative = llm.generate_text(
            prompt=prompt,
            system=_SYSTEM,
            temperature=0.6,
            max_tokens=500,
        )
        if len(narrative.strip().split()) < 30:
            narrative = fallback
    except (llm.LLMUnavailable, Exception):
        narrative = fallback

    # Write to file
    if session_dir is not None:
        out_path = Path(session_dir) / "learning_path.md"
        out_path.write_text(
            f"# Personalized Learning Path\n\n"
            f"**Student:** {student_id}  \n\n"
            f"{narrative}\n",
            encoding="utf-8",
        )

    return narrative
