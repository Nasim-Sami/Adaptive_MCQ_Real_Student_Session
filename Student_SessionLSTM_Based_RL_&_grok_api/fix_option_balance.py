"""
fix_option_balance.py - rebalances MCQ option lengths across the question bank.

The problem: in a large fraction of bank questions, the correct answer's
option text is noticeably longer/more detailed than the three distractors -
a classic MCQ-authoring tell that lets a test-taker guess correctly without
knowing the material. Measured on this bank: 502/675 questions (74%) flagged
by the length-imbalance heuristic below, and the correct answer was the
LONGEST option in 501 of those 502 cases (99.8%) - i.e. length alone is an
almost perfect signal for the correct answer in flagged questions.

Fix: for each flagged question, ask the LLM (Groq) to rewrite all 4 option
texts to similar length while preserving the technical correctness of the
correct answer and the plausibility of the distractors. Writes the result
back into the _qbank_partN_expanded.py source files, regenerated in a
consistent style (docstring header preserved verbatim; only the QUESTIONS
list body is reformatted).

Usage:
    python fix_option_balance.py --report      # just print imbalance stats, no changes
    python fix_option_balance.py --dry-run     # show which questions would be rewritten, no LLM calls
    python fix_option_balance.py               # actually fix them (writes the 5 source files)
"""
from __future__ import annotations

import argparse
import importlib
import re
import sys
from pathlib import Path
from typing import Any

import llm_client as llm

BASE_DIR = Path(__file__).resolve().parent
PART_FILES = [f"_qbank_part{i}_expanded" for i in range(1, 6)]
OPTIONS = ["A", "B", "C", "D"]

# Flag a question if the longest option is >1.5x the average of the other
# three AND at least 15 chars longer in absolute terms (avoids flagging
# trivially short options where the ratio is noisy but the gap is tiny).
RATIO_THRESHOLD = 1.5
MIN_ABS_GAP = 15


def is_imbalanced(q: dict[str, Any]) -> tuple[bool, str, dict[str, int]]:
    lens = {opt: len(q.get(f"option_{opt}", "")) for opt in OPTIONS}
    max_opt = max(lens, key=lens.get)
    max_len = lens[max_opt]
    others_avg = (sum(lens.values()) - max_len) / 3
    flagged = others_avg > 0 and max_len > others_avg * RATIO_THRESHOLD and (max_len - others_avg) > MIN_ABS_GAP
    return flagged, max_opt, lens


_SYSTEM = (
    "You are an exam-question editor. You rewrite multiple-choice options so "
    "all four are similar in length and grammatical style, without changing "
    "which answer is correct, without making the question easier or harder, "
    "and without introducing any length/specificity/hedging cue that gives "
    "away the correct answer. Respond only with a JSON object."
)


def _build_prompt(question: dict[str, Any]) -> str:
    opts = "\n".join(f"{opt}. {question.get(f'option_{opt}', '')}" for opt in OPTIONS)
    return (
        f"Topic: {question['topic']}\n"
        f"Question: {question['question']}\n"
        f"Current options:\n{opts}\n"
        f"Correct answer: {question['answer']}\n\n"
        "Rewrite all 4 options so they are approximately the same length (within "
        "a few words of each other) and the same grammatical style/specificity. "
        f"Option {question['answer']} must remain the technically correct answer "
        "(same meaning, may be reworded). The three incorrect options must stay "
        "clearly wrong but plausible and in the same domain - do not make them "
        "silly or off-topic. Do not let length, detail level, or qualifying "
        "language hint at which option is correct.\n\n"
        'Return ONLY this JSON shape: {"option_A": "...", "option_B": "...", '
        '"option_C": "...", "option_D": "..."}'
    )


def rewrite_options(question: dict[str, Any], *, max_attempts: int = 2) -> dict[str, str] | None:
    for attempt in range(max_attempts):
        try:
            result = llm.generate_json(
                prompt=_build_prompt(question),
                system=_SYSTEM,
                temperature=0.6,
                max_tokens=400,
            )
        except Exception:
            continue
        if not isinstance(result, dict):
            continue
        new_opts = {opt: str(result.get(f"option_{opt}", "")).strip() for opt in OPTIONS}
        if any(len(v) < 3 for v in new_opts.values()):
            continue
        if len({v.lower() for v in new_opts.values()}) < 4:
            continue  # duplicate options - reject
        lens = {opt: len(v) for opt, v in new_opts.items()}
        max_len = max(lens.values())
        others_avg = (sum(lens.values()) - max_len) / 3
        if others_avg > 0 and max_len > others_avg * RATIO_THRESHOLD and (max_len - others_avg) > MIN_ABS_GAP:
            continue  # still imbalanced - retry
        return new_opts
    return None


# ---------------------------------------------------------------------------
# Source-file regeneration
# ---------------------------------------------------------------------------
def _pystr(s: Any) -> str:
    """Render a Python double-quoted string literal, always double-quoted
    (matches the bank's house style) regardless of internal quote chars."""
    s = str(s)
    s = s.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{s}"'


def format_question(q: dict[str, Any]) -> str:
    ds = q.get("distractor_strength", {})
    ds_str = "{" + ", ".join(f"{_pystr(k)}: {v}" for k, v in ds.items()) + "}"
    lines = [
        f'    {{"question_id": {_pystr(q["question_id"])}, "topic": {_pystr(q["topic"])}, '
        f'"inherent_difficulty": {q["inherent_difficulty"]},',
        f'     "question": {_pystr(q["question"])},',
        f'     "option_A": {_pystr(q["option_A"])},',
        f'     "option_B": {_pystr(q["option_B"])},',
        f'     "option_C": {_pystr(q["option_C"])},',
        f'     "option_D": {_pystr(q["option_D"])},',
        f'     "answer": {_pystr(q["answer"])}, "distractor_strength": {ds_str},',
        f'     "subtopic": {_pystr(q.get("subtopic", ""))},',
        f'     "explanation": {_pystr(q.get("explanation", ""))}}},',
    ]
    return "\n".join(lines)


def rewrite_source_file(module_name: str, questions: list[dict[str, Any]]) -> None:
    path = BASE_DIR / f"{module_name}.py"
    original = path.read_text(encoding="utf-8")
    m = re.search(r"^QUESTIONS\s*=\s*\[", original, re.MULTILINE)
    if not m:
        raise RuntimeError(f"Could not find 'QUESTIONS = [' in {path}")
    header = original[: m.start()]
    body = "QUESTIONS = [\n" + "\n".join(format_question(q) for q in questions) + "\n]\n"
    path.write_text(header + body, encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true", help="Print imbalance stats only, no changes.")
    ap.add_argument("--dry-run", action="store_true", help="Show flagged questions, no LLM calls or writes.")
    args = ap.parse_args()

    if not args.report and not args.dry_run and not llm.is_available(force_check=True):
        print("ERROR: Groq is not available (check GROQ_API_KEY). Aborting.", file=sys.stderr)
        sys.exit(1)

    total_flagged = 0
    total_fixed = 0
    total_failed = 0

    for module_name in PART_FILES:
        module = importlib.import_module(module_name)
        questions: list[dict[str, Any]] = module.QUESTIONS
        flagged_in_file = 0
        fixed_in_file = 0
        changed = False

        for q in questions:
            flagged, max_opt, lens = is_imbalanced(q)
            if not flagged:
                continue
            flagged_in_file += 1
            total_flagged += 1
            correct_is_longest = max_opt == q["answer"]

            if args.report:
                continue
            if args.dry_run:
                print(f"  [{module_name}] {q['question_id']} - longest={max_opt} "
                      f"({'CORRECT' if correct_is_longest else 'distractor'}) lens={lens}")
                continue

            new_opts = rewrite_options(q)
            if new_opts is None:
                total_failed += 1
                print(f"  [FAILED] {module_name} {q['question_id']} - keeping original options")
                continue
            q.update(new_opts)
            fixed_in_file += 1
            total_fixed += 1
            changed = True

        print(f"{module_name}: {flagged_in_file} flagged" +
              (f", {fixed_in_file} fixed" if not args.report and not args.dry_run else ""))

        if changed:
            rewrite_source_file(module_name, questions)
            print(f"  -> wrote {module_name}.py")

    print(f"\nTotal flagged: {total_flagged}")
    if not args.report and not args.dry_run:
        print(f"Total fixed: {total_fixed}")
        print(f"Total failed (left unchanged): {total_failed}")


if __name__ == "__main__":
    main()
