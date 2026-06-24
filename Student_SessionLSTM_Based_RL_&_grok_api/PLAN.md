# PLAN.md — Autonomous Code Audit & Fix Guide

This file tells Claude how to autonomously read the codebase, find bugs,
and fix them without needing detailed human instructions each time.

---

## How to use this file

Open Cowork and say:

> "Read PLAN.md in LLM_session_v2 and audit the code. Fix any issues you find."

Claude will:
1. Read this file to understand the system
2. Read the relevant source files
3. Identify deviations from the expected behaviour described below
4. Fix them and report what changed

---

## System overview

The `LLM_session_v2/` folder is an adaptive mechatronics learning session with:
- A **DQN or A2C** RL agent that selects which topic to teach next (45 topics)
- A **question bank** (180 questions, 4 per topic, difficulties spaced ~2.0 apart)
- A **qwen2.5:3b LLM** (via Ollama) that generates questions and explanations
- An **embedding retriever** (nomic-embed-text) grounding the LLM in two PDF textbooks

### File map

| File | Purpose |
|---|---|
| `run_real_student_session.py` | Main entry point — runs one student session |
| `session_ability.py` | Rich ability update formula (continuous, streak-aware) |
| `question_selector.py` | Gap-based question routing: bank vs LLM |
| `difficulty_policy.py` | `compute_target_difficulty()` — richer than simple mapping |
| `session_xai.py` | Prints WHY the RL model chose a topic (occlusion XAI) |
| `pdf_ingest.py` | One-time: extract chunks from both PDFs (with OCR fallback) |
| `retriever.py` | Cosine retrieval of PDF chunks |
| `llm_client.py` | Ollama wrapper |
| `explain_answer.py` | Post-answer LLM explanation |
| `question_generator.py` | LLM MCQ generation + validator |
| `learning_path.py` | End-of-session personalised narrative |
| `student_history.json` | Persistent: play_count + effective_ability + asked_question_ids |

**DO NOT MODIFY:** `_qbank_part*.py`, `mcq_env.py`, `student_simulator.py`,
`train_double_dqn.py`, `train_a2c.py` — these are training files.

---

## Invariants to check (automated audit checklist)

When auditing, verify each invariant. If violated, fix it.

### 1. Question never repeated in the same episode
- `asked_ids_session` grows each step
- `select_question_for_topic` receives `asked_ids_session | asked_ids_history`
- `was_repeat` should only be True when every bank question for the topic is exhausted

### 2. Questions not repeated from previous sessions
- `student_history.json` stores `asked_question_ids` per student
- `asked_ids_history` is loaded at session start and passed to selector
- New questions asked this session are added to the history at session end

### 3. Returning student starts at their last effective ability
- `student_data["effective_ability"]` is loaded and used as `initial_ability`
- At session end, `env.effective_ability` is saved back to history

### 4. New student gets random starting ability in [15, 20]
- When `student_data` is empty or `effective_ability` is None
- `initial_ability = float(random.randint(15, 20))`
- NOT a fixed value like 20.0

### 5. LLM fires when bank gap ≥ 1.0 (not on play-count schedule)
- `select_question_for_topic` tries bank first; if no question within
  `GAP_THRESHOLD = 1.0` of `target_diff`, calls LLM
- `question_source_router.py` is NOT imported or used in the session runner

### 6. Ability update uses continuous formula (not step function)
- `update_session_ability` from `session_ability.py` is called AFTER
  `env.apply_external_answer()` to override `env.effective_ability`
- The override uses `rich_ability_delta` with:
  - Two-slope correct formula (gentle decline tr≤1.5, steep tr>1.5)
  - Consecutive-correct streak bonus (+0.03/answer, cap +0.15)
  - Consecutive-wrong streak penalty (-0.10/answer, cap -0.50)

### 7. Target difficulty uses the rich formula
- `compute_target_difficulty(state)` is called before XAI and before question
  selection (not just `ability_to_target_difficulty(env.effective_ability)`)
- It uses: ability, last3/4 accuracy, last3/4 time ratio, streaks, dither

### 8. Bolton PDF is chunked via OCR
- `pdf_ingest.py` uses pdfplumber first; if < 30 words extracted, falls back
  to `pdf2image` + `pytesseract`
- After running `python pdf_ingest.py`, `pdf_chunks.json` must contain
  entries with `"book": "bolton"`

### 9. XAI block shows correct info
- Shows chosen topic vs runner-up with margin
- Shows top-3 available topics
- Shows occlusion attribution (which feature group flips the choice if removed)
- Shows rich target difficulty (not just simple)

### 10. History format is v2
- `student_history.json` entries must be dicts with keys:
  `play_count`, `effective_ability`, `asked_question_ids`
- Old int format is auto-migrated on load

---

## Debugging commands

```bash
# Verify no Bolton chunks (means OCR not run yet)
python3 -c "
import json; c = json.load(open('pdf_chunks.json'))
from collections import Counter; print(Counter(x['book'] for x in c))
"

# Simulate one session (no Ollama needed) and check output
python run_real_student_session.py \
    --model runs/double_dqn/models/final_model.pt \
    --student-id debug --simulate --sim-ability 22 --no-llm --no-xai

# Check student history
cat student_history.json

# Verify ability formula gives different deltas for tr=0.35 vs tr=0.53
python3 -c "
from session_ability import rich_ability_delta
print(rich_ability_delta(True, 0.35, consecutive_correct=1))
print(rich_ability_delta(True, 0.53, consecutive_correct=1))
"
# Expected: two DIFFERENT values (e.g. ~0.46 and ~0.40)

# Verify question selector triggers LLM at correct gap
python3 -c "
import sys; sys.path.insert(0, '.')
from question_selector import GAP_THRESHOLD
from mcq_env import ability_to_target_difficulty
import question_bank as qb, curriculum
topic = curriculum.TOPICS[0]
target = ability_to_target_difficulty(30)  # high ability → high target
qs = qb.questions_for_topic(topic)
close = [q for q in qs if abs(float(q['inherent_difficulty'])-target) < GAP_THRESHOLD]
print(f'topic={topic}  target={target:.2f}  close_questions={len(close)}')
# If 0, LLM should fire for this topic at ability=30
"
```

---

## Known limitations / future improvements

- Each topic has only 4 bank questions (difficulties ~1.6, 3.8, 6.0, 8.2).
  A student who plays many sessions will exhaust bank questions and rely
  increasingly on LLM generation — this is intended behaviour.
- The RL model selects the same first topic for all students with identical
  starting obs. Randomised initial ability [15,20] partially mitigates this.
  A small amount of epsilon-greedy noise during real sessions could be added.
- OCR quality for Bolton varies by chapter; some formula-heavy pages may
  produce garbled text. Post-OCR cleaning (equation stripping) would help.
