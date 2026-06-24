---
title: Adaptive MCQ LLM LSTM RL Based
emoji: 🧠
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# LLM-Enhanced Adaptive Mechatronics — v2.4

Topic-selection RL system (the agent picks a *topic*; a rule serves the
*question*) extended with an LLM (Groq cloud API) for grounded explanations,
question generation, and learning paths — plus a reworked adaptive-difficulty
engine and three memory-equipped (LSTM) RL agents. PDF-chunk retrieval
embeddings still run locally via Ollama (`nomic-embed-text`) — see
`llm_client.py` / `retriever.py`.

> ⚠ **Retraining is required.** Each version (v2.1 → v2.4) changed the
> observation, action space, and/or reward the policy is trained against, so
> models from an older version are **not** compatible with the current env.
> The pipeline itself needs no code edits to train — `obs_dim`/`n_actions` are
> read from the env at trainer start-up.

---

## Updates v2.1 – v2.3

| Area | Change | Files |
|---|---|---|
| Ability range | 10–30 → **10–50, continuous** (no integer rounding) | `question_bank.py`, `student_simulator.py`, `mcq_env.py` |
| Ability update | difficulty-aware **Elo-style** (beat a hard question ⇒ big rise) | `student_simulator.py`, `session_ability.py` |
| **Difficulty selection** | **delta-driven controller** (`target_difficulty`) instead of scaling ability — hard questions appear when *earned*, not at ability 50 | `difficulty_control.py` (new), `mcq_env.py`, `run_real_student_session.py` |
| Topic exploration | **coverage carryover** across sessions + **anti-massing** (≤2 same topic in a row) | `run_real_student_session.py`, `mcq_env.py` |
| Question bank | **4 → 15 per topic (675 total)**; old 4-per-topic files removed | `_qbank_part*_expanded.py`, `question_bank.py` |
| Diagnosis | torch-free runner to inspect topic + difficulty adaptation | `diagnose_np.py` (new) |
| New-student start | rebased to ~26–34 (30 = average on 10–50) | `run_real_student_session.py` |

The delta equation, coefficients, and verification are in `CHANGES_v2.1.md`,
`CHANGES_v2.2.md`, `CHANGES_v2.3.md`. Limitations that motivated them are in
`LIMITATIONS_and_NEXT.md`.

---

## Updates v2.4 — three memory-equipped agents

`v2.4` replaces the flat-reward a2c/double_dqn setup with **three** recurrent
(LSTM) agents trained on a redesigned env: forgetting/spacing dynamics, a
sparse **true-mastery** reward (the agent must genuinely teach, not farm
shaped bonuses), and a chapter-focused `Discrete(10)` action space (the agent
picks among 10 active topics per episode, not all 45). See `CHANGES_v2.4.md`
for the full design rationale and the baseline-vs-trained scoreboard.

| Trainer | Algorithm | On/off-policy | Checkpoint format | Naming note |
|---|---|---|---|---|
| `train_recurrent_ppo.py` | `sb3-contrib` `RecurrentPPO` (LSTM policy) | on-policy | `.zip` (SB3) | named after the literal SB3 class |
| `train_a2c_lstm.py` | from-scratch A2C + LSTM | on-policy | `.pt`, `algo="a2c_lstm"` | hand-rolled (no SB3 class to mirror) — named for the added LSTM; functionally "RecurrentA2C" |
| `train_double_dqn_lstm.py` | from-scratch Double DQN + LSTM (DRQN-style) | off-policy | `.pt`, `algo="double_dqn_lstm"` | same as above — functionally "RecurrentDoubleDQN" |
| `train_double_dqn.py` | from-scratch Double DQN (memoryless) | off-policy | `.pt`, `algo="double_dqn"` | legacy, kept as a baseline |
| `train_a2c.py` | from-scratch A2C (memoryless) | on-policy | `.pt`, `algo="a2c"` | legacy, kept as a baseline |

All five checkpoint formats are auto-detected by `run_real_student_session.py`,
`evaluate_baselines.py`, and `final_scoreboard.py` — by file extension for
`.zip`, by the `algo` field for `.pt`. **Full train / resume / real-session
commands for every model are in the [Training](#training) and [Running a
real student session](#running-a-real-student-session) sections below — that
is the single canonical command reference; this section is descriptive only.**

Each trainer prints its own final scoreboard (heuristics + itself) at the end
of training. To compare **all trained agents side-by-side** in one ladder,
load multiple model paths via separate `evaluate_baselines.py --model <path>`
runs, or extend `final_scoreboard.py`'s `policies` dict with
`rc.RecurrentTorchPolicyWrapper(model, "a2c_lstm"|"double_dqn_lstm", device)`
for the two torch checkpoints next to the existing `RecurrentPPO[...]` entry.

The XAI block (`--no-xai` to suppress) is enabled and works across all five
model formats. It explains the CHAPTER SLOT actually chosen (mapped through
`env.active_idx`, not an independently-recomputed argmax — important for
stochastic policies like RecurrentPPO's default sampling), and for the
recurrent algos the occlusion counterfactuals reuse the same pre-decision
hidden state that produced the real choice.

---

## What changed (real session only)

| Problem | Fix |
|---|---|
| Fixed explanations | `explain_answer.py` — grounded LLM explanation after every answer |
| No live stats | Per-question line shows time, ratio, ability Δ, streak |
| Static ability update | `session_ability.py` — continuous formula, rewards correct streaks |
| Play-count LLM schedule | `question_selector.py` — LLM fires when bank gap ≥ 1.0 |
| Same question every session | Cross-session `asked_question_ids` in history; random initial ability |
| Fixed starting ability | New students start at random ability in [15, 20] |
| Ability lost between sessions | `effective_ability` persisted in `student_history.json` |
| No topic-choice reasoning | `session_xai.py` — occlusion XAI block before every question |
| Simple difficulty target | `difficulty_policy.py` — ability + accuracy + streaks + time + dither |
| Hidden learning path | Learning path narrative printed to terminal at session end |
| Bolton PDF unreadable | `pdf_ingest.py` — OCR fallback via `pdf2image` + `pytesseract` |

---

## File map

| File | New/Edit | Role |
|---|---|---|
| `pdf_ingest.py` | edit | One-time: extract + chunk both PDFs (OCR fallback for Bolton) |
| `retriever.py` | new | Embed chunks with `nomic-embed-text`; cosine retrieval |
| `llm_client.py` | edit (v2.4) | Groq cloud API wrapper for generation (JSON mode, timeout, retry) — embeddings stay on Ollama, see `retriever.py` |
| `explain_answer.py` | new | Dynamic grounded explanation after each answer |
| `difficulty_control.py` | new (v2.3) | Delta-driven difficulty controller (the equation that picks next-question difficulty) |
| `difficulty_policy.py` | legacy | Old ability-scaling target (kept; no longer the driver) |
| `question_selector.py` | new | Gap-based routing: bank if gap < 1.0, else LLM-generate at target |
| `question_generator.py` | new | Grounded MCQ generation + validator |
| `learning_path.py` | new | LLM narrative learning path + `learning_path.md` output |
| `session_xai.py` | edit (v2.4) | In-session XAI (topic scores, occlusion, curriculum) — supports all 5 model formats, slot-aware |
| `session_ability.py` | edit (v2.2) | Difficulty-aware Elo ability update (real session) |
| `student_simulator.py` | edit (v2.1–2.4) | Continuous ability, Elo update, forgetting/spacing, widened style multiplier |
| `mcq_env.py` | edit (v2.1–2.4) | Continuous init ability, difficulty controller, chapter-focused action space, sparse true-mastery reward |
| `curriculum.py` | edit (v2.4) | `TRANSFER_SCALE` |
| `question_bank.py` | edit (v2.3) | Loads 675-question expanded bank; bank↔difficulty map |
| `_qbank_part*_expanded.py` | new (v2.3) | 45 topics × 15 MCQs = 675 (replaces the 4-per-topic files) |
| `diagnose_np.py` | new | Torch-free model loader + session diagnosis |
| `student_history.json` | edit | Adds `topic_stats` (coverage carryover) |
| `run_real_student_session.py` | edit | Integrated runner: all 5 model formats, controller, carryover, rebased start |
| `rl_common.py` | edit (v2.4) | Shared nets (legacy + LSTM), `EpisodeReplayBuffer`, recurrent inference helpers |
| `train_double_dqn.py`, `train_a2c.py` | unchanged (mask bug fixed in v2.4) | Legacy memoryless baselines |
| `train_recurrent_ppo.py` | new (v2.4) | `sb3-contrib` RecurrentPPO trainer |
| `train_a2c_lstm.py` | new (v2.4) | From-scratch recurrent A2C trainer |
| `train_double_dqn_lstm.py` | new (v2.4) | From-scratch recurrent (DRQN) Double DQN trainer |
| `evaluate_baselines.py` | new (v2.4) | Baseline scoreboard: random / heuristics / style-oracle / trained |
| `final_scoreboard.py` | new (v2.4) | RecurrentPPO vs heuristics, stochastic + argmax eval |
| `analyze_mimicry.py` | new (v2.4) | Behavioural check: did a trained agent just re-learn a heuristic? |
| `question_source_router.py` | obsolete | Replaced by `question_selector.py` |
| `_qbank_part1..5.py` (4-per-topic) | removed | Deleted in v2.3 |

---

## LLM routing — gap-based

The old play-count schedule (`play 1 → 5%, play 2 → 10%, …`) has been
replaced with a principled, need-driven approach:

```
1. Find bank questions for the chosen topic that are UNASKED
   (not asked this session AND not in student_history.asked_question_ids)
2. Check: any question with |difficulty − target| < 1.0?
   YES → use closest bank question
   NO  → LLM generates a fresh question at exactly the target difficulty
   LLM fails → fall back to closest bank question regardless of gap
```

With 15 bank questions per topic, the LLM fires roughly when the target falls
in a gap region. As the student accumulates sessions and exhausts bank
questions, LLM usage grows naturally.

---

## Student history format

`student_history.json` stores a full record per student:

```json
{
  "alice": {
    "play_count": 3,
    "effective_ability": 24.71,
    "asked_question_ids": ["T1Q1", "T1Q3", "T5Q2", "..."],
    "topic_stats": {"Measurement systems": {"asked": 5, "correct": 4, "wrong": 1}}
  }
}
```

The old format (`{"alice": 3}`) is auto-migrated on first load.

**Ability persistence:** a returning student always resumes from their last
session's final ability. A new student starts at a random ability in
`[15, 20]` (or `[26, 34]` for the v2.1+ rebased 10–50 scale).

---

## Prerequisites (one-time setup)

```bash
# Python deps
pip install -r requirements.txt
pip install pdfplumber pytesseract numpy requests --break-system-packages

# Tesseract binary (needed for Bolton OCR)
# macOS:  brew install tesseract
# Ubuntu: sudo apt install tesseract-ocr

# Ollama model — embeddings ONLY (PDF-chunk retrieval, retriever.py)
ollama pull nomic-embed-text

# Groq API key — generation (question gen, explanations, learning path),
# see llm_client.py. Free key: https://console.groq.com -> API Keys
export GROQ_API_KEY="gsk_..."
# Default model: llama-3.1-8b-instant (fast, generous free-tier rate limit -
# matters because one session makes several calls back-to-back: an
# explanation per question + occasional question generation + one
# learning-path call). Override with GROQ_MODEL=llama-3.3-70b-versatile for
# higher quality at the cost of hitting free-tier rate limits sooner.

# Extract + chunk both textbooks (OCR runs automatically for Bolton)
python pdf_ingest.py

# Build embedding cache (runs once, ~270 MB)
python retriever.py --build

# Sanity: 675 questions (45 topics × 15)
python question_bank.py
```

---

## Training

Canonical command reference for **all five** trainers. Every trainer reads
`obs_dim`/`n_actions` from the env, so no code edits are needed to train.

### RecurrentPPO (`train_recurrent_ppo.py`)

`sb3-contrib`, LSTM policy, on-policy. Saves flat into `--save-dir` (no
`models/`/`checkpoints/` subfolders) as `best_model.zip` / `final_model.zip`.

```bash
# Full training run (long — several hours for a few million steps)
python train_recurrent_ppo.py \
    --timesteps 3000000 --n-envs 8 --n-steps 1024 --batch-size 512 \
    --lr 3e-4 --ent-coef 0.01 --eval-freq 30000 --eval-episodes 30 \
    --final-eval-episodes 200 --seed 0 --save-dir runs/recurrent_ppo

# Resume / continue training from a saved checkpoint
python train_recurrent_ppo.py \
    --resume-from runs/recurrent_ppo/final_model.zip \
    --timesteps 2000000 --n-envs 8 --n-steps 1024 --batch-size 512 \
    --lr 3e-4 --ent-coef 0.01 --save-dir runs/recurrent_ppo
```

### A2C + LSTM (`train_a2c_lstm.py`)

From-scratch, on-policy; one full (fixed-length) episode = one LSTM sequence,
backprop through the whole episode. Saves to `models/`, `checkpoints/`,
`logs/` under `--save-dir` (same layout as the legacy trainers below).

```bash
# Full training run
python train_a2c_lstm.py \
    --timesteps 1000000 --eval-freq 20000 --eval-episodes 10 \
    --seed 42 --save-dir runs/a2c_lstm

# Quick smoke test
python train_a2c_lstm.py \
    --timesteps 4000 --sub-episode-length 20 --n-sub-episodes 4 \
    --eval-freq 2000 --save-dir runs/a2c_lstm_smoke

# Resume / continue training from a saved checkpoint
python train_a2c_lstm.py \
    --resume-from runs/a2c_lstm/checkpoints/a2c_lstm_ckpt.pt \
    --timesteps 500000 --save-dir runs/a2c_lstm
```

### Double DQN + LSTM / DRQN-style (`train_double_dqn_lstm.py`)

From-scratch, off-policy; `rl_common.EpisodeReplayBuffer` stores whole
episodes (no padding needed — episodes are fixed-length) and samples batches
of full episodes to unroll.

```bash
# Full training run
python train_double_dqn_lstm.py \
    --timesteps 1000000 --eval-freq 20000 --eval-episodes 10 \
    --learning-starts-episodes 50 --seed 42 --save-dir runs/double_dqn_lstm

# Quick smoke test
python train_double_dqn_lstm.py \
    --timesteps 4000 --sub-episode-length 20 --n-sub-episodes 4 \
    --eval-freq 2000 --learning-starts-episodes 5 \
    --save-dir runs/double_dqn_lstm_smoke

# Resume / continue training from a saved checkpoint
python train_double_dqn_lstm.py \
    --resume-from runs/double_dqn_lstm/checkpoints/double_dqn_lstm_ckpt.pt \
    --timesteps 500000 --save-dir runs/double_dqn_lstm
```

### Double DQN (legacy, memoryless) (`train_double_dqn.py`)

```bash
# Full training run (200k timesteps, checkpoint every 100k steps)
python train_double_dqn.py \
    --timesteps 200000 --save-dir runs/double_dqn \
    --checkpoint-freq 100000 --eval-freq 20000 --seed 42

# Quick smoke test
python train_double_dqn.py \
    --timesteps 4000 --sub-episode-length 20 --n-sub-episodes 4 \
    --eval-freq 2000 --learning-starts 500 --save-dir runs/double_dqn_smoke

# Resume after interruption
python train_double_dqn.py \
    --timesteps 200000 --save-dir runs/double_dqn \
    --checkpoint-freq 100000 \
    --resume-from runs/double_dqn/checkpoints/dqn_ckpt.pt
```

### A2C (legacy, memoryless) (`train_a2c.py`)

```bash
# Full training run
python train_a2c.py \
    --timesteps 200000 --save-dir runs/a2c \
    --checkpoint-freq 100000 --eval-freq 20000 --seed 42

# Quick smoke test
python train_a2c.py \
    --timesteps 4000 --sub-episode-length 20 --n-sub-episodes 4 \
    --eval-freq 2000 --save-dir runs/a2c_smoke

# Resume after interruption
python train_a2c.py \
    --timesteps 200000 --save-dir runs/a2c \
    --checkpoint-freq 100000 \
    --resume-from runs/a2c/checkpoints/a2c_ckpt.pt
```

### Training output structure

```
runs/recurrent_ppo/            # flat - no models/checkpoints subfolders
├── best_model.zip
├── final_model.zip
└── tb/                        # tensorboard logs

runs/a2c_lstm/                 runs/double_dqn_lstm/
runs/a2c/                      runs/double_dqn/
├── models/
│   ├── best_model.pt          # best eval checkpoint (by true mastery gain for *_lstm, by reward for legacy)
│   └── final_model.pt         # end of training
├── checkpoints/
│   └── <algo>_ckpt.pt         # resume checkpoint (a2c_ckpt / dqn_ckpt / a2c_lstm_ckpt / double_dqn_lstm_ckpt)
└── logs/
    ├── training_episodes.csv
    └── evaluation.csv
```

> **Checkpoint note:** `--checkpoint-freq` overwrites a *single* resume file
> periodically. `best_model.pt`/`.zip` and `final_model.pt`/`.zip` are always
> preserved separately.

---

## Running a real student session

`run_real_student_session.py` auto-detects the model format (`.zip` →
RecurrentPPO, `.pt` with `algo` ∈ `{a2c_lstm, double_dqn_lstm}` → recurrent
torch net, `.pt` with `algo` ∈ `{a2c, double_dqn}` → legacy memoryless net)
and carries the LSTM hidden state across the whole session where applicable.
All flags below work identically regardless of which model you point at.

### RecurrentPPO

```bash
# Interactive (a real person answers live in the terminal, A/B/C/D)
python run_real_student_session.py \
    --model runs/recurrent_ppo/best_model.zip --student-id alice

# Simulated (for testing / demos)
python run_real_student_session.py \
    --model runs/recurrent_ppo/best_model.zip \
    --student-id alice --simulate --sim-ability 30
```

### A2C + LSTM

```bash
# Interactive
python run_real_student_session.py \
    --model runs/a2c_lstm/models/best_model.pt --student-id alice

# Simulated
python run_real_student_session.py \
    --model runs/a2c_lstm/models/best_model.pt \
    --student-id alice --simulate --sim-ability 30
```

### Double DQN + LSTM

```bash
# Interactive
python run_real_student_session.py \
    --model runs/double_dqn_lstm/models/best_model.pt --student-id alice

# Simulated
python run_real_student_session.py \
    --model runs/double_dqn_lstm/models/best_model.pt \
    --student-id alice --simulate --sim-ability 30
```

### Double DQN (legacy)

```bash
# Interactive
python run_real_student_session.py \
    --model runs/double_dqn/models/final_model.pt --student-id alice

# Simulated
python run_real_student_session.py \
    --model runs/double_dqn/models/final_model.pt \
    --student-id alice --simulate --sim-ability 22
```

### A2C (legacy)

```bash
# Interactive
python run_real_student_session.py \
    --model runs/a2c/models/final_model.pt --student-id alice

# Simulated
python run_real_student_session.py \
    --model runs/a2c/models/final_model.pt \
    --student-id alice --simulate --sim-ability 22
```

### Useful flags (any model)

```bash
# Override starting ability for this run (ignores history)
python run_real_student_session.py --model <path> --student-id alice --override-ability 25

# Suppress XAI block (faster / quieter; XAI is on by default)
python run_real_student_session.py --model <path> --student-id alice --no-xai

# Bank only — no LLM (Groq offline/no API key / testing)
python run_real_student_session.py --model <path> --student-id alice --no-llm

# Disable cross-session coverage carryover (A/B comparison)
python run_real_student_session.py --model <path> --student-id alice --no-coverage-carryover

# Manual (human-typed) time entry instead of auto-measured
python run_real_student_session.py --model <path> --student-id alice --manual-time

# Fix a hidden learning style for a simulated student (massed/interleaved/blocked)
python run_real_student_session.py --model <path> --student-id alice --simulate --sim-style interleaved
```

---

## Comparing trained agents — scoreboards

```bash
# Baseline ladder only (random / heuristics / style-oracle), no trained model
python evaluate_baselines.py --episodes 200

# Baseline ladder + ONE legacy trained model (.pt, a2c/double_dqn)
python evaluate_baselines.py --episodes 200 --model runs/a2c/models/best_model.pt

# Baseline ladder + RecurrentPPO, both stochastic and argmax eval
python final_scoreboard.py --episodes 200 --model runs/recurrent_ppo/best_model.zip

# Behavioural check: did a trained agent just re-learn `lowest_index`,
# or is it doing something different? (action agreement + style-conditioned
# dwell-time/mastery-gain breakdown)
python analyze_mimicry.py --episodes 150
```

---

## Session output (per run)

```
real_student_sessions/<timestamp>_<algo>_<student_id>/
    session_log.csv        # per-question: source, time, ratio, ability Δ, gap_ok
    session_summary.json   # full summary + topic report
    learning_path.md       # personalized LLM narrative (also printed to terminal)
```

---

## Autonomous code audit

`PLAN.md` in this folder is a machine-readable spec. To make Claude
self-audit and fix the code, open Cowork and say:

> "Read PLAN.md in LLM_session_v2 and audit the code. Fix any issues you find."

Claude will check all invariants (ability persistence, gap routing, streak
formula, OCR, history format, etc.) and fix deviations automatically.

---

## Verification checklist

- [ ] `python pdf_ingest.py` produces chunks from BOTH books (bolton + rajput)
- [ ] New student gets random starting ability 15–20 (not always 20)
- [ ] Returning student resumes from last session's final ability
- [ ] `student_history.json` stores `effective_ability` and `asked_question_ids`
- [ ] Same question never appears twice in one episode
- [ ] Questions from previous sessions are excluded first (not asked again)
- [ ] LLM fires when no bank question is within gap < 1.0 of target (not on play count)
- [ ] Per-question line shows `Δ±x.xxx` and streak info
- [ ] Difficulty Δ differs for tr=0.35 vs tr=0.53 correct answers
- [ ] `--no-llm` / no `GROQ_API_KEY` runs complete without errors (bank-only fallback)
- [ ] `_qbank_part*.py` files byte-for-byte identical to the canonical originals
- [ ] All five trainers (`train_recurrent_ppo.py`, `train_a2c_lstm.py`, `train_double_dqn_lstm.py`, `train_a2c.py`, `train_double_dqn.py`) run a 4k-step smoke test without errors
- [ ] All five checkpoint formats load correctly in `run_real_student_session.py`, `evaluate_baselines.py`, and `final_scoreboard.py`
