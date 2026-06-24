"""
run_real_student_session.py — LLM-enhanced real student session (v2).

What's new vs v1
-----------------
* XAI block printed after every topic selection.
* Per-question result line: time_taken, time_ratio, ability before → after (Δ).
* Learning path printed to terminal at session end.
* Rich ability update (session_ability.py): continuous, streak-aware.
* Gap-based LLM routing (question_selector.py): LLM fires when no bank
  question is within difficulty gap < 1.0 of the target — not on a fixed
  play-count schedule.
* Initial ability: NEW students start at a random value in [15, 20].
  RETURNING students resume from their last session's final ability.
* Persistent ability + asked-question tracking across sessions via
  student_history.json (migrates old format automatically).

Usage
-----
    # Interactive, DQN
    python run_real_student_session.py \\
        --model runs/double_dqn/models/final_model.pt --student-id alice

    # Simulated (for testing / demos)
    python run_real_student_session.py \\
        --model runs/double_dqn/models/final_model.pt \\
        --student-id alice --simulate --sim-ability 22

    # Override starting ability (ignores history for this run)
    python run_real_student_session.py \\
        --model runs/double_dqn/models/final_model.pt \\
        --student-id alice --override-ability 25

    # Disable XAI (faster / quieter)
    python run_real_student_session.py \\
        --model runs/double_dqn/models/final_model.pt --student-id alice --no-xai

    # Disable LLM entirely (bank only)
    python run_real_student_session.py \\
        --model runs/double_dqn/models/final_model.pt --student-id alice --no-llm
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

import rl_common as rc
import student_simulator as sim
import question_bank as qb
from mcq_env import make_env
import difficulty_control as dc
from question_selector import select_question_for_topic
from question_generator import generate_mcq
from explain_answer import generate_explanation
from learning_path import generate_learning_path
from session_xai import explain_topic_choice, format_xai_block
from session_ability import update_session_ability
import retriever
import llm_client as llm

# ---------------------------------------------------------------------------
# Session log fields
# ---------------------------------------------------------------------------
SESSION_FIELDS = [
    "step", "selected_topic", "question_id", "source",
    "target_difficulty", "chosen_difficulty", "gap_ok", "was_repeat",
    "is_correct", "time_taken", "time_ratio",
    "effective_ability_before", "effective_ability_after",
    "running_accuracy", "explanation",
]

HISTORY_FILE = Path(__file__).parent / "student_history.json"
SEP = "=" * 72


# ---------------------------------------------------------------------------
# Student history (v2 format)
# ---------------------------------------------------------------------------
# Schema:
#   {
#     "alice": {
#       "play_count": 3,
#       "effective_ability": 27.5,    # null if not yet tracked
#       "asked_question_ids": [...]   # across ALL sessions
#     }
#   }
# Old format {"alice": 3} is auto-migrated.

def _load_history() -> dict[str, dict]:
    if HISTORY_FILE.exists():
        try:
            raw = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
            out: dict[str, dict] = {}
            for sid, val in raw.items():
                if isinstance(val, int):
                    # migrate old format
                    out[sid] = {
                        "play_count":        val,
                        "effective_ability": None,
                        "asked_question_ids": [],
                    }
                else:
                    out[sid] = val
            return out
        except Exception:
            return {}
    return {}


def _save_history(history: dict[str, dict]) -> None:
    HISTORY_FILE.write_text(json.dumps(history, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="LLM-enhanced real student session v2."
    )
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--student-id", type=str, required=True,
                   help="Student identifier (tracked across sessions).")
    p.add_argument("--n-questions", type=int, default=20)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--device", default="auto")
    p.add_argument("--save-dir", type=Path, default=Path("real_student_sessions"))

    # Ability override (bypasses history for this run)
    p.add_argument("--override-ability", type=float, default=None,
                   help="Force a specific starting ability (ignores student history).")

    # Feature flags
    p.add_argument("--no-xai", action="store_true",
                   help="Suppress the XAI block.")
    p.add_argument("--no-llm", action="store_true",
                   help="Disable LLM features (bank only, static explanations).")
    p.add_argument("--no-coverage-carryover", action="store_true",
                   help="Do NOT seed prior per-topic coverage from history "
                        "(reverts to the old fixed-opening behaviour; useful "
                        "for A/B comparison).")

    # Simulation
    p.add_argument("--simulate", action="store_true",
                   help="Auto-answer with a hidden simulated student.")
    p.add_argument("--manual-time", action="store_true")
    p.add_argument("--sim-ability", type=int, default=30)
    p.add_argument("--sim-style", default=None, choices=[None, *sim.LEARNING_STYLES])
    return p.parse_args()


# ---------------------------------------------------------------------------
# Human answering
# ---------------------------------------------------------------------------
def ask_human(question: dict[str, Any], manual_time: bool = False) -> tuple[str, bool, float]:
    src_tag = " [LLM-generated]" if question.get("source") == "llm" else ""
    print(f"\n  Topic : {question['topic']}{src_tag}")
    print(f"  Q-ID  : {question['question_id']}  |  difficulty: {question['inherent_difficulty']:.1f}")
    print()
    print(f"  {question['question']}")
    for opt in ("A", "B", "C", "D"):
        print(f"    {opt}. {question[f'option_{opt}']}")
    print()
    start = time.perf_counter()
    while True:
        ans = input("  Your answer (A/B/C/D): ").strip().upper()
        if ans in ("A", "B", "C", "D"):
            break
        print("  → Please type A, B, C or D")
    elapsed = time.perf_counter() - start
    if manual_time:
        while True:
            raw = input("  Time taken (seconds): ").strip()
            try:
                elapsed = float(raw)
                if elapsed > 0:
                    break
            except ValueError:
                pass
            print("  → Please enter a positive number")
    is_correct = ans == question["answer"]
    return ans, is_correct, round(float(elapsed), 2)


# ---------------------------------------------------------------------------
# State snapshot for difficulty_policy
# ---------------------------------------------------------------------------
def _state_snapshot(env) -> dict:
    obs = env._get_obs()
    return {
        "effective_ability":       env.effective_ability,
        "last_3_accuracy":         float(obs[5]),
        "last_4_accuracy":         float(obs[6]),
        "last_3_avg_time_ratio":   float(obs[7]) * 3.0,
        "last_4_avg_time_ratio":   float(obs[8]) * 3.0,
        "consecutive_correct":     int(round(float(obs[11]) * 5)),
        "consecutive_wrong":       int(round(float(obs[12]) * 5)),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _wrong_option(question: dict[str, Any]) -> str:
    correct = question["answer"]
    for opt in ("A", "B", "C", "D"):
        if opt != correct:
            return opt
    return "A"


def _print_learning_path(path_text: str, ordered_topics: list[str]) -> None:
    print()
    print(SEP)
    print("PERSONALIZED LEARNING PATH")
    print(SEP)
    print(path_text)
    if ordered_topics:
        print()
        print("Prerequisite-ordered study sequence:")
        for i, t in enumerate(ordered_topics[:10], 1):
            print(f"  {i:2d}. {t}")
    print(SEP)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    rng_seed = np.random.default_rng(args.seed)

    # ── Load RL model ─────────────────────────────────────────────────────────
    # Model formats supported (all act over CHAPTER SLOTS, Discrete(n_active_
    # topics) - see mcq_env.TopicSelectionMCQEnv - mapped to a global topic via
    # env.active_idx[slot]):
    #   *.zip -> sb3-contrib RecurrentPPO checkpoint (obs_dim=89).
    #   *.pt, algo in {a2c_lstm, double_dqn_lstm} -> from-scratch recurrent
    #            torch nets (rl_common.RecurrentActorCritic / RecurrentQNetwork).
    #   *.pt, algo in {a2c, dqn, double_dqn} -> legacy memoryless nets.
    # The LSTM-based models carry their hidden state across the whole session
    # so the policy can infer the student's hidden learning style/forgetting.
    is_recurrent_ppo = args.model.suffix == ".zip"
    device = rc.get_device(args.device)
    is_torch_recurrent = False
    torch_hidden = None
    if is_recurrent_ppo:
        from sb3_contrib import RecurrentPPO
        from train_recurrent_ppo import RecurrentPolicyWrapper
        algo = "recurrent_ppo"
        model = RecurrentPPO.load(str(args.model), device=device)
        rppo_policy = RecurrentPolicyWrapper(model, deterministic=False)
    else:
        ckpt   = rc.load_model(args.model, device)
        algo   = ckpt["algo"]
        model  = rc.build_model_from_checkpoint(ckpt, device)
        is_torch_recurrent = algo in rc.RECURRENT_ALGOS

    # ── Student history ────────────────────────────────────────────────────────
    history = _load_history()
    student_data: dict = history.get(args.student_id, {})
    play_count: int = student_data.get("play_count", 0) + 1

    # Initial effective ability
    if args.override_ability is not None:
        initial_ability = float(args.override_ability)
        print(f"  [override] Starting ability set to {initial_ability:.1f}")
    elif student_data.get("effective_ability") is not None:
        # Returning student: resume from where they left off
        initial_ability = float(student_data["effective_ability"])
    else:
        # New student: random starting ability around the midpoint of the
        # 10..50 scale (30 = "average"), rebased from the old 10..30 default
        # so a fresh learner is not modelled as weak (LIMITATIONS C4).
        initial_ability = float(random.randint(26, 34))

    # Cross-session asked question ids (to avoid repetition)
    asked_ids_history: set[str] = set(student_data.get("asked_question_ids", []))

    # ── LLM availability ──────────────────────────────────────────────────────
    use_llm = (not args.no_llm) and llm.is_available(force_check=True)
    if args.no_llm:
        print("LLM disabled (--no-llm).")
    elif not use_llm:
        print("WARNING: Groq LLM not available (check GROQ_API_KEY) — falling back to bank only.")

    # ── Header ────────────────────────────────────────────────────────────────
    print()
    print(SEP)
    print(f"  ADAPTIVE MECHATRONICS SESSION  —  {algo.upper()} model")
    print(SEP)
    print(f"  Student   : {args.student_id!r}   (session #{play_count})")
    print(f"  Starting ability: {initial_ability:.1f}   "
          f"({'resumed from last session' if student_data.get('effective_ability') is not None else 'random new start'})")
    print(f"  Questions : {args.n_questions}   "
          f"LLM routing: gap-based (fires when bank gap ≥ 1.0)")
    print(f"  Model     : {args.model}")
    print(f"  Questions seen in previous sessions: {len(asked_ids_history)}")
    print(SEP)

    # ── RL environment ────────────────────────────────────────────────────────
    env = make_env(sub_episode_length=args.n_questions, n_sub_episodes=1,
                   seed=args.seed, randomize_initial_ability=False)
    env.reset(seed=args.seed)
    if is_recurrent_ppo:
        rppo_policy.reset()  # fresh LSTM hidden state for this student session
    elif is_torch_recurrent:
        torch_hidden = None  # fresh LSTM hidden state for this student session
    env.effective_ability = float(sim.clip(initial_ability, sim.MIN_ABILITY, sim.MAX_ABILITY))
    env.initial_effective_ability = env.effective_ability
    # re-seed the delta-driven difficulty controller from this student's start
    # ability (reset seeded it from a random ability we just overrode).
    env.target_difficulty = dc.initial_target_difficulty(env.effective_ability)

    # ── Coverage carryover ────────────────────────────────────────────────────
    # Seed the env's per-topic counters from what this student has already
    # covered in previous sessions.  env.reset() zeroes these, so we inject
    # them here (BEFORE the loop).  The trained policy then observes prior
    # coverage (topic_seen / topic_accuracy / topic_asked_rate) at step 0 and
    # naturally moves to less-covered topics instead of replaying the same
    # fixed opening every session.  Topics stay fully askable — availability is
    # driven by the per-question asked_mask, which we do NOT touch here.
    topic_stats: dict = student_data.get("topic_stats", {})
    if not args.no_coverage_carryover and topic_stats:
        seeded = 0
        for topic, st in topic_stats.items():
            ti = env.topic_to_idx.get(topic)
            if ti is None:
                continue
            env.topic_asked[ti]   = float(st.get("asked", 0))
            env.topic_correct[ti] = float(st.get("correct", 0))
            env.topic_wrong[ti]   = float(st.get("wrong", 0))
            seeded += 1
        print(f"  Coverage carryover: seeded prior stats for {seeded} topic(s)")

    # ── Simulated answerer ────────────────────────────────────────────────────
    sim_student = None
    rng_np = np.random.default_rng(args.seed)
    if args.simulate:
        sim_student = sim.create_student(
            "sim_real", ability=args.sim_ability, rng=rng_np,
            learning_style=args.sim_style,
        )

    # ── Session loop ──────────────────────────────────────────────────────────
    rows: list[dict[str, Any]] = []
    n_llm_actual = 0
    n_bank_actual = 0
    n_llm_fallback = 0
    asked_ids_session: set[str] = set()

    def _retrieve(topic: str) -> list[dict]:
        return retriever.retrieve_for_topic(topic, k=6)

    for i in range(args.n_questions):
        print()
        print(f"{'─'*72}")
        print(f"  QUESTION {i+1}/{args.n_questions}   "
              f"ability={env.effective_ability:.2f}   "
              f"accuracy so far={sum(r['is_correct'] for r in rows)}/{i}")

        # 1. RL selects topic. The model always outputs a CHAPTER SLOT index
        # (0..n_active_topics-1) - mcq_env's action space - mapped to a global
        # topic via env.active_idx[slot].
        obs = env._get_obs()
        mask = rc.valid_topic_mask(env)  # slot-length mask
        # Capture the hidden state as it was BEFORE this decision (recurrent
        # algos only) so the XAI explanation below can re-run the same
        # context for occlusion without it having already advanced.
        hidden_before = None
        if is_recurrent_ppo:
            hidden_before = rppo_policy.state
            slot = rppo_policy(env, obs, mask, rng_np)
        elif is_torch_recurrent:
            hidden_before = torch_hidden
            if algo == "a2c_lstm":
                slot, torch_hidden = rc.greedy_topic_from_recurrent_actor(model, obs, mask, torch_hidden, device)
            else:
                slot, torch_hidden = rc.greedy_topic_from_recurrent_q(model, obs, mask, torch_hidden, device)
        else:
            slot = rc.greedy_topic(model, algo, obs, mask, device)
        topic_idx = int(env.active_idx[slot])
        topic = env.topics[topic_idx]

        # 2. Target difficulty for the NEXT question = delta-driven controller
        #    (updated after each answer inside env.apply_external_answer).
        target_diff = float(env.target_difficulty)

        # 3. XAI block - explains the slot actually chosen above (not an
        # independently-recomputed argmax, which could differ for a
        # stochastic policy like RecurrentPPO's default sampling).
        if not args.no_xai:
            try:
                xai = explain_topic_choice(model, algo, obs, mask, env, device,
                                           selected_slot=slot, hidden_state=hidden_before,
                                           precomputed_target_diff=target_diff)
                print()
                print(format_xai_block(xai, i))
            except Exception as e:
                print(f"  [XAI unavailable: {e}]")

        # 4. Gap-based question selection
        #    Combines session + history so neither episode nor cross-session repeats
        asked_combined = asked_ids_session | asked_ids_history

        q_global, question, actual_source, was_repeat, gap_ok = select_question_for_topic(
            topic=topic,
            topic_idx=topic_idx,
            target_diff=target_diff,
            asked_ids=asked_combined,
            env=env,
            use_llm=use_llm,
            retrieve_fn=_retrieve,
            generate_fn=generate_mcq,
        )

        if actual_source == "llm":
            n_llm_actual += 1
        else:
            n_bank_actual += 1

        gap_msg = "" if gap_ok else "  ⚠ gap>1.0 (fallback)"
        src_label = "🤖 LLM-generated" if actual_source == "llm" else "📚 Question bank"
        print(f"\n  Source: {src_label}{gap_msg}"
              f"  target_diff={target_diff:.2f}  chosen_diff={float(question['inherent_difficulty']):.2f}")
        if was_repeat:
            print("  ⚠ All bank questions for this topic seen — repetition allowed")

        # 5. Get answer
        if args.simulate:
            sim_student = sim.with_ability(sim_student, env.effective_ability)
            result      = sim.simulate_answer(sim_student, question, rng_np)
            is_correct  = bool(result["is_correct"])
            time_taken  = float(result["time_taken"])
            student_answer = question["answer"] if is_correct else _wrong_option(question)
            upd        = sim.apply_learning_update(
                sim_student, question, is_correct,
                time_taken / max(float(question.get("base_time", 30)), 1e-6), rng_np,
            )
            sim_student = upd["student"]
            print(f"  Topic : {topic}   diff={question['inherent_difficulty']:.1f}  "
                  f"target={target_diff:.1f}")
            print(f"  → Simulated: {student_answer}  "
                  f"({'CORRECT' if is_correct else 'WRONG'})  time={time_taken:.1f}s")
        else:
            student_answer, is_correct, time_taken = ask_human(
                question, manual_time=args.manual_time
            )

        # 6. Update environment state
        prev_ability = env.effective_ability
        info         = env.apply_external_answer(topic_idx, q_global, is_correct, time_taken)
        time_ratio   = info["time_ratio"]
        eff_before   = info["effective_ability_before"]

        # Override env.effective_ability with the difficulty-aware formula
        rich_after, rich_delta = update_session_ability(
            env, is_correct, time_ratio, prev_ability,
            float(question["inherent_difficulty"]),
        )
        eff_after    = rich_after
        running_acc  = info["accuracy"]

        # Track asked question (bank questions only — LLM questions are unique)
        qid = question["question_id"]
        asked_ids_session.add(qid)

        # 7. Per-question result line
        result_icon = "✓ CORRECT" if is_correct else "✗ WRONG  "
        streak_info = ""
        if is_correct and env.consecutive_correct >= 2:
            streak_info = f" | streak ✓×{env.consecutive_correct}"
        elif not is_correct and env.consecutive_wrong >= 2:
            streak_info = f" | streak ✗×{env.consecutive_wrong}"
        print(f"\n  {result_icon} | time={time_taken:.1f}s | ratio={time_ratio:.2f}"
              f"{streak_info} | ability {eff_before:.2f} → {rich_after:.2f} "
              f"(Δ{rich_delta:+.3f}) | acc={running_acc*100:.0f}%")

        # 8. Dynamic explanation
        if use_llm:
            chunks_for_explain = retriever.retrieve(topic, question["question"], k=5)
            explanation_text = generate_explanation(
                question=question,
                student_answer=student_answer,
                is_correct=is_correct,
                chunks=chunks_for_explain,
            )
        else:
            explanation_text = question.get("explanation", "")

        if explanation_text:
            print(f"\n  💡 {explanation_text}")

        rows.append({
            "step":                    i,
            "selected_topic":          topic,
            "question_id":             qid,
            "source":                  actual_source,
            "target_difficulty":       round(target_diff, 3),
            "chosen_difficulty":       round(float(question["inherent_difficulty"]), 3),
            "gap_ok":                  gap_ok,
            "was_repeat":              was_repeat,
            "is_correct":              is_correct,
            "time_taken":              round(time_taken, 2),
            "time_ratio":              round(time_ratio, 3),
            "effective_ability_before": eff_before,
            "effective_ability_after":  eff_after,
            "running_accuracy":         round(running_acc, 4),
            "explanation":              explanation_text,
        })

    # ── Session summary ────────────────────────────────────────────────────────
    report    = env.get_topic_report()
    topo_path = env.suggest_learning_path()
    n_correct = sum(r["is_correct"] for r in rows)

    # ── Save session ──────────────────────────────────────────────────────────
    stamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.save_dir / f"{stamp}_{algo}_{args.student_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # LLM learning path narrative
    narrative = generate_learning_path(
        ordered_topics=topo_path,
        topic_report=report,
        student_id=args.student_id,
        session_dir=out_dir,
    )

    summary = {
        "student_id":                args.student_id,
        "play_count":                play_count,
        "llm_routing":               "gap_based",
        "n_llm_actual":              n_llm_actual,
        "n_bank_actual":             n_bank_actual,
        "model":                     str(args.model),
        "algo":                      algo,
        "n_questions":               args.n_questions,
        "initial_effective_ability": round(env.initial_effective_ability, 3),
        "final_effective_ability":   round(env.effective_ability, 3),
        "accuracy":                  round(n_correct / max(1, len(rows)), 4),
        "unique_questions":          len({r["question_id"] for r in rows}),
        "repeated_questions":        sum(r["was_repeat"] for r in rows),
        "topics_touched":            sorted({r["selected_topic"] for r in rows}),
        "suggested_learning_path":   topo_path,
    }

    # CSV log
    with (out_dir / "session_log.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SESSION_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in SESSION_FIELDS})

    # JSON summary
    (out_dir / "session_summary.json").write_text(
        json.dumps({"summary": summary, "topic_report": report, "rows": rows},
                   indent=2, default=str),
        encoding="utf-8",
    )

    # ── Update history ─────────────────────────────────────────────────────────
    # Persist effective_ability and all asked question ids
    all_asked = list(asked_ids_history | asked_ids_session)
    # Persist cumulative per-topic coverage.  Because we seeded the env counters
    # from history at the start, env.topic_* already hold prior + this-session
    # totals, so we can read them straight back out.
    topic_stats_out: dict = {}
    for topic, ti in env.topic_to_idx.items():
        asked = int(env.topic_asked[ti])
        if asked > 0:
            topic_stats_out[topic] = {
                "asked":   asked,
                "correct": int(env.topic_correct[ti]),
                "wrong":   int(env.topic_wrong[ti]),
            }
    history[args.student_id] = {
        "play_count":         play_count,
        "effective_ability":  round(env.effective_ability, 4),
        "asked_question_ids": all_asked,
        "topic_stats":        topic_stats_out,
    }
    _save_history(history)

    # ── Console summary ────────────────────────────────────────────────────────
    print()
    print(SEP)
    print("  SESSION COMPLETE")
    print(SEP)
    print(f"  Student  : {args.student_id!r}  (session #{play_count})")
    print(f"  Score    : {n_correct}/{len(rows)} correct ({summary['accuracy']*100:.0f}%)")
    print(f"  Ability  : {summary['initial_effective_ability']:.2f} → "
          f"{summary['final_effective_ability']:.2f}")
    print(f"  Unique Q : {summary['unique_questions']}  |  repeats: {summary['repeated_questions']}")
    print(f"  LLM Qs  : {n_llm_actual}  |  bank: {n_bank_actual}")
    print(f"  Cross-session Q pool: {len(all_asked)} questions seen total")
    topics_str = ", ".join(summary["topics_touched"][:6])
    if len(summary["topics_touched"]) > 6:
        topics_str += " ..."
    print(f"  Topics   : ({len(summary['topics_touched'])}) {topics_str}")
    print(f"  Saved    : {out_dir}")
    print(SEP)

    # ── Print learning path ────────────────────────────────────────────────────
    _print_learning_path(narrative, topo_path)


if __name__ == "__main__":
    main()
