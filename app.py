"""
app.py - Flask web app for the topic-selection adaptive-mechatronics MCQ system.

Ported from the June-3 Flask app (Student_Session_Mechatronics_lab_june_3) onto
the v2.4 backend in this folder: instead of a single RL action = one question
(June-3's DynamicAbilityAdaptiveMCQEnv), the agent here picks a CHAPTER SLOT /
topic (Discrete(10), mcq_env.TopicSelectionMCQEnv), then a gap-based rule
(question_selector.py) serves a bank or Groq-LLM-generated question within
that topic. Supports all 5 trained model formats (RecurrentPPO, A2C+LSTM,
Double-DQN+LSTM, legacy A2C, legacy Double DQN) via the same dispatch used in
run_real_student_session.py.

Differences from June-3, per request:
  * Question difficulty and question ID are NEVER sent to the template - only
    topic, question text, and the 4 options.
  * XAI (session_xai.explain_topic_choice) is NOT shown automatically - a
    "Why this topic?" button fetches /episode/<token>/xai on demand.
  * The "Tesla career goal" feature is replaced by a generic, de-branded
    achievement ladder, with thresholds rescaled from the old 10-30 ability
    scale to this project's 10-50 scale (each threshold's distance from 10
    is doubled: 26->42, 27->44, 28->46, 29->48, 30(max)->50(max)).

Account/episode/answer persistence uses the same SQLite schema design as
June-3 (students / episodes / episode_answers tables), extended with
asked_question_ids_json and topic_stats_json on `students` so cross-session
question-repeat avoidance and topic-coverage carryover (already in
run_real_student_session.py) work the same way here.
"""
from __future__ import annotations

import json
import os
import secrets
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from flask import Flask, flash, redirect, render_template, request, session, url_for

import question_bank as qb
import curriculum
import difficulty_control as dc
import student_simulator as sim
import rl_common as rc
import llm_client as llm
import retriever
import question_generator
import explain_answer
import learning_path as lp
from mcq_env import make_env, TopicSelectionMCQEnv
from question_selector import select_question_for_topic
from session_xai import explain_topic_choice


BASE_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEFAULT_MODEL_PATH = BASE_DIR / "runs" / "double_dqn_lstm" / "models" / "best_model.pt"
# 20 questions per episode - matches run_real_student_session.py's --n-questions
# default exactly, so the web app and the CLI script behave identically.
N_QUESTIONS = int(os.environ.get("EPISODE_LENGTH", "20"))
USE_LLM = os.environ.get("USE_LLM", "1") != "0"
# New-student starting ability: v4(grok) already rebased this to 26-34 on the
# 10-50 scale ("30 = average") - reuse that convention rather than June-3's
# old 15-20 (which was calibrated for the old 10-30 scale).
NEW_STUDENT_ABILITY_LOW = 26
NEW_STUDENT_ABILITY_HIGH = 34


def default_database_path() -> Path:
    configured = os.environ.get("DATABASE_PATH")
    if configured:
        return Path(configured).expanduser()
    persistent_dir = Path("/var/data")
    if os.environ.get("RENDER") and persistent_dir.exists():
        return persistent_dir / "student_sessions.db"
    return BASE_DIR / "data" / "student_sessions.db"


DATABASE_PATH = default_database_path()

# ---------------------------------------------------------------------------
# Achievement ladder (de-branded, rescaled to the 10-50 ability range)
# ---------------------------------------------------------------------------
ACHIEVEMENT_LEVELS = [
    {"level": "A", "requirement": "Finish an episode with effective ability >= 42.",
     "result": "Great job! You achieved A."},
    {"level": "A+", "requirement": "Finish an episode with effective ability >= 44.",
     "result": "Excellent! You achieved A+."},
    {"level": "Golden A+", "requirement": "Finish an episode with effective ability >= 46.",
     "result": "Outstanding! You achieved Golden A+."},
    {"level": "Top-tier interview", "requirement": "Finish an episode with effective ability >= 48.",
     "result": "Selected for a top-tier engineering interview."},
    {"level": "Hired", "requirement": "After interview selection, play again and finish at ability 50.",
     "result": "Hired as a Mechatronics Engineer."},
    {"level": "Senior engineer", "requirement": "After getting hired, play again and finish at ability 50.",
     "result": "Promoted to Senior Mechatronics Engineer."},
    {"level": "Head of department", "requirement": "After promotion, play again, finish at ability 50, and answer every question correctly.",
     "result": "Head of the Mechatronics Engineering Department."},
]

ACHIEVEMENT_RANK = {
    "": 0,
    "A": 1,
    "A+": 2,
    "Golden A+": 3,
    "Selected for interview": 4,
    "Hired as Mechatronics Engineer": 5,
    "Senior Mechatronics Engineer": 6,
    "Head of Mechatronics Engineering Department": 7,
}

ACHIEVEMENT_MESSAGES = {
    "A": "Great job! You achieved A.",
    "A+": "Excellent! You achieved A+.",
    "Golden A+": "Outstanding! You achieved Golden A+.",
    "Selected for interview": "You are selected for a top-tier engineering interview. Play again to get hired as a Mechatronics Engineer.",
    "Hired as Mechatronics Engineer": "Congratulations! You are hired as a Mechatronics Engineer. Play again for a promotion.",
    "Senior Mechatronics Engineer": "Congratulations! You are promoted to Senior Mechatronics Engineer. Play again to become Head of Department.",
    "Head of Mechatronics Engineering Department": "Congratulations! You achieved Head of the Mechatronics Engineering Department.",
}

ACHIEVEMENT_CELEBRATION = {
    "Hired as Mechatronics Engineer": "clap",
    "Senior Mechatronics Engineer": "clap_cheer",
    "Head of Mechatronics Engineering Department": "clap_cheer",
}

NEXT_GOAL_MESSAGES = {
    "": "Reach ability 42 to achieve A.",
    "A": "Reach ability 44 to achieve A+.",
    "A+": "Reach ability 46 to achieve Golden A+.",
    "Golden A+": "Reach ability 48 to get selected for a top-tier interview.",
    "Selected for interview": "Play again and finish at ability 50 to get hired as a Mechatronics Engineer.",
    "Hired as Mechatronics Engineer": "Play again and finish at ability 50 to get promoted.",
    "Senior Mechatronics Engineer": "Play again, finish at ability 50, and answer every question correctly to become Head of Department.",
    "Head of Mechatronics Engineering Department": "You reached the highest level in this simulator.",
}


def determine_achievement(*, current_status: str | None, final_effective_ability: float,
                          total_correct: int, total_wrong: int, questions_answered: int) -> str:
    current_status = current_status or ""
    current_rank = ACHIEVEMENT_RANK.get(current_status, 0)
    finished_at_42 = final_effective_ability >= 42.0
    finished_at_44 = final_effective_ability >= 44.0
    finished_at_46 = final_effective_ability >= 46.0
    finished_at_48 = final_effective_ability >= 48.0
    finished_at_max = final_effective_ability >= (qb.MAX_ABILITY - 0.0001)
    all_correct = questions_answered > 0 and total_wrong == 0 and total_correct == questions_answered

    if current_rank < ACHIEVEMENT_RANK["Selected for interview"]:
        if finished_at_48:
            return "Selected for interview"
        if finished_at_46:
            return "Golden A+"
        if finished_at_44:
            return "A+"
        if finished_at_42:
            return "A"

    if (ACHIEVEMENT_RANK["Selected for interview"] <= current_rank
            < ACHIEVEMENT_RANK["Hired as Mechatronics Engineer"] and finished_at_max):
        return "Hired as Mechatronics Engineer"

    if (ACHIEVEMENT_RANK["Hired as Mechatronics Engineer"] <= current_rank
            < ACHIEVEMENT_RANK["Senior Mechatronics Engineer"] and finished_at_max):
        return "Senior Mechatronics Engineer"

    if (ACHIEVEMENT_RANK["Senior Mechatronics Engineer"] <= current_rank
            < ACHIEVEMENT_RANK["Head of Mechatronics Engineering Department"]
            and finished_at_max and all_correct):
        return "Head of Mechatronics Engineering Department"

    return ""


def better_achievement(existing: str | None, new: str) -> str:
    current = existing or ""
    if ACHIEVEMENT_RANK.get(new, 0) > ACHIEVEMENT_RANK.get(current, 0):
        return new
    return current


def achievement_info(achievement: str | None) -> dict[str, str]:
    achievement = achievement or ""
    return {
        "title": achievement,
        "message": ACHIEVEMENT_MESSAGES.get(achievement, ""),
        "celebration": ACHIEVEMENT_CELEBRATION.get(achievement, ""),
        "next_goal": NEXT_GOAL_MESSAGES.get(achievement, NEXT_GOAL_MESSAGES[""]),
    }


# ---------------------------------------------------------------------------
# Flask app + in-memory episode state
# ---------------------------------------------------------------------------
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "mechatronics-app-dev-secret")

ACTIVE_EPISODES: dict[str, dict[str, Any]] = {}
ENGINE: "ModelEngine | None" = None


@dataclass
class ModelEngine:
    algo: str
    model: Any
    device: Any
    model_path: Path


def resolve_model_path() -> Path:
    configured = os.environ.get("MODEL_PATH")
    if configured:
        return Path(configured).expanduser()
    return DEFAULT_MODEL_PATH


def model_status() -> dict[str, Any]:
    path = resolve_model_path()
    return {
        "path": str(path),
        "exists": path.exists(),
        "loaded": ENGINE is not None,
        "device": os.environ.get("MODEL_DEVICE", "cpu"),
        "algo": ENGINE.algo if ENGINE is not None else None,
    }


def database_status() -> dict[str, Any]:
    return {"path": str(DATABASE_PATH), "exists": DATABASE_PATH.exists()}


def get_engine() -> ModelEngine:
    """Lazily load the trained model (any of the 5 supported formats)."""
    global ENGINE
    if ENGINE is not None:
        return ENGINE

    model_path = resolve_model_path()
    if not model_path.exists():
        raise RuntimeError(
            f"Model file was not found at {model_path}. Set MODEL_PATH to point at a "
            "trained checkpoint (.zip for RecurrentPPO, .pt for the others)."
        )

    device = rc.get_device(os.environ.get("MODEL_DEVICE", "auto"))
    if model_path.suffix == ".zip":
        from sb3_contrib import RecurrentPPO
        model = RecurrentPPO.load(str(model_path), device=device)
        algo = "recurrent_ppo"
    else:
        ckpt = rc.load_model(model_path, device)
        algo = ckpt["algo"]
        model = rc.build_model_from_checkpoint(ckpt, device)

    ENGINE = ModelEngine(algo=algo, model=model, device=device, model_path=model_path)
    return ENGINE


def engine_choose_action(engine: ModelEngine, env: TopicSelectionMCQEnv, obs: np.ndarray,
                         mask: np.ndarray, hidden_state):
    """Returns (chapter_slot, new_hidden_state). hidden_state is None on the
    first decision of an episode, or whatever this function returned last
    time for this episode otherwise. Memoryless algos always return None."""
    if engine.algo == "recurrent_ppo":
        action, new_state = engine.model.predict(
            np.asarray(obs, dtype=np.float32),
            state=hidden_state,
            episode_start=np.array([hidden_state is None]),
            deterministic=False,
        )
        return int(action), new_state
    if engine.algo == "a2c_lstm":
        return rc.greedy_topic_from_recurrent_actor(engine.model, obs, mask, hidden_state, engine.device)
    if engine.algo == "double_dqn_lstm":
        return rc.greedy_topic_from_recurrent_q(engine.model, obs, mask, hidden_state, engine.device)
    return rc.greedy_topic(engine.model, engine.algo, obs, mask, engine.device), None


# ---------------------------------------------------------------------------
# SQLite persistence
# ---------------------------------------------------------------------------
def get_db() -> sqlite3.Connection:
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout = 5000")
    return conn


def init_db() -> None:
    with get_db() as db:
        db.executescript(
            """
            CREATE TABLE IF NOT EXISTS students (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL COLLATE NOCASE UNIQUE,
                student_code TEXT UNIQUE,
                effective_ability REAL NOT NULL,
                highest_achievement TEXT DEFAULT '',
                asked_question_ids_json TEXT DEFAULT '[]',
                topic_stats_json TEXT DEFAULT '{}',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS episodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id INTEGER NOT NULL,
                session_id TEXT NOT NULL,
                started_at TEXT NOT NULL,
                completed_at TEXT NOT NULL,
                initial_effective_ability REAL NOT NULL,
                final_effective_ability REAL NOT NULL,
                questions_answered INTEGER NOT NULL,
                total_correct INTEGER NOT NULL,
                total_wrong INTEGER NOT NULL,
                accuracy REAL NOT NULL,
                total_time_taken REAL NOT NULL,
                achievement TEXT DEFAULT '',
                algo TEXT DEFAULT '',
                suggested_learning_path TEXT DEFAULT '',
                topic_performance TEXT DEFAULT '[]',
                rows_json TEXT DEFAULT '[]',
                FOREIGN KEY (student_id) REFERENCES students(id)
            );

            CREATE TABLE IF NOT EXISTS episode_answers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                episode_id INTEGER NOT NULL,
                step INTEGER NOT NULL,
                question_id TEXT NOT NULL,
                topic TEXT,
                subtopic TEXT,
                question TEXT NOT NULL,
                options_json TEXT NOT NULL,
                chosen_option TEXT NOT NULL,
                chosen_answer_text TEXT,
                correct_answer TEXT NOT NULL,
                correct_answer_text TEXT,
                is_correct INTEGER NOT NULL,
                time_taken REAL NOT NULL,
                time_ratio REAL,
                effective_ability_before REAL,
                effective_ability_after REAL,
                explanation TEXT,
                source TEXT,
                FOREIGN KEY (episode_id) REFERENCES episodes(id)
            );
            """
        )


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().tolist()
    return value


def normalize_name(name: str) -> str:
    return " ".join(name.strip().split())


def student_from_row(row: sqlite3.Row | None) -> dict[str, Any] | None:
    return dict(row) if row is not None else None


def get_student(student_id: int) -> dict[str, Any] | None:
    with get_db() as db:
        row = db.execute("SELECT * FROM students WHERE id = ?", (student_id,)).fetchone()
    return student_from_row(row)


def current_student() -> dict[str, Any] | None:
    raw_id = session.get("student_id")
    if raw_id is None:
        return None
    student = get_student(int(raw_id))
    if student is None:
        session.clear()
    return student


def create_account(name: str) -> dict[str, Any]:
    clean_name = normalize_name(name)
    if not clean_name:
        raise ValueError("Please enter a name.")

    rng = np.random.default_rng()
    initial_ability = float(rng.integers(NEW_STUDENT_ABILITY_LOW, NEW_STUDENT_ABILITY_HIGH + 1))
    created_at = now_iso()

    try:
        with get_db() as db:
            cursor = db.execute(
                "INSERT INTO students (name, effective_ability, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (clean_name, initial_ability, created_at, created_at),
            )
            student_id = int(cursor.lastrowid)
            student_code = f"{student_id:05d}"
            db.execute("UPDATE students SET student_code = ? WHERE id = ?", (student_code, student_id))
    except sqlite3.IntegrityError as exc:
        raise ValueError("This name is already registered. Try a different name.") from exc

    student = get_student(student_id)
    if student is None:
        raise RuntimeError("Account was created but could not be loaded.")
    return student


def find_student_by_credentials(name: str, student_code: str) -> dict[str, Any] | None:
    clean_name = normalize_name(name)
    clean_code = student_code.strip()
    with get_db() as db:
        row = db.execute(
            "SELECT * FROM students WHERE name = ? COLLATE NOCASE AND student_code = ?",
            (clean_name, clean_code),
        ).fetchone()
    return student_from_row(row)


def login_student(student: dict[str, Any]) -> None:
    session.clear()
    session["student_id"] = int(student["id"])


# ---------------------------------------------------------------------------
# Episode lifecycle
# ---------------------------------------------------------------------------
def make_env_for_student(student: dict[str, Any], seed: int) -> TopicSelectionMCQEnv:
    env = make_env(sub_episode_length=N_QUESTIONS, n_sub_episodes=1, seed=seed,
                   randomize_initial_ability=False)
    env.reset(seed=seed)
    ability = float(sim.clip(float(student["effective_ability"]), qb.MIN_ABILITY, qb.MAX_ABILITY))
    env.effective_ability = ability
    env.initial_effective_ability = ability
    env.target_difficulty = dc.initial_target_difficulty(ability)

    topic_stats = json.loads(student.get("topic_stats_json") or "{}")
    for topic, st in topic_stats.items():
        ti = env.topic_to_idx.get(topic)
        if ti is None:
            continue
        env.topic_asked[ti] = float(st.get("asked", 0))
        env.topic_correct[ti] = float(st.get("correct", 0))
        env.topic_wrong[ti] = float(st.get("wrong", 0))
    return env


def question_payload(question: dict[str, Any]) -> dict[str, Any]:
    """Sent to the template - deliberately excludes question_id and difficulty."""
    return {
        "topic": question["topic"],
        "question": question["question"],
        "options": [
            {"key": opt, "text": question.get(f"option_{opt}", "")}
            for opt in qb.OPTIONS
        ],
    }


def select_next_question(episode: dict[str, Any]) -> None:
    engine = get_engine()
    env = episode["env"]
    obs = env._get_obs()
    mask = rc.valid_topic_mask(env)
    hidden_before = episode["hidden_state"]

    slot, new_hidden = engine_choose_action(engine, env, obs, mask, hidden_before)
    topic_idx = int(env.active_idx[slot])
    topic = env.topics[topic_idx]
    target_diff = float(env.target_difficulty)

    asked_combined = episode["asked_ids_session"] | episode["asked_ids_history"]
    retrieve_fn = (lambda t: retriever.retrieve_for_topic(t, k=6)) if USE_LLM else (lambda t: [])
    q_global, question, source, was_repeat, gap_ok = select_question_for_topic(
        topic=topic, topic_idx=topic_idx, target_diff=target_diff,
        asked_ids=asked_combined, env=env, use_llm=USE_LLM,
        retrieve_fn=retrieve_fn, generate_fn=question_generator.generate_mcq,
    )

    episode["obs_at_selection"] = obs
    episode["mask_at_selection"] = mask
    episode["hidden_state_before"] = hidden_before
    episode["hidden_state"] = new_hidden
    episode["selected_slot"] = slot
    episode["selected_topic_idx"] = topic_idx
    episode["current_question"] = question
    episode["q_global"] = q_global
    episode["question_source"] = source
    episode["target_diff"] = target_diff
    episode["question_started_at"] = time.perf_counter()
    episode["xai_cache"] = None


def create_episode_state(student: dict[str, Any]) -> str:
    seed = secrets.randbits(32)
    env = make_env_for_student(student, seed)

    asked_history = set(json.loads(student.get("asked_question_ids_json") or "[]"))

    token = secrets.token_urlsafe(24)
    episode: dict[str, Any] = {
        "token": token,
        "student_db_id": int(student["id"]),
        "student_code": student["student_code"],
        "student_name": student["name"],
        "session_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "started_at": now_iso(),
        "started_at_epoch": time.time(),
        "initial_effective_ability": env.initial_effective_ability,
        "env": env,
        "hidden_state": None,
        "n_questions": N_QUESTIONS,
        "questions_answered": 0,
        "total_correct": 0,
        "total_wrong": 0,
        "total_time_taken": 0.0,
        "asked_ids_session": set(),
        "asked_ids_history": asked_history,
        "rows": [],
        "awaiting_next": False,
        "feedback": None,
    }
    select_next_question(episode)
    ACTIVE_EPISODES[token] = episode
    return token


def get_active_episode(token: str, student: dict[str, Any]) -> dict[str, Any] | None:
    episode = ACTIVE_EPISODES.get(token)
    if episode is None:
        return None
    if int(episode["student_db_id"]) != int(student["id"]):
        return None
    return episode


def parse_answer_time(episode: dict[str, Any]) -> float:
    server_elapsed = time.perf_counter() - float(episode["question_started_at"])
    raw_elapsed_ms = request.form.get("elapsed_ms", "")
    try:
        client_elapsed = float(raw_elapsed_ms) / 1000.0
    except ValueError:
        client_elapsed = server_elapsed
    elapsed = client_elapsed if client_elapsed > 0 else server_elapsed
    return round(float(qb.clip(elapsed, 0.2, 600.0)), 2)


def apply_answer(episode: dict[str, Any], chosen_option: str, time_taken: float) -> tuple[dict[str, Any], bool]:
    env = episode["env"]
    question = episode["current_question"]
    correct_answer = str(question["answer"]).strip().upper()
    is_correct = chosen_option.strip().upper() == correct_answer

    info = env.apply_external_answer(
        episode["selected_topic_idx"], episode["q_global"], is_correct, time_taken,
    )

    if is_correct:
        episode["total_correct"] += 1
    else:
        episode["total_wrong"] += 1
    episode["total_time_taken"] += float(time_taken)

    chunks = retriever.retrieve_for_topic(question["topic"], k=5) if USE_LLM else None
    explanation = explain_answer.generate_explanation(question, chosen_option, is_correct, chunks=chunks)

    qid = question.get("question_id") or f"LLM_{episode['session_id']}_{episode['questions_answered'] + 1}"
    episode["asked_ids_session"].add(qid)

    chosen_text = question.get(f"option_{chosen_option}", "")
    correct_text = question.get(f"option_{correct_answer}", "")

    row = {
        "step": episode["questions_answered"] + 1,
        "question_id": qid,
        "topic": question["topic"],
        "subtopic": question.get("subtopic", ""),
        "question": question["question"],
        "options": [{"key": o, "text": question.get(f"option_{o}", "")} for o in qb.OPTIONS],
        "chosen_option": chosen_option,
        "chosen_answer_text": chosen_text,
        "correct_answer": correct_answer,
        "correct_answer_text": correct_text,
        "is_correct": is_correct,
        "time_taken": round(float(time_taken), 2),
        "time_ratio": info["time_ratio"],
        "effective_ability_before": info["effective_ability_before"],
        "effective_ability_after": info["effective_ability_after"],
        "explanation": explanation,
        "source": episode["question_source"],
    }
    episode["rows"].append(row)
    episode["questions_answered"] += 1

    feedback = {
        "is_correct": is_correct,
        "chosen_option": chosen_option,
        "chosen_answer_text": chosen_text,
        "correct_answer": correct_answer,
        "correct_answer_text": correct_text,
        "time_taken": row["time_taken"],
        "time_ratio": info["time_ratio"],
        "topic": question["topic"],
        "effective_student_ability": info["effective_ability_after"],
        "explanation": explanation,
    }
    terminated = episode["questions_answered"] >= episode["n_questions"]
    return feedback, terminated


def topic_performance_rows(env: TopicSelectionMCQEnv) -> list[dict[str, Any]]:
    report = env.get_topic_report()
    rows = []
    for topic in (env.topics[i] for i in env.active_idx):
        data = report.get(topic, {})
        if data.get("attempts", 0):
            rows.append({"topic": topic, **data})
    return rows


def save_completed_episode(*, student: dict[str, Any], episode: dict[str, Any],
                           final_ability: float) -> int:
    env = episode["env"]
    rows = episode["rows"]
    questions_answered = len(rows)
    total_correct = episode["total_correct"]
    total_wrong = episode["total_wrong"]
    accuracy = total_correct / questions_answered if questions_answered else 0.0
    total_time_taken = round(episode["total_time_taken"], 2)
    topic_performance = topic_performance_rows(env)

    narrative = ""
    if USE_LLM:
        try:
            ordered_topics = env.suggest_learning_path()
            narrative = lp.generate_learning_path(
                ordered_topics=ordered_topics,
                topic_report=env.get_topic_report(),
                student_id=episode["student_code"],
                session_dir=None,
            )
        except Exception:
            narrative = ""
    if not narrative:
        narrative = "Suggested study order:\n" + "\n".join(
            f"  {i + 1}. {t}" for i, t in enumerate(env.suggest_learning_path()[:10])
        )

    achievement = determine_achievement(
        current_status=student.get("highest_achievement"),
        final_effective_ability=final_ability,
        total_correct=total_correct, total_wrong=total_wrong,
        questions_answered=questions_answered,
    )
    highest_achievement = better_achievement(student.get("highest_achievement"), achievement)

    asked_history = set(json.loads(student.get("asked_question_ids_json") or "[]"))
    asked_history |= episode["asked_ids_session"]

    topic_stats = json.loads(student.get("topic_stats_json") or "{}")
    for topic in (env.topics[i] for i in env.active_idx):
        data = env.get_topic_report().get(topic, {})
        if data.get("attempts", 0):
            topic_stats[topic] = {
                "asked": int(data.get("attempts", 0)),
                "correct": int(data.get("correct", 0)),
                "wrong": int(data.get("wrong", 0)),
            }

    with get_db() as db:
        cursor = db.execute(
            """
            INSERT INTO episodes (
                student_id, session_id, started_at, completed_at,
                initial_effective_ability, final_effective_ability,
                questions_answered, total_correct, total_wrong, accuracy,
                total_time_taken, achievement, algo, suggested_learning_path,
                topic_performance, rows_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                int(student["id"]), episode["session_id"], episode["started_at"], now_iso(),
                float(episode["initial_effective_ability"]), round(final_ability, 4),
                questions_answered, total_correct, total_wrong, round(accuracy, 4),
                total_time_taken, achievement, get_engine().algo, narrative,
                json.dumps(json_safe(topic_performance), ensure_ascii=False),
                json.dumps(json_safe(rows), ensure_ascii=False),
            ),
        )
        episode_id = int(cursor.lastrowid)
        for row in rows:
            db.execute(
                """
                INSERT INTO episode_answers (
                    episode_id, step, question_id, topic, subtopic, question,
                    options_json, chosen_option, chosen_answer_text,
                    correct_answer, correct_answer_text, is_correct,
                    time_taken, time_ratio, effective_ability_before,
                    effective_ability_after, explanation, source
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    episode_id, row["step"], row["question_id"], row["topic"], row["subtopic"],
                    row["question"], json.dumps(row["options"], ensure_ascii=False),
                    row["chosen_option"], row["chosen_answer_text"], row["correct_answer"],
                    row["correct_answer_text"], 1 if row["is_correct"] else 0,
                    row["time_taken"], row["time_ratio"], row["effective_ability_before"],
                    row["effective_ability_after"], row["explanation"], row["source"],
                ),
            )
        db.execute(
            """
            UPDATE students
            SET effective_ability = ?, highest_achievement = ?,
                asked_question_ids_json = ?, topic_stats_json = ?, updated_at = ?
            WHERE id = ?
            """,
            (
                round(final_ability, 4), highest_achievement,
                json.dumps(sorted(asked_history)), json.dumps(topic_stats),
                now_iso(), int(student["id"]),
            ),
        )

    return episode_id


def recent_episodes(student_id: int, limit: int = 5) -> list[dict[str, Any]]:
    with get_db() as db:
        rows = db.execute(
            """
            SELECT id, completed_at, initial_effective_ability, final_effective_ability,
                   questions_answered, total_correct, total_wrong, accuracy, achievement
            FROM episodes WHERE student_id = ? ORDER BY id DESC LIMIT ?
            """,
            (student_id, limit),
        ).fetchall()
    return [dict(row) for row in rows]


def load_episode_result(episode_id: int, student_id: int):
    with get_db() as db:
        episode = db.execute(
            "SELECT * FROM episodes WHERE id = ? AND student_id = ?", (episode_id, student_id),
        ).fetchone()
        if episode is None:
            return None
        answers = db.execute(
            "SELECT * FROM episode_answers WHERE episode_id = ? ORDER BY step ASC", (episode_id,),
        ).fetchall()

    episode_dict = dict(episode)
    episode_dict["topic_performance"] = json.loads(episode_dict.get("topic_performance") or "[]")
    answer_dicts = [dict(row) for row in answers]
    for answer in answer_dicts:
        answer["options"] = json.loads(answer.get("options_json") or "[]")
    return episode_dict, answer_dicts


def render_episode_page(*, student: dict[str, Any], episode: dict[str, Any],
                        feedback: dict[str, Any] | None = None) -> str:
    env = episode["env"]
    return render_template(
        "play.html",
        student=student,
        episode={
            "token": episode["token"],
            "question_number": episode["questions_answered"] if feedback is not None else episode["questions_answered"] + 1,
            "episode_length": episode["n_questions"],
            "initial_effective_ability": episode["initial_effective_ability"],
            "current_effective_ability": round(float(env.effective_ability), 4),
            "elapsed_seconds": max(0.0, time.time() - float(episode.get("started_at_epoch", time.time()))),
            "algo": get_engine().algo,
        },
        question=question_payload(episode["current_question"]),
        feedback=feedback,
        achievement_levels=ACHIEVEMENT_LEVELS,
        current_status_info=achievement_info(student.get("highest_achievement")),
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/")
def index() -> str:
    student = current_student()
    return render_template(
        "index.html",
        student=student,
        recent_episodes=recent_episodes(int(student["id"])) if student else [],
        model_status=model_status(),
        database_status=database_status(),
        achievement_levels=ACHIEVEMENT_LEVELS,
        current_status_info=achievement_info(student.get("highest_achievement") if student else ""),
        ability_min=qb.MIN_ABILITY,
        ability_max=qb.MAX_ABILITY,
    )


@app.post("/signup")
def signup() -> Any:
    try:
        student = create_account(request.form.get("name", ""))
    except ValueError as exc:
        flash(str(exc), "error")
        return redirect(url_for("index"))
    login_student(student)
    flash(f"Account created. Your student ID is {student['student_code']}. Save it for sign in.", "success")
    return redirect(url_for("index"))


@app.post("/signin")
def signin() -> Any:
    student = find_student_by_credentials(request.form.get("name", ""), request.form.get("student_code", ""))
    if student is None:
        flash("Name and student ID did not match. Check spelling and ID.", "error")
        return redirect(url_for("index"))
    login_student(student)
    flash("Signed in successfully.", "success")
    return redirect(url_for("index"))


@app.post("/logout")
def logout() -> Any:
    session.clear()
    flash("Signed out.", "success")
    return redirect(url_for("index"))


@app.post("/episode/start")
def start_episode() -> Any:
    student = current_student()
    if student is None:
        flash("Please sign in first.", "error")
        return redirect(url_for("index"))
    try:
        token = create_episode_state(student)
    except Exception as exc:
        flash(str(exc), "error")
        return redirect(url_for("index"))
    session["active_episode_token"] = token
    return redirect(url_for("play_episode", token=token))


@app.get("/episode/<token>")
def play_episode(token: str) -> Any:
    student = current_student()
    if student is None:
        flash("Please sign in first.", "error")
        return redirect(url_for("index"))
    episode = get_active_episode(token, student)
    if episode is None:
        flash("This episode expired. Start a new one.", "error")
        return redirect(url_for("index"))
    return render_episode_page(student=student, episode=episode, feedback=episode.get("feedback"))


@app.post("/episode/<token>/answer")
def answer_episode(token: str) -> Any:
    student = current_student()
    if student is None:
        flash("Please sign in first.", "error")
        return redirect(url_for("index"))
    episode = get_active_episode(token, student)
    if episode is None:
        flash("This episode expired. Start a new one.", "error")
        return redirect(url_for("index"))
    if episode.get("awaiting_next"):
        flash("Review the feedback, then press Next Question.", "error")
        return redirect(url_for("play_episode", token=token))

    chosen_option = request.form.get("chosen_option", "").strip().upper()
    if chosen_option not in qb.OPTIONS:
        flash("Choose option A, B, C, or D.", "error")
        return redirect(url_for("play_episode", token=token))

    time_taken = parse_answer_time(episode)
    feedback, terminated = apply_answer(episode, chosen_option, time_taken)

    if terminated:
        episode_id = save_completed_episode(
            student=student, episode=episode,
            final_ability=feedback["effective_student_ability"],
        )
        ACTIVE_EPISODES.pop(token, None)
        session.pop("active_episode_token", None)
        return redirect(url_for("episode_result", episode_id=episode_id))

    episode["feedback"] = feedback
    episode["awaiting_next"] = True
    return render_episode_page(student=student, episode=episode, feedback=feedback)


@app.post("/episode/<token>/next")
def next_question(token: str) -> Any:
    student = current_student()
    if student is None:
        flash("Please sign in first.", "error")
        return redirect(url_for("index"))
    episode = get_active_episode(token, student)
    if episode is None:
        flash("This episode expired. Start a new one.", "error")
        return redirect(url_for("index"))
    if not episode.get("awaiting_next"):
        return redirect(url_for("play_episode", token=token))

    episode["feedback"] = None
    episode["awaiting_next"] = False
    select_next_question(episode)
    return redirect(url_for("play_episode", token=token))


@app.get("/episode/<token>/xai")
def episode_xai(token: str) -> Any:
    student = current_student()
    if student is None:
        return "<p class='empty-state'>Please sign in first.</p>", 401
    episode = get_active_episode(token, student)
    if episode is None:
        return "<p class='empty-state'>This episode expired.</p>", 404

    if episode.get("xai_cache") is None:
        try:
            engine = get_engine()
            xai = explain_topic_choice(
                engine.model, engine.algo,
                episode["obs_at_selection"], episode["mask_at_selection"],
                episode["env"], engine.device,
                selected_slot=episode["selected_slot"],
                hidden_state=episode["hidden_state_before"],
                precomputed_target_diff=episode["target_diff"],
            )
            episode["xai_cache"] = json_safe(xai)
        except Exception as exc:
            return f"<p class='empty-state'>XAI unavailable: {exc}</p>", 200

    return render_template("_xai_panel.html", xai=episode["xai_cache"])


@app.get("/episode/result/<int:episode_id>")
def episode_result(episode_id: int) -> Any:
    student = current_student()
    if student is None:
        flash("Please sign in first.", "error")
        return redirect(url_for("index"))
    result = load_episode_result(episode_id, int(student["id"]))
    if result is None:
        flash("Episode result was not found.", "error")
        return redirect(url_for("index"))
    episode, answers = result
    refreshed_student = current_student() or student
    return render_template(
        "result.html",
        student=refreshed_student,
        episode=episode,
        answers=answers,
        achievement_info=achievement_info(episode.get("achievement")),
        current_status_info=achievement_info(refreshed_student.get("highest_achievement")),
        achievement_levels=ACHIEVEMENT_LEVELS,
    )


@app.get("/healthz")
def healthz() -> dict[str, Any]:
    return {"ok": True, "model": model_status(), "database": database_status()}


init_db()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5050"))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug)
