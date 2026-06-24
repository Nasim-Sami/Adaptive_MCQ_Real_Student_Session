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
from flask import Flask, flash, redirect, render_template, request, session, url_for

from dynamic_ability_env import DynamicAbilityAdaptiveMCQEnv, clip
from dynamic_student_simulator import (
    MAX_ABILITY,
    MIN_ABILITY,
    OPTIONS,
    QUESTIONS_WITH_METADATA,
    create_student,
)
from run_dynamic_real_student_session import (
    DEFAULT_INITIAL_ABILITY_HIGH,
    DEFAULT_INITIAL_ABILITY_LOW,
    apply_real_answer,
    build_model,
    initialize_real_student_state,
    load_checkpoint,
    select_pure_model_action,
    topic_performance_rows,
    valid_action_mask,
)
from train_dynamic_adaptive_mcq import get_device


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_THESIS_MODEL_PATH = Path(
    "/Users/sami/Documents/4-2/Thesis/Latest working/Adaptive_RL_English/"
    "june_1_building_own_env/runs/dynamic_dqn_random_guess_1m/models/final_model.pt"
)
DEFAULT_BUNDLED_MODEL_PATH = BASE_DIR / "models" / "final_model_slim.pt"
DATABASE_PATH = Path(
    os.environ.get("DATABASE_PATH", str(BASE_DIR / "data" / "student_sessions.db"))
).expanduser()

ACHIEVEMENT_NOTICE_LINES = [
    "Do you want to get A+ or Golden A+ !!",
    "How to Get A+ or Golden A+ ?",
    "First, play one episode and try to answer as many questions correctly as possible.",
    "As you answer correctly, your effective ability will increase.",
    "When your effective ability reaches 30, you become qualified for A+ or Golden A+.",
    "To achieve A+:",
    "Finish the next episode while keeping your effective ability at 30.",
    "To achieve Golden A+:",
    "Answer every question correctly in the next episode and finish while keeping your effective ability at 30.",
    "So, aim for correct answers, maintain your ability level, and try to reach the highest achievement!",
]

ACHIEVEMENT_RANK = {
    "": 0,
    "Qualified for A+ next episode": 1,
    "A+": 2,
    "Golden A+": 3,
}

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "student-simulator-dev-secret")

ACTIVE_EPISODES: dict[str, dict[str, Any]] = {}
ENGINE: "ModelEngine | None" = None


@dataclass
class ModelEngine:
    algo: str
    model: Any
    device: Any
    episode_length: int
    model_path: Path
    checkpoint_step: int | None

    def choose_action(self, env: DynamicAbilityAdaptiveMCQEnv, obs: np.ndarray) -> int:
        return select_pure_model_action(
            algo=self.algo,
            model=self.model,
            obs=obs,
            mask=valid_action_mask(env),
            device=self.device,
        )


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def active_model_path() -> Path:
    configured = os.environ.get("MODEL_PATH")
    if configured:
        return Path(configured).expanduser()
    if DEFAULT_BUNDLED_MODEL_PATH.exists():
        return DEFAULT_BUNDLED_MODEL_PATH
    return DEFAULT_THESIS_MODEL_PATH


def model_status() -> dict[str, Any]:
    path = active_model_path()
    return {
        "path": str(path),
        "exists": path.exists(),
        "loaded": ENGINE is not None,
        "device": os.environ.get("MODEL_DEVICE", "cpu"),
    }


def get_engine() -> ModelEngine:
    global ENGINE
    if ENGINE is not None:
        return ENGINE

    model_path = active_model_path()
    if not model_path.exists():
        raise RuntimeError(
            "Model file was not found. Put final_model_slim.pt in ./models or set MODEL_PATH in Render."
        )

    device = get_device(os.environ.get("MODEL_DEVICE", "cpu"))
    checkpoint = load_checkpoint(model_path, device)
    saved_algo = checkpoint.get("algo") or "dqn"
    algo = os.environ.get("MODEL_ALGO", saved_algo)
    if saved_algo and algo != saved_algo:
        raise RuntimeError(f"Checkpoint was saved for {saved_algo}, but MODEL_ALGO is {algo}.")

    saved_args = checkpoint.get("args", {}) or {}
    episode_length = int(os.environ.get("EPISODE_LENGTH", saved_args.get("episode_length", 15)))
    probe_env = make_env("model_probe", 20, episode_length, seed=1)
    model = build_model(
        algo=algo,
        checkpoint=checkpoint,
        obs_dim=probe_env.observation_space.shape[0],
        n_actions=probe_env.action_space.n,
        device=device,
    )

    ENGINE = ModelEngine(
        algo=algo,
        model=model,
        device=device,
        episode_length=episode_length,
        model_path=model_path,
        checkpoint_step=checkpoint.get("step"),
    )
    return ENGINE


def get_db() -> sqlite3.Connection:
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
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
                suggested_learning_path TEXT DEFAULT '[]',
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
                response_label TEXT,
                effective_ability_before REAL,
                effective_ability_after REAL,
                ability_delta REAL,
                explanation TEXT,
                FOREIGN KEY (episode_id) REFERENCES episodes(id)
            );
            """
        )


def normalize_name(name: str) -> str:
    return " ".join(name.strip().split())


def student_from_row(row: sqlite3.Row | None) -> dict[str, Any] | None:
    return dict(row) if row is not None else None


def get_student(student_id: int) -> dict[str, Any] | None:
    with get_db() as db:
        row = db.execute("SELECT * FROM students WHERE id = ?", (student_id,)).fetchone()
    return student_from_row(row)


def current_student() -> dict[str, Any] | None:
    raw_student_id = session.get("student_id")
    if raw_student_id is None:
        return None
    student = get_student(int(raw_student_id))
    if student is None:
        session.clear()
    return student


def create_account(name: str) -> dict[str, Any]:
    clean_name = normalize_name(name)
    if not clean_name:
        raise ValueError("Please enter a name.")

    rng = np.random.default_rng()
    initial_ability = int(
        rng.integers(DEFAULT_INITIAL_ABILITY_LOW, DEFAULT_INITIAL_ABILITY_HIGH + 1)
    )
    created_at = now_iso()

    try:
        with get_db() as db:
            cursor = db.execute(
                """
                INSERT INTO students (name, effective_ability, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                """,
                (clean_name, float(initial_ability), created_at, created_at),
            )
            student_id = int(cursor.lastrowid)
            student_code = f"{student_id:05d}"
            db.execute(
                "UPDATE students SET student_code = ? WHERE id = ?",
                (student_code, student_id),
            )
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
            """
            SELECT * FROM students
            WHERE name = ? COLLATE NOCASE AND student_code = ?
            """,
            (clean_name, clean_code),
        ).fetchone()
    return student_from_row(row)


def login_student(student: dict[str, Any]) -> None:
    session.clear()
    session["student_id"] = int(student["id"])


def make_env(
    student_code: str,
    initial_effective_ability: float,
    episode_length: int,
    seed: int | None,
) -> DynamicAbilityAdaptiveMCQEnv:
    ability = int(round(clip(float(initial_effective_ability), MIN_ABILITY, MAX_ABILITY)))
    env = DynamicAbilityAdaptiveMCQEnv(
        questions=QUESTIONS_WITH_METADATA,
        students=[create_student(student_code, ability=ability)],
        episode_length=episode_length,
        seed=seed,
        repair_invalid_action=True,
        random_first_question=False,
    )
    env.action_space.seed(seed)
    return env


def question_payload(question: dict[str, Any]) -> dict[str, Any]:
    return {
        "question_id": question["question_id"],
        "topic": question["topic"],
        "subtopic": question.get("subtopic", ""),
        "question": question["question"],
        "options": [
            {"key": option, "text": question.get(f"option_{option}", "")}
            for option in OPTIONS
        ],
    }


def current_question_payload(episode: dict[str, Any]) -> dict[str, Any]:
    env = episode["env"]
    return question_payload(env.questions[int(episode["selected_action"])])


def select_next_question(episode: dict[str, Any]) -> None:
    engine = get_engine()
    env = episode["env"]
    action = engine.choose_action(env, episode["obs"])
    episode["selected_action"] = int(action)
    episode["question_started_at"] = time.perf_counter()


def create_episode_state(student: dict[str, Any]) -> str:
    engine = get_engine()
    initial_effective_ability = float(student["effective_ability"])
    initial_for_env = int(round(clip(initial_effective_ability, MIN_ABILITY, MAX_ABILITY)))
    seed = secrets.randbits(32)
    env = make_env(
        student_code=student["student_code"],
        initial_effective_ability=initial_for_env,
        episode_length=engine.episode_length,
        seed=seed,
    )
    obs, reset_info = initialize_real_student_state(
        env=env,
        student_id=student["student_code"],
        initial_effective_ability=initial_for_env,
    )

    token = secrets.token_urlsafe(24)
    episode = {
        "token": token,
        "student_db_id": int(student["id"]),
        "student_code": student["student_code"],
        "student_name": student["name"],
        "session_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "started_at": now_iso(),
        "initial_effective_ability": float(reset_info["initial_effective_ability"]),
        "env": env,
        "obs": obs,
        "model_path": str(engine.model_path),
        "algo": engine.algo,
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
    return round(float(clip(elapsed, 0.2, 600.0)), 2)


def determine_achievement(
    *,
    initial_effective_ability: float,
    final_effective_ability: float,
    total_correct: int,
    total_wrong: int,
    questions_answered: int,
) -> str:
    started_at_30 = initial_effective_ability >= MAX_ABILITY
    finished_at_30 = final_effective_ability >= (MAX_ABILITY - 0.0001)
    all_correct = questions_answered > 0 and total_wrong == 0 and total_correct == questions_answered

    if started_at_30 and finished_at_30 and all_correct:
        return "Golden A+"
    if started_at_30 and finished_at_30:
        return "A+"
    if finished_at_30:
        return "Qualified for A+ next episode"
    return ""


def better_achievement(existing: str | None, new: str) -> str:
    current = existing or ""
    if ACHIEVEMENT_RANK.get(new, 0) > ACHIEVEMENT_RANK.get(current, 0):
        return new
    return current


def answer_options_json(row: dict[str, Any]) -> str:
    options = [
        {"key": option, "text": row.get(f"option_{option}", "")}
        for option in OPTIONS
    ]
    return json.dumps(options, ensure_ascii=False)


def save_completed_episode(
    *,
    student: dict[str, Any],
    episode: dict[str, Any],
    final_info: dict[str, Any],
) -> int:
    env = episode["env"]
    rows = [json_safe(row) for row in env.episode_rows]
    questions_answered = len(rows)
    total_correct = int(env.total_correct)
    total_wrong = int(env.total_wrong)
    final_effective_ability = float(
        final_info.get("effective_student_ability", episode["initial_effective_ability"])
    )
    accuracy = float(env._safe_accuracy())
    total_time_taken = round(float(env.total_time_taken), 2)
    learning_path = env.suggest_learning_path()
    topic_performance = topic_performance_rows(env)
    achievement = determine_achievement(
        initial_effective_ability=float(episode["initial_effective_ability"]),
        final_effective_ability=final_effective_ability,
        total_correct=total_correct,
        total_wrong=total_wrong,
        questions_answered=questions_answered,
    )
    highest_achievement = better_achievement(student.get("highest_achievement"), achievement)

    with get_db() as db:
        cursor = db.execute(
            """
            INSERT INTO episodes (
                student_id, session_id, started_at, completed_at,
                initial_effective_ability, final_effective_ability,
                questions_answered, total_correct, total_wrong, accuracy,
                total_time_taken, achievement, suggested_learning_path,
                topic_performance, rows_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                int(student["id"]),
                episode["session_id"],
                episode["started_at"],
                now_iso(),
                float(episode["initial_effective_ability"]),
                round(final_effective_ability, 4),
                questions_answered,
                total_correct,
                total_wrong,
                round(accuracy, 4),
                total_time_taken,
                achievement,
                json.dumps(json_safe(learning_path), ensure_ascii=False),
                json.dumps(json_safe(topic_performance), ensure_ascii=False),
                json.dumps(rows, ensure_ascii=False),
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
                    time_taken, time_ratio, response_label,
                    effective_ability_before, effective_ability_after,
                    ability_delta, explanation
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    episode_id,
                    int(row.get("step", 0)),
                    row.get("question_id", ""),
                    row.get("topic", ""),
                    row.get("subtopic", ""),
                    row.get("question", ""),
                    answer_options_json(row),
                    row.get("chosen_option", ""),
                    row.get("chosen_answer_text", ""),
                    row.get("correct_answer", ""),
                    row.get("correct_answer_text", ""),
                    1 if row.get("is_correct") else 0,
                    float(row.get("time_taken", 0.0)),
                    float(row.get("time_ratio", 0.0)),
                    row.get("response_label", ""),
                    float(row.get("effective_ability_before", 0.0)),
                    float(row.get("effective_student_ability", 0.0)),
                    float(row.get("ability_delta", 0.0)),
                    row.get("explanation", ""),
                ),
            )
        db.execute(
            """
            UPDATE students
            SET effective_ability = ?, highest_achievement = ?, updated_at = ?
            WHERE id = ?
            """,
            (
                round(final_effective_ability, 4),
                highest_achievement,
                now_iso(),
                int(student["id"]),
            ),
        )

    return episode_id


def recent_episodes(student_id: int, limit: int = 5) -> list[dict[str, Any]]:
    with get_db() as db:
        rows = db.execute(
            """
            SELECT id, completed_at, initial_effective_ability, final_effective_ability,
                   questions_answered, total_correct, total_wrong, accuracy, achievement
            FROM episodes
            WHERE student_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (student_id, limit),
        ).fetchall()
    return [dict(row) for row in rows]


def load_episode_result(episode_id: int, student_id: int) -> tuple[dict[str, Any], list[dict[str, Any]]] | None:
    with get_db() as db:
        episode = db.execute(
            "SELECT * FROM episodes WHERE id = ? AND student_id = ?",
            (episode_id, student_id),
        ).fetchone()
        if episode is None:
            return None
        answers = db.execute(
            """
            SELECT * FROM episode_answers
            WHERE episode_id = ?
            ORDER BY step ASC
            """,
            (episode_id,),
        ).fetchall()

    episode_dict = dict(episode)
    episode_dict["suggested_learning_path"] = json.loads(
        episode_dict.get("suggested_learning_path") or "[]"
    )
    episode_dict["topic_performance"] = json.loads(
        episode_dict.get("topic_performance") or "[]"
    )
    answer_dicts = [dict(row) for row in answers]
    for answer in answer_dicts:
        answer["options"] = json.loads(answer.get("options_json") or "[]")
    return episode_dict, answer_dicts


def render_episode_page(
    *,
    student: dict[str, Any],
    episode: dict[str, Any],
    feedback: dict[str, Any] | None = None,
) -> str:
    env = episode["env"]
    return render_template(
        "play.html",
        student=student,
        episode={
            "token": episode["token"],
            "question_number": int(env.current_step) + 1,
            "episode_length": int(env.episode_length),
            "initial_effective_ability": episode["initial_effective_ability"],
            "current_effective_ability": round(float(env.effective_ability), 4),
            "algo": episode["algo"],
            "model_path": episode["model_path"],
        },
        question=current_question_payload(episode),
        feedback=feedback,
        notice_lines=ACHIEVEMENT_NOTICE_LINES,
    )


@app.get("/")
def index() -> str:
    student = current_student()
    return render_template(
        "index.html",
        student=student,
        recent_episodes=recent_episodes(int(student["id"])) if student else [],
        model_status=model_status(),
        notice_lines=ACHIEVEMENT_NOTICE_LINES,
        ability_min=MIN_ABILITY,
        ability_max=MAX_ABILITY,
    )


@app.post("/signup")
def signup() -> Any:
    try:
        student = create_account(request.form.get("name", ""))
    except ValueError as exc:
        flash(str(exc), "error")
        return redirect(url_for("index"))

    login_student(student)
    flash(
        f"Account created. Your student ID is {student['student_code']}. Save it for sign in.",
        "success",
    )
    return redirect(url_for("index"))


@app.post("/signin")
def signin() -> Any:
    student = find_student_by_credentials(
        request.form.get("name", ""),
        request.form.get("student_code", ""),
    )
    if student is None:
        flash("Name and student ID did not match.", "error")
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

    return render_episode_page(student=student, episode=episode)


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

    chosen_option = request.form.get("chosen_option", "").strip().upper()
    if chosen_option not in OPTIONS:
        flash("Choose option A, B, C, or D.", "error")
        return redirect(url_for("play_episode", token=token))

    time_taken = parse_answer_time(episode)
    obs, _, terminated, info = apply_real_answer(
        env=episode["env"],
        session_id=episode["session_id"],
        selected_action=int(episode["selected_action"]),
        chosen_option=chosen_option,
        time_taken=time_taken,
    )
    episode["obs"] = obs
    feedback = json_safe(info)

    if terminated:
        episode_id = save_completed_episode(
            student=student,
            episode=episode,
            final_info=feedback,
        )
        ACTIVE_EPISODES.pop(token, None)
        session.pop("active_episode_token", None)
        return redirect(url_for("episode_result", episode_id=episode_id))

    select_next_question(episode)
    return render_episode_page(student=student, episode=episode, feedback=feedback)


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
        notice_lines=ACHIEVEMENT_NOTICE_LINES,
    )


@app.get("/healthz")
def healthz() -> dict[str, Any]:
    return {"ok": True, "model": model_status()}


init_db()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5001"))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug)
