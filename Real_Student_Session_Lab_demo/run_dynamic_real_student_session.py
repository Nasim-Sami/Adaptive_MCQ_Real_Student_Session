from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from dynamic_ability_env import DynamicAbilityAdaptiveMCQEnv, clip
from dynamic_student_simulator import (
    MAX_ABILITY,
    MIN_ABILITY,
    QUESTIONS_WITH_METADATA,
    create_student,
    update_dynamic_student_state,
)
from train_dynamic_adaptive_mcq import ActorCritic, QNetwork, get_device, tensor


OPTIONS = ["A", "B", "C", "D"]
DEFAULT_INITIAL_ABILITY_LOW = 17
DEFAULT_INITIAL_ABILITY_HIGH = 23

SESSION_FIELDS = [
    "session_id",
    "student_id",
    "step",
    "question_id",
    "topic",
    "subtopic",
    "selected_action",
    "selection_source",
    "chosen_option",
    "chosen_answer_text",
    "correct_answer",
    "correct_answer_text",
    "explanation",
    "is_correct",
    "time_taken",
    "time_ratio",
    "student_reward",
    "agent_reward",
    "response_label",
    "inherent_difficulty",
    "target_inherent_difficulty",
    "ideal_inherent_difficulty",
    "difficulty_gap",
    "difficulty_match_reward",
    "topic_reward",
    "diversity_reward",
    "invalid_action_penalty",
    "initial_effective_ability",
    "fast_correct_pair_boost",
    "slow_wrong_pair_penalty",
    "raw_ability_delta",
    "effective_student_ability_10",
    "effective_ability_before",
    "effective_student_ability",
    "ability_delta",
    "accuracy",
    "total_correct",
    "total_wrong",
    "total_time_taken",
    "question",
    "option_A",
    "option_B",
    "option_C",
    "option_D",
]


def json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def valid_action_mask(env: DynamicAbilityAdaptiveMCQEnv) -> np.ndarray:
    mask = env.asked_mask < 0.5
    if not np.any(mask):
        mask = np.ones(env.n_questions, dtype=bool)
    return mask.astype(bool)


def masked_argmax(values: torch.Tensor, mask: np.ndarray) -> int:
    mask_t = torch.as_tensor(mask, dtype=torch.bool, device=values.device)
    masked_values = values.masked_fill(~mask_t, -1e9)
    return int(torch.argmax(masked_values).item())


def select_pure_model_action(
    *,
    algo: str,
    model: QNetwork | ActorCritic,
    obs: np.ndarray,
    mask: np.ndarray,
    device: torch.device,
) -> int:
    with torch.no_grad():
        if algo == "dqn":
            q_values = model(tensor(obs[None, :], device))[0]  # type: ignore[operator]
            return masked_argmax(q_values, mask)

        logits, _ = model(tensor(obs[None, :], device))  # type: ignore[operator]
        mask_t = torch.as_tensor(mask[None, :], dtype=torch.bool, device=device)
        masked_logits = logits.masked_fill(~mask_t, -1e9)
        return int(torch.argmax(masked_logits, dim=-1).item())


def build_model(
    *,
    algo: str,
    checkpoint: dict[str, Any],
    obs_dim: int,
    n_actions: int,
    device: torch.device,
) -> QNetwork | ActorCritic:
    saved_args = checkpoint.get("args", {}) or {}
    hidden_dim = int(saved_args.get("hidden_dim", 256))

    if algo == "dqn":
        model: QNetwork | ActorCritic = QNetwork(obs_dim, n_actions, hidden_dim)
    else:
        model = ActorCritic(obs_dim, n_actions, hidden_dim)

    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model


def format_numbered_list(items: list[str]) -> str:
    if not items:
        return "No learning path available."
    return "\n".join(f"{idx}. {item}" for idx, item in enumerate(items, start=1))


def rounded_or_none(value: Any, digits: int = 2) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def topic_performance_rows(env: DynamicAbilityAdaptiveMCQEnv) -> list[dict[str, Any]]:
    seen_topics: list[str] = []
    for row in env.episode_rows:
        topic = str(row.get("topic", ""))
        if topic and topic not in seen_topics:
            seen_topics.append(topic)

    rows = []
    for topic in seen_topics:
        topic_idx = env.topic_to_idx[topic]
        correct = int(env.topic_correct[topic_idx])
        wrong = int(env.topic_wrong[topic_idx])
        attempts = correct + wrong
        rows.append(
            {
                "topic": topic,
                "correct": correct,
                "wrong": wrong,
                "attempts": attempts,
                "display": f"{topic} ({correct}/{attempts})",
            }
        )
    return rows


def initialize_real_student_state(
    *,
    env: DynamicAbilityAdaptiveMCQEnv,
    student_id: str,
    initial_effective_ability: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Real students do not have a known hidden simulator profile.

    The first observation should therefore start from one guessed ability. The
    model receives that ability through the same observation features used in
    training, then each real answer/time update creates the next effective
    ability and next observation.
    """
    initial_effective_ability = int(
        round(clip(initial_effective_ability, MIN_ABILITY, MAX_ABILITY))
    )

    env._initialize_episode_state()
    env.current_student_id = student_id
    env.profile_starting_ability = float(initial_effective_ability)
    env.initial_ability = float(initial_effective_ability)
    env.initial_effective_ability = float(initial_effective_ability)
    env.student_profile = create_student(student_id, ability=initial_effective_ability)
    env.effective_ability = float(initial_effective_ability)
    env.previous_effective_ability = float(initial_effective_ability)
    env.last_observation_target_ability = float(initial_effective_ability)
    env.last_observation_score = 0.5
    env._apply_effective_ability_to_current_student()

    info = {
        "student_id": student_id,
        "initial_effective_ability": float(initial_effective_ability),
        "topics": env.topics,
    }
    info.update(env._ability_info())
    info.update(env._student_state_info())
    return env._get_obs(), info


def ask_answer(question: dict[str, Any], manual_time: bool) -> tuple[str | None, float]:
    print("\n" + "=" * 72)
    print(f'{question["question_id"]} | {question["topic"]} | {question.get("subtopic", "")}')
    print(question["question"])
    for option in OPTIONS:
        print(f"  {option}. {question[f'option_{option}']}")

    start = time.perf_counter()
    while True:
        answer = input("Your answer (A/B/C/D, or q to stop): ").strip().upper()
        if answer in OPTIONS:
            break
        if answer in {"Q", "QUIT", "STOP", "EXIT"}:
            return None, 0.0
        print("Please enter A, B, C, D, or q.")

    elapsed = time.perf_counter() - start
    if manual_time:
        while True:
            raw_time = input("Time taken in seconds: ").strip()
            try:
                elapsed = float(raw_time)
                if elapsed > 0:
                    break
            except ValueError:
                pass
            print("Please enter a positive number.")

    return answer, round(float(elapsed), 2)


def apply_real_answer(
    *,
    env: DynamicAbilityAdaptiveMCQEnv,
    session_id: str,
    selected_action: int,
    chosen_option: str,
    time_taken: float,
) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
    previous_obs = env._get_obs()
    question = env.questions[selected_action]
    topic = env.question_topics[selected_action]
    topic_idx = env.topic_to_idx[topic]
    effective_ability_before = env.effective_ability
    correct_answer = str(question["answer"]).upper()
    is_correct = chosen_option == correct_answer
    chosen_answer_text = str(question.get(f"option_{chosen_option}", ""))
    correct_answer_text = str(question.get(f"option_{correct_answer}", ""))
    explanation = str(question.get("explanation", "No explanation available."))

    result = {
        "is_correct": is_correct,
        "time_taken": float(time_taken),
        "chosen_option": chosen_option,
        "correct_answer": correct_answer,
    }

    student_state_before = env._student_state_info(env.current_student)
    student_reward, agent_reward, time_ratio, response_label = env._compute_rewards(
        result=result,
        question=question,
        topic_idx=topic_idx,
        invalid_action=False,
    )

    env._update_estimated_student_speed(result, question)

    if is_correct:
        env.topic_correct[topic_idx] += 1
        env.total_correct += 1
        env.consecutive_wrong_count = 0
    else:
        env.topic_wrong[topic_idx] += 1
        env.total_wrong += 1
        env.consecutive_wrong_count += 1

    if is_correct and time_ratio < 0.55:
        env.fast_correct_streak += 1
    else:
        env.fast_correct_streak = 0

    if response_label == "overload":
        env.overload_streak += 1
    else:
        env.overload_streak = 0

    env.asked_mask[selected_action] = 1.0
    env.previous_question_one_hot[:] = 0.0
    env.previous_question_one_hot[selected_action] = 1.0

    env.last_correct = float(is_correct)
    env.last_time_ratio = float(time_ratio)
    env.last_student_reward = float(student_reward)
    env.last_agent_reward = float(agent_reward)

    env.correct_history.append(float(is_correct))
    env.time_ratio_history.append(float(time_ratio))
    env.topic_history.append(topic)

    env.total_time_taken += float(time_taken)
    env.total_student_reward += float(student_reward)
    env.total_agent_reward += float(agent_reward)

    env.student_profile = update_dynamic_student_state(
        student=env.current_student,
        question=question,
        is_correct=is_correct,
        time_ratio=time_ratio,
    )

    base_next_obs = env._get_base_obs()
    ability_delta = env._compute_ability_delta(base_next_obs, is_correct, time_ratio)
    raw_ability_delta = ability_delta
    previous_row = env.episode_rows[-1] if env.episode_rows else None
    previous_fast_correct = bool(
        previous_row
        and previous_row.get("is_correct") is True
        and float(previous_row.get("time_ratio", 999.0)) < 0.65
    )
    current_fast_correct = bool(is_correct and time_ratio < 0.65)
    fast_correct_pair_boost = bool(
        previous_fast_correct
        and current_fast_correct
        and ability_delta > 0.0
    )
    if fast_correct_pair_boost:
        ability_delta *= 3.0

    previous_slow_wrong = bool(
        previous_row
        and previous_row.get("is_correct") is False
        and float(previous_row.get("time_ratio", 0.0)) > 1.25
    )
    current_slow_wrong = bool((not is_correct) and time_ratio > 1.25)
    slow_wrong_pair_penalty = bool(
        previous_slow_wrong
        and current_slow_wrong
        and ability_delta < 0.0
    )
    if slow_wrong_pair_penalty:
        ability_delta *= 2.5

    env.previous_effective_ability = env.effective_ability
    env.effective_ability = clip(
        env.effective_ability + ability_delta,
        env.min_effective_ability,
        env.max_effective_ability,
    )
    env.last_ability_delta = env.effective_ability - env.previous_effective_ability
    env._apply_effective_ability_to_current_student()

    ability_info = env._ability_info()
    student_state_after = env._student_state_info()

    row = {
        "session_id": session_id,
        "student_id": env.current_student_id,
        "step": env.current_step + 1,
        "question_id": question["question_id"],
        "topic": topic,
        "subtopic": question.get("subtopic", ""),
        "selected_action": selected_action,
        "selection_source": "pure_model_masked_greedy",
        "chosen_option": chosen_option,
        "chosen_answer_text": chosen_answer_text,
        "correct_answer": correct_answer,
        "correct_answer_text": correct_answer_text,
        "explanation": explanation,
        "is_correct": is_correct,
        "time_taken": round(float(time_taken), 2),
        "time_ratio": round(float(time_ratio), 4),
        "student_reward": round(float(student_reward), 4),
        "agent_reward": round(float(agent_reward), 4),
        "response_label": response_label,
        "inherent_difficulty": env.last_inherent_difficulty,
        "target_inherent_difficulty": env.last_target_inherent_difficulty,
        "ideal_inherent_difficulty": env.last_ideal_inherent_difficulty,
        "difficulty_gap": env.last_difficulty_gap,
        "difficulty_match_reward": env.last_difficulty_match_reward,
        "topic_reward": env.last_topic_reward,
        "diversity_reward": env.last_diversity_reward,
        "invalid_action_penalty": env.last_invalid_action_penalty,
        "effective_ability_before": round(float(effective_ability_before), 4),
        "fast_correct_pair_boost": fast_correct_pair_boost,
        "slow_wrong_pair_penalty": slow_wrong_pair_penalty,
        "raw_ability_delta": round(float(raw_ability_delta), 4),
        "accuracy": round(env._safe_accuracy(), 4),
        "total_correct": int(env.total_correct),
        "total_wrong": int(env.total_wrong),
        "total_time_taken": round(float(env.total_time_taken), 2),
        "question": question["question"],
        "option_A": question["option_A"],
        "option_B": question["option_B"],
        "option_C": question["option_C"],
        "option_D": question["option_D"],
        **student_state_before,
        **student_state_after,
        **ability_info,
    }
    env.episode_rows.append(row)
    env.current_step += 1

    terminated = bool(
        env.current_step >= env.episode_length
        or env.asked_mask.sum() >= env.n_questions
    )
    next_obs = env._get_obs()

    info = {
        "previous_obs": previous_obs,
        "next_obs": next_obs,
        "episode_rows": env.episode_rows,
        "suggested_learning_path": env.suggest_learning_path(),
        "total_agent_reward": env.total_agent_reward,
        "total_student_reward": env.total_student_reward,
        "total_time_taken": env.total_time_taken,
        "accuracy": env._safe_accuracy(),
        **row,
    }
    return next_obs, float(agent_reward), terminated, info


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    extra_fields = sorted(
        {
            key
            for row in rows
            for key in row.keys()
            if key not in SESSION_FIELDS
        }
    )
    fields = SESSION_FIELDS + extra_fields
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a real student session using only the trained dynamic DQN/A2C model."
    )
    parser.add_argument("--algo", choices=["dqn", "a2c"], required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--student-id", default="real_student")
    parser.add_argument("--episode-length", type=int, default=15)
    parser.add_argument(
        "--initial-effective-ability",
        type=int,
        default=None,
        help="Optional fixed initial ability guess. Default randomly guesses from 17 to 23.",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--manual-time", action="store_true")
    parser.add_argument("--save-dir", type=Path, default=Path("real_student_sessions"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device(args.device)
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    rng = np.random.default_rng(args.seed)

    if args.initial_effective_ability is None:
        initial_effective_ability = int(
            rng.integers(DEFAULT_INITIAL_ABILITY_LOW, DEFAULT_INITIAL_ABILITY_HIGH + 1)
        )
    else:
        initial_effective_ability = int(
            clip(args.initial_effective_ability, MIN_ABILITY, MAX_ABILITY)
        )

    env = DynamicAbilityAdaptiveMCQEnv(
        questions=QUESTIONS_WITH_METADATA,
        students=[create_student(args.student_id, ability=initial_effective_ability)],
        episode_length=args.episode_length,
        seed=args.seed,
        repair_invalid_action=True,
        random_first_question=False,
    )
    obs, reset_info = initialize_real_student_state(
        env=env,
        student_id=args.student_id,
        initial_effective_ability=initial_effective_ability,
    )

    checkpoint = load_checkpoint(args.model_path, device)
    saved_algo = checkpoint.get("algo")
    if saved_algo is not None and saved_algo != args.algo:
        raise ValueError(
            f"Model was saved as algo={saved_algo!r}, but --algo {args.algo!r} was given."
        )

    model = build_model(
        algo=args.algo,
        checkpoint=checkpoint,
        obs_dim=env.observation_space.shape[0],
        n_actions=env.action_space.n,
        device=device,
    )

    print("Dynamic real student session")
    print(f"  selection: pure trained {args.algo.upper()} model, masked greedy")
    print("  no calibration, no epsilon, no top-k, no hybrid/rule scoring")
    print(f"  student_id: {args.student_id}")
    print(f"  initial_effective_ability: {reset_info.get('initial_effective_ability')}")
    print(f"  initial_ability_10: {reset_info.get('effective_student_ability_10')}")
    print(f"  model_path: {args.model_path}")

    rows: list[dict[str, Any]] = []
    final_info: dict[str, Any] = {}

    while env.current_step < env.episode_length and env.asked_mask.sum() < env.n_questions:
        mask = valid_action_mask(env)
        selected_action = select_pure_model_action(
            algo=args.algo,
            model=model,
            obs=obs,
            mask=mask,
            device=device,
        )
        question = env.questions[selected_action]
        chosen_option, time_taken = ask_answer(question, args.manual_time)
        if chosen_option is None:
            print("Session stopped by user.")
            break

        obs, _, terminated, info = apply_real_answer(
            env=env,
            session_id=session_id,
            selected_action=selected_action,
            chosen_option=chosen_option,
            time_taken=time_taken,
        )
        final_info = info
        rows.append(env.episode_rows[-1])

        print(
            f"Result: {'correct' if info['is_correct'] else 'wrong'} | "
            f"your answer: {info['chosen_option']} | "
            f"time_taken: {info['time_taken']:.2f}s | "
            f"time_ratio: {info['time_ratio']:.3f} | "
            f"label: {info['response_label']} | "
            f"effective_ability: {info['effective_student_ability']}"
        )
        print(f"Right answer: {info['correct_answer']}. {info['correct_answer_text']}")
        print(f"Explanation: {info['explanation']}")

        if terminated:
            break

    session_dir = args.save_dir / f"{session_id}_{args.student_id}_{args.algo}"
    csv_path = session_dir / "session_results.csv"
    summary_path = session_dir / "session_summary.json"

    if rows:
        write_csv(csv_path, rows)

    final_effective_ability = final_info.get(
        "effective_student_ability",
        reset_info.get("effective_student_ability"),
    )
    final_adaptive_ability_score = final_info.get(
        "effective_student_ability_10",
        reset_info.get("effective_student_ability_10"),
    )
    learning_path = env.suggest_learning_path()
    topic_performance = topic_performance_rows(env)

    summary = {
        "session_id": session_id,
        "student_id": args.student_id,
        "algo": args.algo,
        "model_path": str(args.model_path),
        "selection_mode": "pure_model_masked_greedy",
        "initial_effective_ability": reset_info.get("initial_effective_ability"),
        "initial_ability_10": reset_info.get("effective_student_ability_10"),
        "final_effective_ability": final_effective_ability,
        "final_adaptive_ability_score_10": rounded_or_none(final_adaptive_ability_score),
        "questions_answered": len(rows),
        "total_correct": int(env.total_correct),
        "total_wrong": int(env.total_wrong),
        "accuracy": round(env._safe_accuracy(), 4),
        "total_time_taken": round(float(env.total_time_taken), 2),
        "total_agent_reward": round(float(env.total_agent_reward), 4),
        "total_student_reward": round(float(env.total_student_reward), 4),
        "suggested_learning_path": learning_path,
        "topic_performance": topic_performance,
        "results_csv": str(csv_path) if rows else None,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=json_default),
        encoding="utf-8",
    )

    print("\nSession summary")
    print(f"  answered: {summary['questions_answered']}")
    print(f"  correct: {summary['total_correct']}")
    print(f"  wrong: {summary['total_wrong']}")
    print(f"  accuracy: {summary['accuracy']:.3f}")
    print(f"  total_time_taken: {summary['total_time_taken']:.2f}s")
    print(f"  final_effective_ability: {summary['final_effective_ability']}")
    print(f"  final_adaptive_ability_score_10: {summary['final_adaptive_ability_score_10']} / 10")

    print("\nLearning path")
    print(format_numbered_list(summary["suggested_learning_path"]))

    print("\nTopic performance (correct/attempts)")
    if topic_performance:
        for item in topic_performance:
            print(item["display"])
    else:
        print("No topic attempts recorded.")

    print(f"  saved_summary: {summary_path}")
    if rows:
        print(f"  saved_results: {csv_path}")


if __name__ == "__main__":
    main()
