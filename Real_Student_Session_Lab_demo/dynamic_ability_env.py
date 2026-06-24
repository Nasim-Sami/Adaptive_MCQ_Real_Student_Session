from __future__ import annotations

import math
from collections import deque
from typing import Any, Callable

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from dynamic_student_simulator import (
    DIFFICULTY_HIGH,
    MAX_ABILITY,
    MIN_ABILITY,
    QUESTIONS_WITH_METADATA,
    ability_to_difficulty_scale,
    ability_to_10_scale,
    create_student,
    simulate_answer,
    update_dynamic_student_state,
    with_updated_ability,
)


def clip(value: float, low: float, high: float) -> float:
    return float(max(low, min(high, value)))


def default_topic_group(topic: str) -> str:
    t = topic.lower().strip()

    if "conditional" in t:
        return "Conditional sentence"
    if "article" in t:
        return "Articles"
    if "vocabulary" in t or "antonym" in t:
        return "Vocabulary"
    if "logical comparison" in t:
        return "Logical comparison"
    if "subject-verb" in t:
        return "Subject-verb agreement"
    if "subjunctive" in t:
        return "Subjunctive mood"
    if "inversion" in t:
        return "Inversion"

    return topic.strip()


class DynamicAbilityAdaptiveMCQEnv(gym.Env):
    """
    Standalone adaptive MCQ environment with within-episode dynamic ability.

    This env owns reset, step, observation construction, reward shaping, topic
    reports, and learning-path suggestions. It still uses student_simulator.py
    for question metadata and probabilistic answer/time simulation.
    """

    metadata = {"render_modes": []}

    OBS_LAST_CORRECT = 2
    OBS_LAST_TIME_RATIO_NORM = 4
    OBS_LAST_3_ACCURACY = 5
    OBS_LAST_5_ACCURACY = 6
    OBS_LAST_5_AVG_TIME_RATIO_NORM = 8
    OBS_CUMULATIVE_ACCURACY = 9
    OBS_CONSECUTIVE_WRONG_NORM = 11
    OBS_FAST_CORRECT_STREAK_NORM = 12
    OBS_OVERLOAD_STREAK_NORM = 13
    OBS_ANSWERED_NORM = 19
    OBS_ACCURACY_TREND = 21
    OBS_ESTIMATED_ABILITY_NORM = 22
    OBS_STRUGGLING_FLAG = 23
    OBS_READY_TO_ADVANCE_FLAG = 24

    def __init__(
        self,
        *,
        questions: list[dict[str, Any]] | None = None,
        students: list[dict[str, Any]] | None = None,
        episode_length: int = 15,
        seed: int | None = None,
        repair_invalid_action: bool = True,
        random_first_question: bool = False,
        min_effective_ability: float = MIN_ABILITY,
        max_effective_ability: float = MAX_ABILITY,
        max_single_step_change: float = 1.0,
        topic_mapper: Callable[[str], str] = default_topic_group,
    ) -> None:
        super().__init__()

        self.questions = questions if questions is not None else QUESTIONS_WITH_METADATA
        self.n_questions = len(self.questions)

        if students is None:
            self.students = [
                create_student(f"S{i:02d}", ability=i)
                for i in range(MIN_ABILITY, MAX_ABILITY + 1)
            ]
        else:
            self.students = students

        self.episode_length = min(int(episode_length), self.n_questions)
        self.repair_invalid_action = bool(repair_invalid_action)
        self.random_first_question = bool(random_first_question)
        self.rng = np.random.default_rng(seed)

        self.min_effective_ability = float(min_effective_ability)
        self.max_effective_ability = float(max_effective_ability)
        self.max_single_step_change = float(max_single_step_change)

        self.topic_mapper = topic_mapper
        self.question_topics = [
            self.topic_mapper(q["topic"])
            for q in self.questions
        ]
        self.topics = sorted(set(self.question_topics))
        self.topic_to_idx = {
            topic: i
            for i, topic in enumerate(self.topics)
        }
        self.n_topics = len(self.topics)

        self.action_space = spaces.Discrete(self.n_questions)

        self.scalar_dim = 25
        self.history_dim = 10
        self.dynamic_feature_dim = 5
        self.base_obs_dim = (
            self.scalar_dim
            + self.history_dim
            + self.n_questions
            + self.n_questions
            + self.n_topics
            + self.n_topics
            + self.n_topics
        )
        self.obs_dim = self.base_obs_dim + self.dynamic_feature_dim

        base_low = np.full(self.base_obs_dim, -1.0, dtype=np.float32)
        base_high = np.full(self.base_obs_dim, 1.0, dtype=np.float32)
        dynamic_low = np.array(
            [0.0, -DIFFICULTY_HIGH, -DIFFICULTY_HIGH, 0.0, 0.0],
            dtype=np.float32,
        )
        dynamic_high = np.array(
            [DIFFICULTY_HIGH, DIFFICULTY_HIGH, DIFFICULTY_HIGH, DIFFICULTY_HIGH, 1.0],
            dtype=np.float32,
        )

        self.observation_space = spaces.Box(
            low=np.concatenate([base_low, dynamic_low]).astype(np.float32),
            high=np.concatenate([base_high, dynamic_high]).astype(np.float32),
            dtype=np.float32,
        )

        self._initialize_episode_state()

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)

        if seed is not None:
            self.rng = np.random.default_rng(seed)

        student_idx = int(self.rng.integers(0, len(self.students)))
        starting_student = self.students[student_idx]

        self._initialize_episode_state()
        self.current_student_id = str(starting_student.get("student_id", "dynamic_student"))
        self.profile_starting_ability = float(starting_student["ability"])
        self.initial_ability = self.profile_starting_ability
        self.initial_effective_ability = float(
            self.rng.integers(
                int(round(self.min_effective_ability)),
                int(round(self.max_effective_ability)) + 1,
            )
        )
        self.student_profile = with_updated_ability(starting_student, self.profile_starting_ability)
        self.effective_ability = self.initial_effective_ability
        self.previous_effective_ability = self.initial_effective_ability
        self.last_observation_target_ability = self.initial_effective_ability
        self.last_observation_score = 0.5
        self._apply_effective_ability_to_current_student()

        info = {
            "student_id": self.current_student["student_id"],
            "hidden_student_ability": self.current_student["ability"],
            "topics": self.topics,
        }
        info.update(self._ability_info())
        info.update(self._student_state_info())

        return self._get_obs(), info

    def step(self, action: int):
        previous_obs = self._get_obs()
        self._apply_effective_ability_to_current_student()

        action = int(action)
        first_random_question = bool(
            self.random_first_question
            and self.current_step == 0
            and self.asked_mask.sum() == 0
        )

        if first_random_question:
            selected_action = self._sample_valid_action()
            invalid_action = False
        else:
            invalid_action = bool(self.asked_mask[action] == 1.0)
            if invalid_action and self.repair_invalid_action:
                selected_action = self._sample_valid_action()
            else:
                selected_action = action

        question = self.questions[selected_action]
        topic = self.question_topics[selected_action]
        topic_idx = self.topic_to_idx[topic]
        simulation_ability_before = self._simulation_ability()
        effective_ability_before = self.effective_ability

        result = simulate_answer(
            student=self.current_student,
            question=question,
            rng=self.rng,
        )
        student_state_before = self._student_state_info(self.current_student)

        student_reward, agent_reward, time_ratio, response_label = self._compute_rewards(
            result=result,
            question=question,
            topic_idx=topic_idx,
            invalid_action=invalid_action,
        )

        self._update_estimated_student_speed(result, question)

        is_correct = bool(result["is_correct"])

        if is_correct:
            self.topic_correct[topic_idx] += 1
            self.total_correct += 1
            self.consecutive_wrong_count = 0
        else:
            self.topic_wrong[topic_idx] += 1
            self.total_wrong += 1
            self.consecutive_wrong_count += 1

        if is_correct and time_ratio < 0.55:
            self.fast_correct_streak += 1
        else:
            self.fast_correct_streak = 0

        if response_label == "overload":
            self.overload_streak += 1
        else:
            self.overload_streak = 0

        self.asked_mask[selected_action] = 1.0
        self.previous_question_one_hot[:] = 0.0
        self.previous_question_one_hot[selected_action] = 1.0

        self.last_correct = float(is_correct)
        self.last_time_ratio = float(time_ratio)
        self.last_student_reward = float(student_reward)
        self.last_agent_reward = float(agent_reward)

        self.correct_history.append(float(is_correct))
        self.time_ratio_history.append(float(time_ratio))
        self.topic_history.append(topic)

        self.total_time_taken += float(result["time_taken"])
        self.total_student_reward += float(student_reward)
        self.total_agent_reward += float(agent_reward)

        self.student_profile = update_dynamic_student_state(
            student=self.current_student,
            question=question,
            is_correct=is_correct,
            time_ratio=time_ratio,
        )

        base_next_obs = self._get_base_obs()
        ability_delta = self._compute_ability_delta(base_next_obs, is_correct, time_ratio)
        self.previous_effective_ability = self.effective_ability
        self.effective_ability = clip(
            self.effective_ability + ability_delta,
            self.min_effective_ability,
            self.max_effective_ability,
        )
        self.last_ability_delta = self.effective_ability - self.previous_effective_ability
        self._apply_effective_ability_to_current_student()

        ability_info = self._ability_info()
        student_state_after = self._student_state_info()

        self.episode_rows.append(
            {
                "step": self.current_step,
                "student_id": self.current_student_id,
                "question_id": question["question_id"],
                "topic": topic,
                "subtopic": question.get("subtopic"),
                "chosen_option": result["chosen_option"],
                "correct_answer": result["correct_answer"],
                "is_correct": is_correct,
                "time_taken": result["time_taken"],
                "time_ratio": time_ratio,
                "student_reward": student_reward,
                "agent_reward": agent_reward,
                "invalid_action": invalid_action,
                "first_random_question": first_random_question,
                "response_label": response_label,
                "inherent_difficulty": self.last_inherent_difficulty,
                "target_inherent_difficulty": self.last_target_inherent_difficulty,
                "ideal_inherent_difficulty": self.last_ideal_inherent_difficulty,
                "difficulty_gap": self.last_difficulty_gap,
                "difficulty_match_reward": self.last_difficulty_match_reward,
                "topic_reward": self.last_topic_reward,
                "diversity_reward": self.last_diversity_reward,
                "invalid_action_penalty": self.last_invalid_action_penalty,
                "acting_mode": result.get("acting_mode"),
                "acting_ability": result.get("acting_ability"),
                "acting_ability_10": result.get("acting_ability_10"),
                "acting_ability_difficulty_scale": result.get("acting_ability_difficulty_scale"),
                "effective_ability_before": round(effective_ability_before, 4),
                "simulation_ability_before": simulation_ability_before,
                **{
                    f"{key}_before": value
                    for key, value in student_state_before.items()
                },
                **student_state_after,
                **ability_info,
            }
        )

        self.current_step += 1

        terminated = bool(
            self.current_step >= self.episode_length
            or self.asked_mask.sum() >= self.n_questions
        )
        truncated = False
        next_obs = self._get_obs()

        info = {
            "previous_obs": previous_obs,
            "next_obs": next_obs,
            "student_id": self.current_student_id,
            "hidden_student_ability": simulation_ability_before,
            "sampled_perceived_difficulty": result["sampled_perceived_difficulty"],
            "option_distribution": result["option_distribution"],
            "selected_action": selected_action,
            "original_action": action,
            "invalid_action": invalid_action,
            "first_random_question": first_random_question,
            "question_id": question["question_id"],
            "topic": topic,
            "subtopic": question.get("subtopic"),
            "chosen_option": result["chosen_option"],
            "correct_answer": result["correct_answer"],
            "is_correct": is_correct,
            "time_taken": result["time_taken"],
            "time_ratio": time_ratio,
            "response_label": response_label,
            "inherent_difficulty": self.last_inherent_difficulty,
            "target_inherent_difficulty": self.last_target_inherent_difficulty,
            "ideal_inherent_difficulty": self.last_ideal_inherent_difficulty,
            "difficulty_gap": self.last_difficulty_gap,
            "difficulty_match_reward": self.last_difficulty_match_reward,
            "topic_reward": self.last_topic_reward,
            "diversity_reward": self.last_diversity_reward,
            "invalid_action_penalty": self.last_invalid_action_penalty,
            "acting_mode": result.get("acting_mode"),
            "acting_ability": result.get("acting_ability"),
            "acting_ability_10": result.get("acting_ability_10"),
            "acting_ability_difficulty_scale": result.get("acting_ability_difficulty_scale"),
            "student_reward": student_reward,
            "agent_reward": agent_reward,
            "topic_correct_count": int(self.topic_correct[topic_idx]),
            "topic_wrong_count": int(self.topic_wrong[topic_idx]),
            "total_student_reward": self.total_student_reward,
            "total_agent_reward": self.total_agent_reward,
            "total_time_taken": self.total_time_taken,
            "accuracy": self._safe_accuracy(),
            "simulation_ability_before": simulation_ability_before,
            "effective_ability_before": round(effective_ability_before, 4),
        }
        info.update(ability_info)
        info.update(
            {
                f"{key}_before": value
                for key, value in student_state_before.items()
            }
        )
        info.update(student_state_after)

        if terminated:
            info["topic_report"] = self.get_topic_report()
            info["suggested_learning_path"] = self.suggest_learning_path()
            info["episode_rows"] = self.episode_rows

        return next_obs, float(agent_reward), terminated, truncated, info

    def _initialize_episode_state(self) -> None:
        self.current_student_id = "dynamic_student"
        self.current_student = create_student(self.current_student_id, ability=MIN_ABILITY)
        self.student_profile = dict(self.current_student)

        self.current_step = 0
        self.asked_mask = np.zeros(self.n_questions, dtype=np.float32)
        self.previous_question_one_hot = np.zeros(self.n_questions, dtype=np.float32)

        self.topic_correct = np.zeros(self.n_topics, dtype=np.float32)
        self.topic_wrong = np.zeros(self.n_topics, dtype=np.float32)

        self.last_correct = 0.0
        self.last_time_ratio = 1.0
        self.last_student_reward = 0.0
        self.last_agent_reward = 0.0
        self.last_inherent_difficulty = 0.0
        self.last_target_inherent_difficulty = 1.0
        self.last_ideal_inherent_difficulty = 1.0
        self.last_difficulty_gap = 0.0
        self.last_difficulty_match_reward = 0.0
        self.last_topic_reward = 0.0
        self.last_diversity_reward = 0.0
        self.last_invalid_action_penalty = 0.0

        self.consecutive_wrong_count = 0
        self.fast_correct_streak = 0
        self.overload_streak = 0

        self.total_correct = 0
        self.total_wrong = 0
        self.total_time_taken = 0.0
        self.total_student_reward = 0.0
        self.total_agent_reward = 0.0

        self.estimated_student_speed = 1.0

        self.correct_history = deque(maxlen=5)
        self.time_ratio_history = deque(maxlen=5)
        self.topic_history = deque(maxlen=5)
        self.episode_rows: list[dict[str, Any]] = []

        self.profile_starting_ability = float(MIN_ABILITY)
        self.initial_ability = float(MIN_ABILITY)
        self.initial_effective_ability = float(MIN_ABILITY)
        self.effective_ability = float(MIN_ABILITY)
        self.previous_effective_ability = float(MIN_ABILITY)
        self.last_ability_delta = 0.0
        self.last_observation_target_ability = float(MIN_ABILITY)
        self.last_observation_score = 0.5

    def _get_obs(self) -> np.ndarray:
        return np.concatenate(
            [
                self._get_base_obs(),
                self._dynamic_features(),
            ]
        ).astype(np.float32)

    def _get_base_obs(self) -> np.ndarray:
        current_step_norm = self.current_step / max(1, self.episode_length)
        answered_count = self.total_correct + self.total_wrong
        questions_remaining_norm = clip(
            (self.episode_length - self.current_step) / max(1, self.episode_length),
            0.0,
            1.0,
        )

        last_3_accuracy = self._recent_accuracy(3)
        last_5_accuracy = self._recent_accuracy(5)
        last_3_avg_time_ratio = self._recent_avg_time_ratio(3)
        last_5_avg_time_ratio = self._recent_avg_time_ratio(5)
        cumulative_accuracy = self._safe_accuracy()
        cumulative_avg_time_ratio = self._safe_avg_time_ratio()

        consecutive_wrong_norm = clip(self.consecutive_wrong_count / 5.0, 0.0, 1.0)
        fast_correct_streak_norm = clip(self.fast_correct_streak / 5.0, 0.0, 1.0)
        overload_streak_norm = clip(self.overload_streak / 5.0, 0.0, 1.0)
        last_time_ratio_norm = clip(self.last_time_ratio / 3.0, 0.0, 1.0)

        last_student_reward_scaled = clip(self.last_student_reward / 2.0, -1.0, 1.0)
        last_agent_reward_scaled = clip(self.last_agent_reward / 3.0, -1.0, 1.0)

        total_time_norm = clip(self.total_time_taken / 900.0, 0.0, 1.0)
        total_correct_norm = clip(self.total_correct / max(1, self.episode_length), 0.0, 1.0)
        total_wrong_norm = clip(self.total_wrong / max(1, self.episode_length), 0.0, 1.0)
        answered_norm = clip(answered_count / max(1, self.episode_length), 0.0, 1.0)

        last_time_increase_norm = 0.0
        if len(self.episode_rows) >= 2:
            last_time = float(self.episode_rows[-1].get("time_taken", 0.0))
            previous_time = float(self.episode_rows[-2].get("time_taken", 0.0))
            last_time_increase_norm = clip((last_time - previous_time) / 60.0, -1.0, 1.0)

        accuracy_trend = clip(last_3_accuracy - last_5_accuracy, -1.0, 1.0)
        time_adjustment = 0.0
        if answered_count > 0:
            if last_5_avg_time_ratio < 0.22:
                time_adjustment += 0.08
            elif last_5_avg_time_ratio > 0.57:
                time_adjustment -= 0.11
            elif last_5_avg_time_ratio > 0.43:
                time_adjustment -= 0.05

        estimated_ability_norm = clip(
            0.45 * 0.50
            + 0.35 * cumulative_accuracy
            + 0.20 * last_5_accuracy
            + time_adjustment
            - 0.06 * min(self.consecutive_wrong_count, 3),
            0.0,
            1.0,
        )
        struggling_flag = float(
            self.consecutive_wrong_count >= 2
            or self.overload_streak >= 1
            or (answered_count >= 3 and last_5_accuracy <= 0.40)
        )
        ready_to_advance_flag = float(
            answered_count >= 3
            and last_5_accuracy >= 0.80
            and last_5_avg_time_ratio <= 0.50
        )
        last_wrong = float(1.0 - self.last_correct) if answered_count > 0 else 0.0

        scalar_features = np.array(
            [
                current_step_norm,
                questions_remaining_norm,
                self.last_correct,
                last_wrong,
                last_time_ratio_norm,
                last_3_accuracy,
                last_5_accuracy,
                last_3_avg_time_ratio,
                last_5_avg_time_ratio,
                cumulative_accuracy,
                cumulative_avg_time_ratio,
                consecutive_wrong_norm,
                fast_correct_streak_norm,
                overload_streak_norm,
                last_student_reward_scaled,
                last_agent_reward_scaled,
                total_time_norm,
                total_correct_norm,
                total_wrong_norm,
                answered_norm,
                last_time_increase_norm,
                accuracy_trend,
                estimated_ability_norm,
                struggling_flag,
                ready_to_advance_flag,
            ],
            dtype=np.float32,
        )

        recent_correct_features = np.zeros(5, dtype=np.float32)
        recent_time_features = np.zeros(5, dtype=np.float32)

        recent_correct_values = list(self.correct_history)[-5:]
        if recent_correct_values:
            recent_correct_features[-len(recent_correct_values):] = np.asarray(
                recent_correct_values,
                dtype=np.float32,
            )

        recent_time_values = [
            clip(value / 3.0, 0.0, 1.0)
            for value in list(self.time_ratio_history)[-5:]
        ]
        if recent_time_values:
            recent_time_features[-len(recent_time_values):] = np.asarray(
                recent_time_values,
                dtype=np.float32,
            )

        topic_attempts = self.topic_correct + self.topic_wrong
        topic_attempt_rate = topic_attempts / max(1, self.episode_length)
        topic_accuracy = np.divide(self.topic_correct, np.maximum(topic_attempts, 1.0))
        topic_wrong_rate = np.divide(self.topic_wrong, np.maximum(topic_attempts, 1.0))

        return np.concatenate(
            [
                scalar_features,
                recent_correct_features,
                recent_time_features,
                self.previous_question_one_hot.astype(np.float32),
                self.asked_mask.astype(np.float32),
                topic_attempt_rate.astype(np.float32),
                topic_accuracy.astype(np.float32),
                topic_wrong_rate.astype(np.float32),
            ]
        ).astype(np.float32)

    def _dynamic_features(self) -> np.ndarray:
        effective_ability_difficulty_scale = ability_to_difficulty_scale(self.effective_ability)
        effective_minus_initial_difficulty_scale = self._scale_ability_delta_to_difficulty_scale(
            self.effective_ability - self.initial_ability
        )
        last_delta_difficulty_scale = self._scale_ability_delta_to_difficulty_scale(
            self.last_ability_delta
        )
        target_ability_difficulty_scale = ability_to_difficulty_scale(
            self.last_observation_target_ability
        )

        return np.array(
            [
                clip(effective_ability_difficulty_scale, 0.0, DIFFICULTY_HIGH),
                clip(effective_minus_initial_difficulty_scale, -DIFFICULTY_HIGH, DIFFICULTY_HIGH),
                clip(last_delta_difficulty_scale, -DIFFICULTY_HIGH, DIFFICULTY_HIGH),
                clip(target_ability_difficulty_scale, 0.0, DIFFICULTY_HIGH),
                clip(self.last_observation_score, 0.0, 1.0),
            ],
            dtype=np.float32,
        )

    def _compute_rewards(
        self,
        result: dict[str, Any],
        question: dict[str, Any],
        topic_idx: int,
        invalid_action: bool,
    ) -> tuple[float, float, float, str]:
        is_correct = bool(result["is_correct"])
        time_taken = float(result["time_taken"])
        base_time = float(question["base_time"])
        time_ratio = time_taken / max(base_time * self.estimated_student_speed, 1e-6)
        inherent_difficulty = float(question["inherent_difficulty"])

        ability_norm = (
            (self.effective_ability - self.min_effective_ability)
            / max(self.max_effective_ability - self.min_effective_ability, 1e-6)
        )
        target_difficulty = 1.0 + clip(ability_norm, 0.0, 1.0) * 9.0
        ideal_difficulty = clip(target_difficulty + 0.35, 1.0, 10.0)
        difficulty_gap = inherent_difficulty - ideal_difficulty
        abs_gap = abs(difficulty_gap)

        if -0.75 <= difficulty_gap <= 0.75:
            difficulty_match_reward = 1.30
            difficulty_label = "best_match"
        elif 0.75 < difficulty_gap <= 1.50:
            difficulty_match_reward = 1.20
            difficulty_label = "slightly_hard"
        elif -1.50 <= difficulty_gap < -0.75:
            difficulty_match_reward = 0.30
            difficulty_label = "slightly_easy"
        elif difficulty_gap > 1.50:
            difficulty_match_reward = -0.40 - 0.35 * min(difficulty_gap - 1.50, 4.00)
            difficulty_label = "too_hard"
        else:
            difficulty_match_reward = -0.40 - 0.30 * min(abs_gap - 1.50, 4.00)
            difficulty_label = "too_easy"

        difficulty_is_suitable = -1.50 <= difficulty_gap <= 1.50
        difficulty_is_too_hard = difficulty_gap > 1.50
        difficulty_is_too_easy = difficulty_gap < -1.50

        if is_correct:
            if time_ratio <= 1.50:
                student_reward = 2.00
            elif time_ratio <= 2.20:
                student_reward = 1.50
            elif time_ratio <= 3.00:
                student_reward = 1.00
            else:
                student_reward = 0.50
        else:
            student_reward = 0.00

        topic_attempts_before = self.topic_correct[topic_idx] + self.topic_wrong[topic_idx]
        if topic_attempts_before > 0:
            topic_accuracy_before = self.topic_correct[topic_idx] / topic_attempts_before
            topic_is_weak_before = topic_accuracy_before < 0.50
            topic_is_strong_before = topic_accuracy_before >= 0.75
            topic_is_unseen_before = False
        else:
            topic_is_weak_before = False
            topic_is_strong_before = False
            topic_is_unseen_before = True

        topic_reward = 0.0
        topic_label = "topic_neutral"
        if topic_is_weak_before and difficulty_is_suitable:
            topic_reward += 0.60
            topic_label = "weak_topic_suitable"
        elif topic_is_weak_before and difficulty_is_too_hard:
            topic_reward -= 0.40
            topic_label = "weak_topic_too_hard"
        elif topic_is_strong_before and difficulty_is_too_easy:
            topic_reward -= 0.30
            topic_label = "strong_topic_too_easy"
        elif topic_is_unseen_before and difficulty_is_suitable:
            topic_reward += 0.20
            topic_label = "unseen_topic_suitable"

        recent_same_topic_count = 0
        current_topic = self.topics[topic_idx]
        for old_topic in list(self.topic_history)[-3:]:
            if old_topic == current_topic:
                recent_same_topic_count += 1

        diversity_reward = 0.0
        diversity_label = "diversity_neutral"
        if recent_same_topic_count >= 2 and not topic_is_weak_before:
            diversity_reward -= 0.25
            diversity_label = "repeated_topic_not_weak"
        elif recent_same_topic_count == 0 and difficulty_is_suitable:
            diversity_reward += 0.30
            diversity_label = "new_useful_topic"

        invalid_action_penalty = 0.0
        if invalid_action:
            invalid_action_penalty = -0.25

        agent_reward = (
            difficulty_match_reward
            + topic_reward
            + diversity_reward
            + invalid_action_penalty
        )
        response_label = f"{difficulty_label}_{topic_label}_{diversity_label}"

        self.last_inherent_difficulty = round(inherent_difficulty, 4)
        self.last_target_inherent_difficulty = round(target_difficulty, 4)
        self.last_ideal_inherent_difficulty = round(ideal_difficulty, 4)
        self.last_difficulty_gap = round(difficulty_gap, 4)
        self.last_difficulty_match_reward = round(difficulty_match_reward, 4)
        self.last_topic_reward = round(topic_reward, 4)
        self.last_diversity_reward = round(diversity_reward, 4)
        self.last_invalid_action_penalty = round(invalid_action_penalty, 4)

        agent_reward = clip(agent_reward, -5.0, 3.0)
        student_reward = clip(student_reward, 0.0, 2.0)
        return student_reward, agent_reward, float(time_ratio), response_label

    def _compute_ability_delta(
        self,
        base_obs: np.ndarray,
        is_correct: bool,
        time_ratio: float,
    ) -> float:
        metrics = self._obs_metrics(base_obs)
        target_ability = self._target_ability_from_obs(metrics, is_correct, time_ratio)
        self.last_observation_target_ability = target_ability
        self.last_observation_score = metrics["performance_score"]

        target_error = target_ability - self.effective_ability
        answered_confidence = clip(0.15 + metrics["answered_norm"], 0.15, 0.70)
        target_delta = clip(0.16 * target_error * answered_confidence, -0.30, 0.30)

        immediate_delta = self._time_ratio_ability_delta(
            is_correct=is_correct,
            time_ratio=time_ratio,
            consecutive_wrong_norm=metrics["consecutive_wrong_norm"],
        )

        return clip(
            target_delta + immediate_delta,
            -self.max_single_step_change,
            self.max_single_step_change,
        )

    def _time_ratio_ability_delta(
        self,
        *,
        is_correct: bool,
        time_ratio: float,
        consecutive_wrong_norm: float,
    ) -> float:
        if is_correct:
            if time_ratio <= 0.30:
                return 0.55
            if time_ratio <= 0.60:
                return 0.45
            if time_ratio <= 0.80:
                return 0.35
            if time_ratio <= 1.00:
                return 0.25
            if time_ratio <= 1.20:
                return 0.15
            if time_ratio <= 1.50:
                return 0.05
            if time_ratio <= 1.80:
                return -0.40
            if time_ratio <= 2.20:
                return -0.60
            if time_ratio <= 2.60:
                return -0.80
            if time_ratio <= 3.00:
                return -1.00
            return -1.15

        if time_ratio < 0.70:
            immediate_delta = -0.35
        elif time_ratio <= 1.00:
            immediate_delta = -0.55
        elif time_ratio <= 1.50:
            immediate_delta = -0.75
        elif time_ratio <= 2.20:
            immediate_delta = -0.95
        elif time_ratio <= 3.00:
            immediate_delta = -1.10
        else:
            immediate_delta = -1.25

        return immediate_delta - 0.25 * consecutive_wrong_norm

    def _obs_metrics(self, obs: np.ndarray) -> dict[str, float]:
        obs = np.asarray(obs, dtype=np.float32)

        def value(index: int, default: float = 0.0) -> float:
            if index >= obs.shape[0]:
                return default
            return float(obs[index])

        last_correct = value(self.OBS_LAST_CORRECT)
        last_time_ratio = value(self.OBS_LAST_TIME_RATIO_NORM) * 3.0
        last_3_accuracy = value(self.OBS_LAST_3_ACCURACY)
        last_5_accuracy = value(self.OBS_LAST_5_ACCURACY)
        cumulative_accuracy = value(self.OBS_CUMULATIVE_ACCURACY)
        consecutive_wrong_norm = value(self.OBS_CONSECUTIVE_WRONG_NORM)
        fast_correct_streak_norm = value(self.OBS_FAST_CORRECT_STREAK_NORM)
        overload_streak_norm = value(self.OBS_OVERLOAD_STREAK_NORM)
        accuracy_trend = value(self.OBS_ACCURACY_TREND)
        estimated_ability_norm = value(self.OBS_ESTIMATED_ABILITY_NORM, 0.5)
        struggling_flag = value(self.OBS_STRUGGLING_FLAG)
        ready_to_advance_flag = value(self.OBS_READY_TO_ADVANCE_FLAG)

        if last_correct >= 0.5:
            if last_time_ratio <= 0.30:
                time_score = 1.00
            elif last_time_ratio <= 0.80:
                time_score = 0.92
            elif last_time_ratio <= 1.50:
                time_score = 0.78
            elif last_time_ratio <= 2.20:
                time_score = 0.58
            elif last_time_ratio <= 3.00:
                time_score = 0.42
            else:
                time_score = 0.22
        else:
            if last_time_ratio < 0.70:
                time_score = 0.30
            elif last_time_ratio <= 1.50:
                time_score = 0.18
            else:
                time_score = 0.06

        trend_score = clip(0.50 + accuracy_trend, 0.0, 1.0)
        performance_score = (
            0.28 * cumulative_accuracy
            + 0.24 * last_5_accuracy
            + 0.16 * last_3_accuracy
            + 0.14 * time_score
            + 0.12 * estimated_ability_norm
            + 0.06 * trend_score
            + 0.08 * ready_to_advance_flag
            + 0.05 * fast_correct_streak_norm
            - 0.13 * struggling_flag
            - 0.10 * consecutive_wrong_norm
            - 0.08 * overload_streak_norm
        )

        return {
            "last_correct": last_correct,
            "last_time_ratio": last_time_ratio,
            "last_3_accuracy": last_3_accuracy,
            "last_5_accuracy": last_5_accuracy,
            "last_5_avg_time_ratio": value(self.OBS_LAST_5_AVG_TIME_RATIO_NORM) * 3.0,
            "cumulative_accuracy": cumulative_accuracy,
            "consecutive_wrong_norm": consecutive_wrong_norm,
            "answered_norm": value(self.OBS_ANSWERED_NORM),
            "estimated_ability_norm": estimated_ability_norm,
            "performance_score": clip(performance_score, 0.0, 1.0),
        }

    def _target_ability_from_obs(
        self,
        metrics: dict[str, float],
        is_correct: bool,
        time_ratio: float,
    ) -> float:
        ability_range = self.max_effective_ability - self.min_effective_ability
        target = self.min_effective_ability + ability_range * metrics["performance_score"]

        if is_correct and time_ratio <= 0.80:
            target += 1.25
        elif is_correct and time_ratio <= 1.50:
            target += 0.75
        elif not is_correct and time_ratio > 1.50:
            target -= 1.25

        return clip(target, self.min_effective_ability, self.max_effective_ability)

    def _apply_effective_ability_to_current_student(self) -> None:
        simulation_ability = self._simulation_ability()
        self.current_student = with_updated_ability(self.student_profile, simulation_ability)
        self.current_student["effective_ability"] = round(self.effective_ability, 4)
        self.current_student["effective_ability_10"] = round(
            ability_to_10_scale(self.effective_ability),
            4,
        )
        self.current_student["effective_ability_difficulty_scale"] = round(
            ability_to_difficulty_scale(self.effective_ability),
            4,
        )

    def _simulation_ability(self) -> int:
        return int(round(clip(self.effective_ability, MIN_ABILITY, MAX_ABILITY)))

    def _ability_info(self) -> dict[str, float | int]:
        ability_range = max(1.0, float(MAX_ABILITY - MIN_ABILITY))
        return {
            "initial_student_ability": round(self.initial_ability, 4),
            "initial_effective_ability": round(self.initial_effective_ability, 4),
            "effective_student_ability": round(self.effective_ability, 4),
            "effective_student_ability_10": round(ability_to_10_scale(self.effective_ability), 4),
            "effective_student_ability_difficulty_scale": round(
                ability_to_difficulty_scale(self.effective_ability),
                4,
            ),
            "simulation_student_ability": self._simulation_ability(),
            "ability_delta": round(self.last_ability_delta, 4),
            "ability_delta_10": round(self._scale_ability_delta_to_10(self.last_ability_delta), 4),
            "ability_delta_difficulty_scale": round(
                self._scale_ability_delta_to_difficulty_scale(self.last_ability_delta),
                4,
            ),
            "observation_target_ability": round(self.last_observation_target_ability, 4),
            "observation_target_ability_10": round(
                ability_to_10_scale(self.last_observation_target_ability),
                4,
            ),
            "observation_target_ability_difficulty_scale": round(
                ability_to_difficulty_scale(self.last_observation_target_ability),
                4,
            ),
            "observation_performance_score": round(self.last_observation_score, 4),
            "effective_ability_norm": round((self.effective_ability - MIN_ABILITY) / ability_range, 4),
        }

    def _student_state_info(self, student: dict[str, Any] | None = None) -> dict[str, float]:
        student = self.current_student if student is None else student
        topic_bias = student.get("topic_mastery_bias", {}) or {}
        topic_values = [float(value) for value in topic_bias.values()]
        topic_mean = float(np.mean(topic_values)) if topic_values else 0.0
        topic_std = float(np.std(topic_values)) if topic_values else 0.0
        return {
            "student_confidence": round(float(student.get("confidence", 0.5)), 4),
            "student_fatigue": round(float(student.get("fatigue", 0.0)), 4),
            "student_guessing_tendency": round(float(student.get("guessing_tendency", 0.1)), 4),
            "student_speed_tendency": round(float(student.get("speed_tendency", 1.0)), 4),
            "topic_mastery_bias_mean": round(topic_mean, 4),
            "topic_mastery_bias_std": round(topic_std, 4),
            "last_dynamic_topic_delta": round(float(student.get("last_dynamic_topic_delta", 0.0)), 4),
        }

    def _scale_ability_delta_to_10(self, delta: float) -> float:
        ability_range = max(1.0, self.max_effective_ability - self.min_effective_ability)
        return (float(delta) / ability_range) * 10.0

    def _scale_ability_delta_to_difficulty_scale(self, delta: float) -> float:
        ability_range = max(1.0, self.max_effective_ability - self.min_effective_ability)
        difficulty_range = float(DIFFICULTY_HIGH)
        return (float(delta) / ability_range) * difficulty_range

    def _update_estimated_student_speed(
        self,
        result: dict[str, Any],
        question: dict[str, Any],
    ) -> None:
        if not bool(result["is_correct"]):
            return

        time_taken = float(result["time_taken"])
        base_time = float(question["base_time"])
        observed_speed = clip(time_taken / max(base_time, 1.0), 0.40, 2.50)
        self.estimated_student_speed = 0.85 * self.estimated_student_speed + 0.15 * observed_speed

    def _sample_valid_action(self) -> int:
        valid_actions = np.flatnonzero(self.asked_mask == 0.0)
        if len(valid_actions) == 0:
            return 0
        return int(self.rng.choice(valid_actions))

    def _recent_accuracy(self, n: int) -> float:
        values = list(self.correct_history)[-n:]
        if not values:
            return 0.0
        return float(np.mean(values))

    def _recent_avg_time_ratio(self, n: int) -> float:
        values = list(self.time_ratio_history)[-n:]
        if not values:
            return 0.0
        return clip(float(np.mean(values)) / 3.0, 0.0, 1.0)

    def _safe_accuracy(self) -> float:
        total = self.total_correct + self.total_wrong
        if total == 0:
            return 0.0
        return float(self.total_correct / total)

    def _safe_avg_time_ratio(self) -> float:
        values = list(self.time_ratio_history)
        if not values:
            return 0.0
        return clip(float(np.mean(values)) / 3.0, 0.0, 1.0)

    def get_topic_report(self) -> dict[str, dict[str, Any]]:
        report = {}

        for topic, idx in self.topic_to_idx.items():
            correct = int(self.topic_correct[idx])
            wrong = int(self.topic_wrong[idx])
            attempts = correct + wrong

            if attempts == 0:
                accuracy = None
                status = "unseen"
            else:
                accuracy = correct / attempts
                if accuracy >= 0.75 and attempts >= 2:
                    status = "strength"
                elif accuracy <= 0.50:
                    status = "needs_improvement"
                else:
                    status = "developing"

            report[topic] = {
                "correct": correct,
                "wrong": wrong,
                "attempts": attempts,
                "accuracy": accuracy,
                "status": status,
            }

        return report

    def suggest_learning_path(self, max_topics: int = 6) -> list[str]:
        report = self.get_topic_report()
        weak_topics = []
        developing_topics = []
        unseen_topics = []

        for topic, stats in report.items():
            attempts = stats["attempts"]
            accuracy = stats["accuracy"]
            wrong = stats["wrong"]

            if attempts == 0:
                unseen_topics.append(topic)
                continue

            if stats["status"] == "needs_improvement":
                weak_topics.append((topic, accuracy, -wrong))
            elif stats["status"] == "developing":
                developing_topics.append((topic, accuracy, -wrong))

        weak_topics.sort(key=lambda x: (x[1], x[2]))
        developing_topics.sort(key=lambda x: (x[1], x[2]))

        path = []
        path.extend([x[0] for x in weak_topics])
        path.extend([x[0] for x in developing_topics])
        path.extend(unseen_topics)
        return path[:max_topics]

    def render(self):
        print(f"Step: {self.current_step}/{self.episode_length}")
        print(f"Accuracy: {self._safe_accuracy():.2f}")
        print(f"Effective ability: {self.effective_ability:.2f}")
        print(f"Total agent reward: {self.total_agent_reward:.2f}")
        print(f"Total student reward: {self.total_student_reward:.2f}")
        print("Suggested path:", " -> ".join(self.suggest_learning_path()))


def make_dynamic_ability_env(
    *,
    questions: list[dict[str, Any]] | None = None,
    students: list[dict[str, Any]] | None = None,
    episode_length: int = 15,
    seed: int | None = None,
    repair_invalid_action: bool = True,
    random_first_question: bool = False,
) -> DynamicAbilityAdaptiveMCQEnv:
    return DynamicAbilityAdaptiveMCQEnv(
        questions=questions,
        students=students,
        episode_length=episode_length,
        seed=seed,
        repair_invalid_action=repair_invalid_action,
        random_first_question=random_first_question,
    )


if __name__ == "__main__":
    env = make_dynamic_ability_env(
        students=[create_student("demo_dynamic_S27", ability=27)],
        episode_length=15,
        seed=7,
    )
    obs, info = env.reset(seed=7)
    print("observation_dim:", obs.shape[0])
    print("reset:", info)

    terminated = False
    truncated = False
    while not terminated and not truncated:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(
            info["question_id"],
            "mode:", info["acting_mode"],
            "acting_ability:", info["acting_ability"],
            "correct:", info["is_correct"],
            "time_ratio:", round(info["time_ratio"], 3),
            "reward:", round(reward, 3),
            "effective_ability:", info["effective_student_ability"],
            "ability_10:", info["effective_student_ability_10"],
            "ability_difficulty_scale:", info["effective_student_ability_difficulty_scale"],
            "delta:", info["ability_delta"],
        )
