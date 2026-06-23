"""
Topic-selection RL environment for the June-15 adaptive-mechatronics system.

Key differences from the earlier (June-5) design
-------------------------------------------------
1. THE AGENT SELECTS A TOPIC, not a question.
       action_space = Discrete(45 topics)
   A deterministic RULE then picks, *inside that topic*, the question whose
   inherent difficulty is closest to the student's current (hidden) effective
   ability.  We are not choosing question IDs with RL.

2. The student's effective ability / per-topic mastery are NEVER in the
   observation.  The agent only sees behavioural signals (correctness, time
   ratios, streaks, observed per-topic accuracy, asked counts) and the
   *structural* prerequisite information.  (Requirement 1.)

3. No difficulty-gap term in the reward.  The reward is
       learning-gain  +  prerequisite-respect  +  coverage
   (Requirement: "Learning gain + prerequisites + coverage").

4. Prerequisite DAG, transfer/interference, recent-performance features and a
   hidden learning style are all in play (Requirements 2-5).

Episode structure (training)
----------------------------
* One *episode* = ``n_sub_episodes`` x ``sub_episode_length`` questions
  (default 4 x 20 = 80), like 4 "lives" in Atari Breakout.
* The student is NEVER reset between sub-episodes - effective ability, mastery,
  fatigue, confidence and study history all carry over.  The agent keeps making
  decisions on the carried state.
* A full reset (new student) happens only after all sub-episodes finish.
* Questions are not repeated within an episode (enforced by the action mask:
  a topic becomes unavailable once all four of its questions are used).

For the *real student session* we instantiate with ``n_sub_episodes=1`` so one
episode is exactly 20 questions with no sub-episodes.
"""
from __future__ import annotations

from collections import deque
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

import curriculum
import difficulty_control as dc
import question_bank as qb
import student_simulator as sim

clip = sim.clip

BANK_DIFFICULTIES = [float(q["inherent_difficulty"]) for q in qb.QUESTIONS_WITH_METADATA]
BANK_MIN_DIFF = min(BANK_DIFFICULTIES)
BANK_MAX_DIFF = max(BANK_DIFFICULTIES)


def ability_to_target_difficulty(effective_ability: float) -> float:
    """Map effective ability (10..30) to a target inherent difficulty in the bank."""
    norm = (clip(effective_ability, sim.MIN_ABILITY, sim.MAX_ABILITY) - sim.MIN_ABILITY) / (
        sim.MAX_ABILITY - sim.MIN_ABILITY
    )
    return BANK_MIN_DIFF + norm * (BANK_MAX_DIFF - BANK_MIN_DIFF)


class TopicSelectionMCQEnv(gym.Env):
    metadata = {"render_modes": []}

    # ---- scalar (recent-performance) feature layout --------------------------
    SCALAR_FEATURES = [
        "current_step_norm",
        "sub_episode_norm",
        "step_in_sub_norm",
        "last_correct",
        "last_time_ratio_norm",
        "last_3_accuracy",
        "last_4_accuracy",
        "last_3_avg_time_ratio_norm",
        "last_4_avg_time_ratio_norm",
        "cumulative_accuracy",
        "cumulative_avg_time_ratio_norm",
        "consecutive_correct_norm",
        "consecutive_wrong_norm",
        "fast_correct_streak_norm",
        "overload_streak_norm",
        "struggling_flag",
        "ready_to_advance_flag",
        "repeat_last_topic_flag",
        "same_cluster_as_last_flag",
    ]
    # per-topic feature blocks (each length n_topics)
    TOPIC_BLOCKS = [
        "topic_asked_rate",
        "topic_accuracy",
        "topic_wrong_rate",
        "topic_seen_flag",
        "topic_prereq_ready_observed",
        "topic_available_flag",
        "topic_recency_norm",   # steps since last practiced (spacing signal)
    ]

    def __init__(
        self,
        *,
        students: list[dict[str, Any]] | None = None,
        sub_episode_length: int = 20,
        n_sub_episodes: int = 4,
        seed: int | None = None,
        randomize_initial_ability: bool = True,
        # Each episode focuses on a contiguous "chapter" of this many topics
        # (a realistic tutoring session covers a chapter, not all 45 topics).
        # Restricting the action set removes the 45-topic dilution, shrinks the
        # effective action space so the policy converges, and makes inferring the
        # hidden learning style tractable.  Set >= n_topics to disable focus.
        n_active_topics: int = 10,
        # ---- reward (true-mastery based) ------------------------------------
        # Primary signal is the per-step change in the student's TRUE latent mean
        # mastery (dense, but equal to the genuine learning gain - not a farmable
        # proxy).  Extra per-step shaping is kept tiny.
        reward_mastery_coef: float = 100.0,  # weight on per-step mastery gain (dominant)
        reward_terminal_coef: float = 10.0,  # weight on fraction of topics mastered
        reward_shaping_coef: float = 0.05,   # tiny per-step prereq/penalty shaping
        mastery_threshold: float = 0.6,      # "mastered" cutoff for terminal bonus
    ) -> None:
        super().__init__()
        self.n_active_topics = int(n_active_topics)
        self.reward_mastery_coef = float(reward_mastery_coef)
        self.reward_terminal_coef = float(reward_terminal_coef)
        self.reward_shaping_coef = float(reward_shaping_coef)
        self.mastery_threshold = float(mastery_threshold)
        self.topics = list(curriculum.TOPICS)
        self.n_topics = len(self.topics)
        self.topic_to_idx = dict(curriculum.TOPIC_TO_IDX)
        self.questions = qb.QUESTIONS_WITH_METADATA
        self.n_questions = len(self.questions)
        self.qid_to_idx = {q["question_id"]: i for i, q in enumerate(self.questions)}

        self.sub_episode_length = int(sub_episode_length)
        self.n_sub_episodes = int(n_sub_episodes)
        self.episode_length = self.sub_episode_length * self.n_sub_episodes
        self.randomize_initial_ability = bool(randomize_initial_ability)
        # Anti-massing used to be a hard mask constraint; it is now DISABLED (0)
        # so the agent must *learn* when to spread vs. drill (this depends on the
        # hidden learning style / forgetting, which a fixed rule cannot infer).
        # Massing is instead governed by the student model + reward.
        self.max_consecutive_same_topic = 0

        self.rng = np.random.default_rng(seed)
        if students is None:
            students = sim.create_student_population(variants_per_ability=4, seed=seed)
        self.students = students

        self.n_scalar = len(self.SCALAR_FEATURES)
        # Action space + per-topic observation are over the ACTIVE CHAPTER
        # (n_active_topics).  Action k -> chapter slot k -> global topic
        # self.active_idx[k].  This removes the 35/45 invalid-action exploration
        # waste that crippled the old Discrete(45) design.
        self._n_act = int(min(n_active_topics, self.n_topics))
        self.obs_dim = self.n_scalar + len(self.TOPIC_BLOCKS) * self._n_act

        self.action_space = spaces.Discrete(self._n_act)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.obs_dim,), dtype=np.float32
        )

        self._init_state()

    # ------------------------------------------------------------------ state
    def _init_state(self) -> None:
        self.current_step = 0
        self.student: dict[str, Any] = sim.create_student("placeholder", ability=sim.MIN_ABILITY, rng=self.rng)
        self.effective_ability = float(sim.MIN_ABILITY)
        self.initial_effective_ability = float(sim.MIN_ABILITY)

        self.asked_mask = np.zeros(self.n_questions, dtype=np.float32)  # over all bank questions (675)
        self.topic_correct = np.zeros(self.n_topics, dtype=np.float32)
        self.topic_wrong = np.zeros(self.n_topics, dtype=np.float32)
        self.topic_asked = np.zeros(self.n_topics, dtype=np.float32)
        # step index at which each topic was last practiced (-1 = never)
        self.topic_last_practiced = np.full(self.n_topics, -1, dtype=np.int64)
        # active "chapter" for this episode (set in reset); default = all topics
        self.active_idx = np.arange(self.n_topics, dtype=np.int64)

        self.total_correct = 0
        self.total_wrong = 0
        self.total_time_taken = 0.0
        self.total_agent_reward = 0.0
        self.total_student_reward = 0.0
        self.total_learning_gain = 0.0

        self.consecutive_correct = 0
        self.consecutive_wrong = 0
        self.fast_correct_streak = 0
        self.overload_streak = 0

        self.last_correct = 0.0
        self.last_time_ratio = 1.0
        self.last_topic_idx = -1
        self.topic_selection_history: deque[int] = deque(maxlen=6)

        self.correct_history: deque[float] = deque(maxlen=5)
        self.time_ratio_history: deque[float] = deque(maxlen=5)
        self.episode_rows: list[dict[str, Any]] = []
        self.initial_mean_mastery = 0.0

    # ------------------------------------------------------------------ reset
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._init_state()

        base = self.students[int(self.rng.integers(0, len(self.students)))]
        self.student = sim.ensure_student(base)
        self.student["study_history"] = []
        if self.randomize_initial_ability:
            # continuous starting ability across the (now wider) range
            init_ability = float(self.rng.uniform(sim.MIN_ABILITY, sim.MAX_ABILITY))
        else:
            init_ability = float(self.student["ability"])
        self.effective_ability = init_ability
        self.initial_effective_ability = init_ability
        self.student = sim.with_ability(self.student, self.effective_ability)
        # pick this episode's contiguous "chapter" of active topics (topological
        # order => within-chapter prerequisites mostly precede their dependents).
        n_active = min(self.n_active_topics, self.n_topics)
        start = int(self.rng.integers(0, self.n_topics - n_active + 1))
        self.active_idx = np.arange(start, start + n_active, dtype=np.int64)
        # assume earlier chapters are already done: lift the mastery of any
        # prerequisite that lies OUTSIDE the active chapter so the chapter is
        # actually learnable (chapter-entry topics have ready prerequisites).
        active_topics = {self.topics[i] for i in self.active_idx}
        m = dict(self.student["topic_mastery"])
        for i in self.active_idx:
            for anc in curriculum.all_ancestors(self.topics[i]):
                if anc not in active_topics:
                    m[anc] = max(float(m.get(anc, 0.0)), 0.80)
        self.student["topic_mastery"] = m
        # snapshot the starting mastery: forgetting consolidates back toward this
        # baseline (prior knowledge), not toward zero.
        sim.set_mastery_baseline(self.student)
        self.initial_mean_mastery = self._active_mean_mastery()
        # running baseline for the dense per-step mastery reward.
        self.prev_mastery = self.initial_mean_mastery
        # delta-driven difficulty controller: seed once from starting ability,
        # then nudge by difficulty_delta after every answer (drives question
        # selection instead of scaling effective ability).
        self.target_difficulty = dc.initial_target_difficulty(self.effective_ability)

        return self._get_obs(), self._reset_info()

    def _reset_info(self) -> dict[str, Any]:
        return {
            "student_id": self.student["student_id"],
            "hidden_learning_style": self.student["learning_style"],
            "hidden_initial_effective_ability": round(self.initial_effective_ability, 4),
            "initial_mean_mastery": round(self.initial_mean_mastery, 4),
            "topics": self.topics,
        }

    # ------------------------------------------------- topic -> question rule
    def select_question_in_topic(self, topic_idx: int) -> tuple[int, bool, float]:
        """Rule: pick the question in ``topic`` whose inherent difficulty is
        closest to the delta-driven ``target_difficulty`` controller.  Prefer
        questions not yet asked this episode; if all are used, allow a repeat.

        Returns (global_question_index, was_repeat, target_difficulty).
        """
        topic = self.topics[topic_idx]
        target = float(self.target_difficulty)
        topic_questions = qb.questions_for_topic(topic)

        unasked = [q for q in topic_questions if self.asked_mask[self.qid_to_idx[q["question_id"]]] < 0.5]
        pool = unasked if unasked else topic_questions
        was_repeat = not unasked

        best = min(pool, key=lambda q: abs(float(q["inherent_difficulty"]) - target))
        return self.qid_to_idx[best["question_id"]], was_repeat, target

    # --------------------------------------------------------------- step
    def step(self, action: int):
        action = int(action)
        # ``action`` is a CHAPTER SLOT index (0..n_active_topics-1).  Map it to
        # the global topic.  If the slot's topic is exhausted, redirect to any
        # still-valid slot.
        slot_mask = self.valid_slot_mask()
        invalid_topic = not bool(slot_mask[action])
        if invalid_topic and slot_mask.any():
            action = int(self.rng.choice(np.flatnonzero(slot_mask)))
        topic_idx = int(self.active_idx[action])
        topic = self.topics[topic_idx]

        q_global, was_repeat, target_difficulty = self.select_question_in_topic(topic_idx)
        question = self.questions[q_global]

        # sync student ability for the simulation, then simulate the answer
        self.student = sim.with_ability(self.student, self.effective_ability)
        result = sim.simulate_answer(self.student, question, self.rng)
        is_correct = bool(result["is_correct"])
        time_taken = float(result["time_taken"])
        time_ratio = time_taken / max(float(question["base_time"]), 1e-6)

        # hidden learning update (mastery, transfer, affect) BEFORE bookkeeping
        readiness = sim.prerequisite_readiness(self.student, topic)
        learn = sim.apply_learning_update(self.student, question, is_correct, time_ratio, self.rng)
        self.student = learn["student"]
        # forgetting: every other topic erodes one step toward the floor, so the
        # agent must time revisits (spaced repetition).  The topic just practiced
        # is exempt.  Applied before the boundary reward so mastery reflects decay.
        sim.apply_forgetting(self.student, practiced_topic=topic)

        # update hidden effective ability (used by the difficulty rule)
        prev_ability = self.effective_ability
        self.effective_ability = sim.update_effective_ability(
            self.effective_ability, is_correct, time_ratio,
            float(question["inherent_difficulty"]), consecutive_wrong=self.consecutive_wrong
        )

        # bookkeeping / behavioural counters (shared with the real session)
        self._update_counters(topic_idx, q_global, is_correct, time_ratio, time_taken)
        # advance the delta-driven difficulty controller for the NEXT question
        _prev_tr = self.time_ratio_history[-2] if len(self.time_ratio_history) >= 2 else 1.0
        self.target_difficulty = dc.step_target_difficulty(
            self.target_difficulty, is_correct, time_ratio, _prev_tr,
            self.consecutive_correct, self.consecutive_wrong)
        self.total_learning_gain += float(learn["total_learning_gain"])

        # reward: tiny per-step shaping now, sparse true-mastery reward at the
        # sub-episode boundary / episode end (computed after the step counter is
        # advanced, below).
        shaping, reward_parts, student_reward = self._step_shaping(
            readiness=readiness,
            was_repeat=was_repeat,
            invalid_topic=invalid_topic,
            is_correct=is_correct,
            time_ratio=time_ratio,
        )
        self.total_student_reward += student_reward

        # advance the step counter, then pay the true-mastery reward.
        step_idx = self.current_step
        sub_ep = self.current_step // self.sub_episode_length
        self.current_step += 1
        terminated = self.current_step >= self.episode_length
        truncated = False

        # DENSE true-mastery reward: the change in the student's true latent mean
        # mastery on THIS step (the practiced topic's gain/transfer minus the
        # forgetting decay of every idle topic).  This telescopes to the total
        # episode mastery gain, so it is the genuine learning objective - not a
        # farmable proxy (random scores <= 0 on it).  Dense delivery gives the
        # per-step credit a sparse boundary reward could not.
        now_mastery = self._active_mean_mastery()
        mastery_reward = self.reward_mastery_coef * (now_mastery - self.prev_mastery)
        self.prev_mastery = now_mastery
        terminal_reward = (self.reward_terminal_coef * self._topics_mastered_fraction()
                           if terminated else 0.0)

        agent_reward = shaping + mastery_reward + terminal_reward
        reward_parts["mastery"] = round(mastery_reward, 4)
        reward_parts["terminal"] = round(terminal_reward, 4)
        self.total_agent_reward += agent_reward

        row = {
            "step": step_idx,
            "sub_episode": sub_ep,
            "step_in_sub": step_idx % self.sub_episode_length,
            "selected_topic": topic,
            "selected_topic_idx": topic_idx,
            "question_id": question["question_id"],
            "inherent_difficulty": round(float(question["inherent_difficulty"]), 3),
            "target_difficulty": round(target_difficulty, 3),
            "was_repeat": was_repeat,
            "invalid_topic_redirect": invalid_topic,
            "is_correct": is_correct,
            "time_taken": round(time_taken, 2),
            "time_ratio": round(time_ratio, 3),
            "effective_ability_before": round(prev_ability, 4),
            "effective_ability_after": round(self.effective_ability, 4),
            "prerequisite_readiness": learn["prerequisite_readiness"],
            "style_multiplier": learn["style_multiplier"],
            "topic_gain": learn["topic_gain"],
            "transfer_gain": learn["transfer_gain"],
            "total_learning_gain": learn["total_learning_gain"],
            "agent_reward": round(agent_reward, 4),
            "student_reward": round(student_reward, 4),
            **{f"r_{k}": round(v, 4) for k, v in reward_parts.items()},
            "acting_mode": result["acting_mode"],
            "hidden_learning_style": self.student["learning_style"],
        }
        self.episode_rows.append(row)

        info = dict(row)
        info["selected_action"] = topic_idx
        info["sub_episode_boundary"] = (self.current_step % self.sub_episode_length == 0) and not terminated
        info["mean_mastery"] = round(sim.mean_mastery(self.student), 4)
        info["accuracy"] = self._safe_accuracy()
        info["total_agent_reward"] = round(self.total_agent_reward, 4)
        info["total_student_reward"] = round(self.total_student_reward, 4)
        info["total_learning_gain"] = round(self.total_learning_gain, 4)
        if terminated:
            info["topic_report"] = self.get_topic_report()
            info["suggested_learning_path"] = self.suggest_learning_path()
            info["episode_rows"] = self.episode_rows
            # mastery metrics are over the ACTIVE chapter (what this episode worked on)
            final_active = self._active_mean_mastery()
            info["initial_mean_mastery"] = round(self.initial_mean_mastery, 4)
            info["final_mean_mastery"] = round(final_active, 4)
            info["mastery_improvement"] = round(final_active - self.initial_mean_mastery, 4)

        return self._get_obs(), float(agent_reward), terminated, truncated, info

    # ----------------------------------------------- shared counter update
    def _update_counters(self, topic_idx, q_global, is_correct, time_ratio, time_taken) -> None:
        """Update behavioural counters from one answered question.

        Used by both ``step`` (simulated answer) and ``apply_external_answer``
        (real human answer), so the observation is built identically in training
        and in the real student session.
        """
        self.asked_mask[q_global] = 1.0
        self.topic_asked[topic_idx] += 1
        if is_correct:
            self.topic_correct[topic_idx] += 1
            self.total_correct += 1
            self.consecutive_correct += 1
            self.consecutive_wrong = 0
        else:
            self.topic_wrong[topic_idx] += 1
            self.total_wrong += 1
            self.consecutive_wrong += 1
            self.consecutive_correct = 0
        self.fast_correct_streak = self.fast_correct_streak + 1 if (is_correct and time_ratio < 0.80) else 0
        overloaded = (not is_correct and time_ratio > 1.80) or time_ratio > 3.00
        self.overload_streak = self.overload_streak + 1 if overloaded else 0

        self.correct_history.append(float(is_correct))
        self.time_ratio_history.append(float(time_ratio))
        self.total_time_taken += float(time_taken)

        self.last_correct = float(is_correct)
        self.last_time_ratio = float(time_ratio)
        self.last_topic_idx = topic_idx
        self.topic_last_practiced[topic_idx] = self.current_step
        self.topic_selection_history.append(topic_idx)

    def apply_external_answer(self, topic_idx: int, q_global: int, is_correct: bool, time_taken: float) -> dict[str, Any]:
        """Apply a REAL (human) answer for the real student session.

        Unlike ``step`` this does *not* simulate the answer and does *not* touch
        hidden mastery / reward - it only updates the behavioural counters and
        the effective ability (the quantity the difficulty rule uses), so the
        next observation and the next topic decision reflect the real student.
        """
        question = self.questions[q_global]
        time_ratio = float(time_taken) / max(float(question["base_time"]), 1e-6)
        prev_ability = self.effective_ability
        self._update_counters(topic_idx, q_global, is_correct, time_ratio, time_taken)
        # advance the delta-driven difficulty controller for the NEXT question
        _prev_tr = self.time_ratio_history[-2] if len(self.time_ratio_history) >= 2 else 1.0
        self.target_difficulty = dc.step_target_difficulty(
            self.target_difficulty, is_correct, time_ratio, _prev_tr,
            self.consecutive_correct, self.consecutive_wrong)
        self.effective_ability = sim.update_effective_ability(
            self.effective_ability, is_correct, time_ratio,
            float(question["inherent_difficulty"]), consecutive_wrong=self.consecutive_wrong
        )
        self.current_step += 1
        return {
            "topic": self.topics[topic_idx],
            "question_id": question["question_id"],
            "inherent_difficulty": round(float(question["inherent_difficulty"]), 3),
            "is_correct": bool(is_correct),
            "time_taken": round(float(time_taken), 2),
            "time_ratio": round(time_ratio, 3),
            "effective_ability_before": round(prev_ability, 4),
            "effective_ability_after": round(self.effective_ability, 4),
            "accuracy": self._safe_accuracy(),
        }

    # --------------------------------------------------------------- reward
    def _step_shaping(self, *, readiness, was_repeat, invalid_topic, is_correct, time_ratio):
        """Tiny per-step shaping ONLY.

        The real objective (true mastery improvement) is paid sparsely at
        sub-episode boundaries inside ``step``.  Here we add a small, bounded
        nudge that discourages obviously bad actions (teaching a topic whose
        prerequisites are unmet, wasting a turn on a repeat/invalid pick) so the
        agent gets a faint early gradient.  It is deliberately too small to farm
        into a high return on its own.
        """
        shaping = 0.0
        # gentle pull toward respecting prerequisites: in [-0.5, +0.5]
        shaping += (readiness - 0.50)
        if was_repeat:
            shaping -= 0.5    # had to reuse a question (topic exhausted)
        if invalid_topic:
            shaping -= 0.5    # chose an unavailable topic (redirected)
        shaping = float(clip(shaping, -1.0, 1.0)) * self.reward_shaping_coef

        # student_reward: outcome quality, for logging/analysis only
        if is_correct:
            student_reward = 2.0 if time_ratio <= 1.5 else (1.5 if time_ratio <= 2.2 else 1.0)
        else:
            student_reward = 0.0

        parts = {"shaping": round(shaping, 4)}
        return shaping, parts, student_reward

    def _active_mean_mastery(self) -> float:
        """Mean TRUE mastery over this episode's active chapter."""
        m = self.student.get("topic_mastery", {})
        return float(np.mean([float(m.get(self.topics[i], 0.0)) for i in self.active_idx]))

    def _topics_mastered_fraction(self) -> float:
        """Fraction of the ACTIVE chapter driven to mastery."""
        m = self.student.get("topic_mastery", {})
        n = sum(1 for i in self.active_idx if float(m.get(self.topics[i], 0.0)) >= self.mastery_threshold)
        return n / float(len(self.active_idx))

    # --------------------------------------------------------------- masks
    def valid_slot_mask(self) -> np.ndarray:
        """Boolean mask over chapter SLOTS (length n_active_topics).

        A slot is valid iff its global topic still has an unasked question.  This
        is what the trained policy sees; ``valid_topic_mask`` (global, 45-wide)
        is kept as a debugging convenience.
        """
        full = self.valid_topic_mask()
        return full[self.active_idx]

    def valid_topic_mask(self) -> np.ndarray:
        """A topic is valid iff it still has at least one unasked question.

        The mask now constrains ONLY truly invalid actions (exhausted topics).
        Behaviour rules such as anti-massing are deliberately NOT baked in: when
        to spread vs. drill depends on the student's hidden learning style and
        forgetting, so it must be *learned* through the reward, not handed to the
        agent for free.  ``max_consecutive_same_topic`` is retained but defaults
        to 0 (disabled).
        """
        mask = np.zeros(self.n_topics, dtype=bool)
        for i, topic in enumerate(self.topics):
            for q in qb.questions_for_topic(topic):
                if self.asked_mask[self.qid_to_idx[q["question_id"]]] < 0.5:
                    mask[i] = True
                    break

        # restrict to this episode's active chapter
        active = np.zeros(self.n_topics, dtype=bool)
        active[self.active_idx] = True
        mask &= active

        cap = int(getattr(self, "max_consecutive_same_topic", 0))
        if cap > 0 and len(self.topic_selection_history) >= cap:
            recent = list(self.topic_selection_history)[-cap:]
            if len(set(recent)) == 1:
                blocked = recent[0]
                if mask[blocked] and mask.sum() > 1:
                    mask[blocked] = False

        if not mask.any():
            mask[self.active_idx] = True
        return mask

    # --------------------------------------------------------------- obs
    def _recent(self, hist: deque, n: int) -> float:
        vals = list(hist)[-n:]
        return float(np.mean(vals)) if vals else 0.0

    def _safe_accuracy(self) -> float:
        total = self.total_correct + self.total_wrong
        return float(self.total_correct / total) if total else 0.0

    def _observed_prereq_ready(self) -> np.ndarray:
        """For each topic: fraction of its prerequisites that have been *seen and
        passed* (observed accuracy >= 0.5).  Structural+behavioural only - no
        hidden ability/mastery is exposed."""
        ready = np.zeros(self.n_topics, dtype=np.float32)
        for i, topic in enumerate(self.topics):
            prereqs = curriculum.prerequisites_of(topic)
            if not prereqs:
                ready[i] = 1.0
                continue
            satisfied = 0
            for p in prereqs:
                pi = self.topic_to_idx[p]
                attempts = self.topic_asked[pi]
                if attempts > 0 and (self.topic_correct[pi] / attempts) >= 0.5:
                    satisfied += 1
            ready[i] = satisfied / len(prereqs)
        return ready

    def _get_obs(self) -> np.ndarray:
        answered = self.total_correct + self.total_wrong
        last4_acc = self._recent(self.correct_history, 4)
        last4_tr = self._recent(self.time_ratio_history, 4)
        struggling = float(
            self.consecutive_wrong >= 2 or self.overload_streak >= 1 or (answered >= 3 and last4_acc <= 0.40)
        )
        ready_adv = float(answered >= 3 and last4_acc >= 0.80 and last4_tr <= 0.60)

        # behavioural style cues from the agent's own recent selections: do the
        # last two picks repeat the same topic / stay in the same cluster?  These
        # let the agent correlate its sequencing pattern with observed outcomes
        # and thereby *infer* the hidden learning style.  The style itself is
        # never revealed.
        repeat_last = 0.0
        same_cluster = 0.0
        if len(self.topic_selection_history) >= 2:
            a, b = self.topic_selection_history[-1], self.topic_selection_history[-2]
            repeat_last = float(a == b)
            same_cluster = float(
                curriculum.TOPIC_TO_CLUSTER.get(self.topics[a])
                == curriculum.TOPIC_TO_CLUSTER.get(self.topics[b])
            )

        scalars = np.array(
            [
                self.current_step / max(1, self.episode_length),
                (self.current_step // self.sub_episode_length) / max(1, self.n_sub_episodes),
                (self.current_step % self.sub_episode_length) / max(1, self.sub_episode_length),
                self.last_correct,
                clip(self.last_time_ratio / 3.0, 0.0, 1.0),
                self._recent(self.correct_history, 3),
                last4_acc,
                clip(self._recent(self.time_ratio_history, 3) / 3.0, 0.0, 1.0),
                clip(last4_tr / 3.0, 0.0, 1.0),
                self._safe_accuracy(),
                clip(self._recent(self.time_ratio_history, 5) / 3.0, 0.0, 1.0),
                clip(self.consecutive_correct / 5.0, 0.0, 1.0),
                clip(self.consecutive_wrong / 5.0, 0.0, 1.0),
                clip(self.fast_correct_streak / 5.0, 0.0, 1.0),
                clip(self.overload_streak / 5.0, 0.0, 1.0),
                struggling,
                ready_adv,
                repeat_last,
                same_cluster,
            ],
            dtype=np.float32,
        )

        # All per-topic feature blocks are emitted over the ACTIVE CHAPTER
        # (length n_active_topics), in slot order, so block[k] aligns with
        # action k.  This is the alignment that lets PPO learn quickly: the
        # policy doesn't have to discover the obs<->action mapping.
        idx = self.active_idx
        attempts_full = self.topic_asked
        attempts = attempts_full[idx]
        topic_accuracy = np.divide(self.topic_correct[idx], np.maximum(attempts, 1.0))
        topic_wrong_rate = np.divide(self.topic_wrong[idx], np.maximum(attempts, 1.0))
        topic_asked_rate = np.clip(attempts / max(1, self.sub_episode_length), 0.0, 1.0)
        topic_seen = (attempts > 0).astype(np.float32)
        prereq_ready = self._observed_prereq_ready()[idx]
        available = self.valid_slot_mask().astype(np.float32)
        # recency: normalised steps since each slot was last practiced.  Seen
        # slots range 0 (just practiced) -> 1 (stale); never-practiced -> 1.0.
        horizon = float(max(1, self.sub_episode_length))
        last_prac = self.topic_last_practiced[idx]
        recency = np.where(
            last_prac >= 0,
            np.clip((self.current_step - last_prac) / horizon, 0.0, 1.0),
            1.0,
        ).astype(np.float32)

        return np.concatenate(
            [
                scalars,
                topic_asked_rate.astype(np.float32),
                topic_accuracy.astype(np.float32),
                topic_wrong_rate.astype(np.float32),
                topic_seen,
                prereq_ready,
                available,
                recency,
            ]
        ).astype(np.float32)

    # --------------------------------------------------------------- reports
    def get_topic_report(self) -> dict[str, dict[str, Any]]:
        report = {}
        for topic, idx in self.topic_to_idx.items():
            correct = int(self.topic_correct[idx])
            wrong = int(self.topic_wrong[idx])
            attempts = correct + wrong
            if attempts == 0:
                accuracy, status = None, "unseen"
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

    def suggest_learning_path(self, max_topics: int = 8) -> list[str]:
        """Order weak/unseen topics so that prerequisites come first (topo order)."""
        report = self.get_topic_report()
        topo = curriculum.topological_order()
        weak = [t for t in topo if report[t]["status"] in ("needs_improvement", "developing")]
        unseen = [t for t in topo if report[t]["status"] == "unseen"]
        return (weak + unseen)[:max_topics]

    def render(self):
        print(
            f"step {self.current_step}/{self.episode_length} "
            f"acc={self._safe_accuracy():.2f} eff_ability(hidden)={self.effective_ability:.2f} "
            f"mastery(hidden)={sim.mean_mastery(self.student):.3f} "
            f"reward={self.total_agent_reward:.2f}"
        )


def make_env(
    students: list[dict[str, Any]] | None = None,
    *,
    sub_episode_length: int = 20,
    n_sub_episodes: int = 4,
    seed: int | None = None,
    randomize_initial_ability: bool = True,
) -> TopicSelectionMCQEnv:
    env = TopicSelectionMCQEnv(
        students=students,
        sub_episode_length=sub_episode_length,
        n_sub_episodes=n_sub_episodes,
        seed=seed,
        randomize_initial_ability=randomize_initial_ability,
    )
    env.action_space.seed(seed)
    return env


if __name__ == "__main__":
    env = make_env(seed=7)
    obs, info = env.reset(seed=7)
    print("obs_dim:", obs.shape[0], "| n_topics(actions):", env.action_space.n)
    print("episode_length:", env.episode_length, "(", env.n_sub_episodes, "x", env.sub_episode_length, ")")
    print("hidden style:", info["hidden_learning_style"], "| init ability(hidden):", info["hidden_initial_effective_ability"])
    done = False
    asked = []
    while not done:
        mask = env.valid_topic_mask()
        a = int(np.random.choice(np.flatnonzero(mask)))
        obs, r, term, trunc, info = env.step(a)
        asked.append(info["question_id"])
        done = term or trunc
    print("steps:", len(asked), "| unique questions:", len(set(asked)))
    print("final accuracy:", round(info["accuracy"], 3))
    print("mastery improvement:", info["mastery_improvement"])
    print("suggested path:", info["suggested_learning_path"][:5])
